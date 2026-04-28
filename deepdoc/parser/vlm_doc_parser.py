# -*- coding: utf-8 -*-
#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
VLMDocParser: VLM-based multimodal document parser.

Refactored from doc_extraction.py. Accepts file bytes, calls VLM APIs directly
(OpenAI-compatible), and returns structured (markdown_str, figures_dict, captions_dict).

api_config dict keys:
    vlm_api_key      - API key for the vision-language model (layout parsing + captioning)
    vlm_api_url      - Base URL for the VLM endpoint
    vlm_modelid      - Model ID for the VLM
    markdown_api_key - API key for the chat model (HTML → Markdown conversion)
    markdown_api_url - Base URL for the chat model endpoint
    markdown_modelid - Model ID for the chat model
"""

import base64
import logging
import math
import os
import re
import tempfile
import time
from io import BytesIO
from pathlib import Path

import fitz
import requests
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup, Tag
from openai import OpenAI

logger = logging.getLogger(__name__)
FIGURE_TOKEN_PREFIX = "VLM_FIG_TOKEN_"
_ZERO_WIDTH_AND_BIDI_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e]")
_NOISE_SEPARATOR_RE = re.compile(r"^\s*(?:[=~\-*_#/\\|丨·•▪▫‣⁃]{3,}|#{3,}|-{3,}|_{3,}|\*{3,}|~{3,})\s*$")
_NOISE_FRAGMENT_PATTERNS = (
    re.compile(r"(?<!\w)#ID#\d+(?:=+)?(?!\w)", re.IGNORECASE),
    re.compile(r"(?<!\w)==+\d+==+(?!\w)"),
    re.compile(r"(?<!\w)\d+==+(?!\w)"),
    re.compile(r"(?<!\w)[①②③④⑤⑥⑦⑧⑨⑩ⓘ〇○●◆◇■□▲△▶▷◀◁★☆✦✧❖❀❁❂❃❉❋✪✳✴➤➜➝➞➟➠](?!\w)"),
    re.compile(r"(?<!\w)[·•▪▫‣⁃](?!\w)"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning("Invalid %s value, fallback to %s.", name, default)
        return default


VLM_LAYOUT_TIMEOUT_SECONDS = _env_int("RAGFLOW_VLM_LAYOUT_TIMEOUT_SECONDS", 300)
VLM_MARKDOWN_TIMEOUT_SECONDS = _env_int("RAGFLOW_VLM_MARKDOWN_TIMEOUT_SECONDS", 300)


def _retry(func, wait_sec=5, max_retries=5, name="API call"):
    """Retry wrapper for flaky API calls."""
    for attempt in range(1, max_retries + 1):
        try:
            result = func()
            if result:
                return result
            logger.warning("%s attempt %d returned empty, retrying in %ds…", name, attempt, wait_sec)
        except Exception as exc:
            logger.warning("%s attempt %d raised: %s", name, attempt, exc)
        time.sleep(wait_sec)
    logger.error("%s failed after %d retries.", name, max_retries)
    return None


def _compress_image(img_path: str, max_width=2500, max_height=3000, quality=70):
    """Resize and compress a JPEG on disk to fit within max dimensions."""
    with Image.open(img_path) as image:
        orig_w, orig_h = image.width, image.height
        if orig_w > max_width or orig_h > max_height:
            ratio = min(max_width / orig_w, max_height / orig_h, 1.0)
            new_size = (int(orig_w * ratio), int(orig_h * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            image.save(img_path, format="JPEG", quality=quality, optimize=True)


def _round_by_factor(value: int, factor: int) -> int:
    return max(factor, round(value / factor) * factor)


def _ceil_by_factor(value: float, factor: int) -> int:
    return max(factor, math.ceil(value / factor) * factor)


def _floor_by_factor(value: float, factor: int) -> int:
    return max(factor, math.floor(value / factor) * factor)


def _smart_resize_local(height: int, width: int,
                        min_pixels: int,
                        max_pixels: int,
                        factor: int = 32) -> tuple[int, int]:
    """
    Local fallback for qwen_vl_utils.smart_resize.
    Returns resized (height, width) in model input space.
    """
    if height <= 0 or width <= 0:
        return height, width

    resized_height = _round_by_factor(height, factor)
    resized_width = _round_by_factor(width, factor)
    pixels = resized_height * resized_width

    if pixels > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / scale, factor)
        resized_width = _floor_by_factor(width / scale, factor)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * scale, factor)
        resized_width = _ceil_by_factor(width * scale, factor)

    return resized_height, resized_width


def _encode_image(image_path: str) -> str:
    """Return base64-encoded image content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _pdf_to_images(pdf_path: str, work_dir: str, zoom=3.0,
                   max_width=2500, max_height=3000,
                   from_page: int = 0, to_page: int = 100000) -> list:
    """Render a page range of a PDF to JPEG files and return their paths."""
    os.makedirs(work_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total = len(doc)
    start = max(0, from_page)
    end = min(total, to_page)
    paths = []
    for i in range(start, end):
        page = doc[i]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = os.path.join(work_dir, f"page_{i + 1}.jpeg")
        pix.save(img_path)
        _compress_image(img_path, max_width=max_width, max_height=max_height)
        paths.append(img_path)
    return paths


def _crop_figures(image_path: str, layout_html: str, fig_dir: str,
                  page_num: int) -> tuple[list, str]:
    """
    Parse bounding boxes from layout HTML, crop figure regions from the page
    image, and save them to fig_dir.

    Returns
    -------
    (saved, filtered_layout_html)
        saved contains accepted (fig_key, fig_path) pairs.
        filtered_layout_html removes rejected image tags so markdown conversion
        sees the same figure set as the cropper.
    """
    def _box_area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _intersection(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def _iou(box1, box2):
        inter = _intersection(box1, box2)
        if inter <= 0:
            return 0.0
        union = _box_area(box1) + _box_area(box2) - inter
        return inter / max(union, 1)

    def _is_contained(inner, outer, threshold=0.92):
        inner_area = _box_area(inner)
        if inner_area <= 0:
            return False
        return _intersection(inner, outer) / inner_area >= threshold

    def _is_false_positive(crop_img: Image.Image, box, page_size) -> bool:
        crop_w, crop_h = crop_img.size
        page_w, page_h = page_size
        if crop_w < 48 or crop_h < 48:
            return True

        aspect_ratio = max(crop_w / max(crop_h, 1), crop_h / max(crop_w, 1))
        if aspect_ratio >= 14:
            return True

        if crop_h < 96 and crop_w > crop_h * 5:
            return True

        if _box_area(box) / max(page_w * page_h, 1) < 0.0025:
            return True

        gray = crop_img.convert("L")
        stat = ImageStat.Stat(gray)
        mean = stat.mean[0]
        stddev = stat.stddev[0]
        min_pixel, max_pixel = gray.getextrema()

        if mean > 245 and stddev < 8:
            return True
        if (max_pixel - min_pixel) < 12:
            return True

        return False

    def _is_non_figure_context(context_text: str, box, page_size) -> bool:
        if not context_text:
            return False

        normalized = " ".join(context_text.split())
        if _TABLE_CONTEXT_PATTERN.search(normalized):
            return True

        box_w = max(0, box[2] - box[0])
        box_h = max(0, box[3] - box[1])
        page_w, page_h = page_size
        has_figure_marker = bool(_FIGURE_CONTEXT_PATTERN.search(normalized))

        # A wide, shallow strip without any figure marker is typically a
        # caption band or plain text rather than a standalone figure.
        if not has_figure_marker and box_h / max(page_h, 1) < 0.12 and box_w / max(page_w, 1) > 0.35:
            return True

        return False
    os.makedirs(fig_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    soup = BeautifulSoup(layout_html, "html.parser")
    image_tags = soup.find_all("div", class_="image")

    saved = []
    accepted_boxes = []
    for tag in image_tags:
        bbox_str = tag.get("data-bbox", "")
        if not bbox_str:
            tag.decompose()
            continue
        try:
            x1, y1, x2, y2 = map(int, bbox_str.split())
        except ValueError:
            tag.decompose()
            continue

        # bbox coords are in the VLM's resized space; the image on disk may
        # have different dimensions – use the original image size directly.
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        width = x2 - x1
        height = y2 - y1
        pad = max(8, min(24, int(min(width, height) * 0.05)))
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

        # Clamp to image bounds
        x1 = max(0, min(x1, orig_w - 1))
        x2 = max(0, min(x2, orig_w))
        y1 = max(0, min(y1, orig_h - 1))
        y2 = max(0, min(y2, orig_h))

        box = (x1, y1, x2, y2)
        if x2 <= x1 or y2 <= y1:
            tag.decompose()
            continue

        duplicate = False
        for kept_box in accepted_boxes:
            if _iou(box, kept_box) >= 0.82 or _is_contained(box, kept_box) or _is_contained(kept_box, box):
                duplicate = True
                break
        if duplicate:
            tag.decompose()
            continue

        context_text = _get_context(soup, tag, max_chars=240)
        if _is_non_figure_context(context_text, box, (orig_w, orig_h)):
            tag.decompose()
            continue

        crop = image.crop((x1, y1, x2, y2))
        if _is_false_positive(crop, box, (orig_w, orig_h)):
            tag.decompose()
            continue

        accepted_boxes.append(box)
        tag["data-bbox"] = f"{x1} {y1} {x2} {y2}"
        fig_key = f"p{page_num}_{len(saved) + 1}"
        fig_path = os.path.join(fig_dir, f"{fig_key}.jpg")
        crop.save(fig_path)
        saved.append((fig_key, fig_path))

    return saved, str(soup)


def _extract_image_blocks(layout_html: str) -> list:
    """
    Mirror doc_extraction.draw_bbox(): collect all elements with data-bbox
    whose class list contains "image".
    """
    soup = BeautifulSoup(layout_html, "html.parser")
    elements = soup.find_all(attrs={"data-bbox": True})
    image_blocks = []
    for el in elements:
        bbox_str = el.get("data-bbox", "")
        if not bbox_str:
            continue
        try:
            x1, y1, x2, y2 = map(int, bbox_str.split())
        except ValueError:
            continue
        class_list = el.get("class", [])
        if "image" in class_list:
            image_blocks.append(((x1, y1, x2, y2), el))
    return image_blocks


def _parse_bbox(bbox_str: str) -> tuple[int, int, int, int] | None:
    try:
        x1, y1, x2, y2 = map(int, bbox_str.split())
    except ValueError:
        return None
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection(box1: tuple[int, int, int, int],
                       box2: tuple[int, int, int, int]) -> int:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _bbox_overlap_ratio(box1: tuple[int, int, int, int],
                        box2: tuple[int, int, int, int]) -> float:
    inter = _bbox_intersection(box1, box2)
    if inter <= 0:
        return 0.0
    return inter / max(min(_bbox_area(box1), _bbox_area(box2)), 1)


def _crop_figures_doc_extraction(image_path: str, layout_html: str, fig_dir: str,
                                 page_num: int) -> list:
    """
    Faithful port of doc_extraction.draw_bbox() cropping behavior:
    no dedupe, no semantic filtering, no heuristic rejection.
    """
    os.makedirs(fig_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_blocks = _extract_image_blocks(layout_html)

    saved = []
    for idx, (bbox, _) in enumerate(image_blocks):
        x1, y1, x2, y2 = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        pad_x = max(12, min(48, int(box_w * 0.04)))
        pad_top = max(12, min(48, int(box_h * 0.08)))
        pad_bottom = max(18, min(72, int(box_h * 0.14)))

        x1 = max(0, x1 - pad_x)
        x2 = min(image.width, x2 + pad_x)
        y1 = max(0, y1 - pad_top)
        y2 = min(image.height, y2 + pad_bottom)

        crop = image.crop((x1, y1, x2, y2))
        fig_key = f"p{page_num}_{idx + 1}"
        fig_path = os.path.join(fig_dir, f"{fig_key}.jpg")
        crop.save(fig_path)
        saved.append((fig_key, fig_path))

    return saved


def _save_layout_debug(image_path: str, layout_html: str, debug_dir: str, page_num: int,
                       name_suffix: str = "") -> None:
    """
    Save raw layout HTML and a page image with bbox overlays for debugging.
    This mirrors doc_extraction.py's bbox visualization, but is optional and
    only used by test/debug flows.
    """
    os.makedirs(debug_dir, exist_ok=True)

    suffix = f"_{name_suffix}" if name_suffix else ""
    html_path = os.path.join(debug_dir, f"page_{page_num}_layout{suffix}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(layout_html)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    soup = BeautifulSoup(layout_html, "html.parser")
    elements = soup.find_all(attrs={"data-bbox": True})

    for el in elements:
        bbox_str = el.get("data-bbox", "")
        if not bbox_str:
            continue
        try:
            x1, y1, x2, y2 = map(int, bbox_str.split())
        except ValueError:
            continue

        class_list = el.get("class", [])
        is_image = "image" in class_list
        outline = "red" if is_image else "blue"
        line_width = 4 if is_image else 2

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        draw.rectangle((x1, y1, x2, y2), outline=outline, width=line_width)

    bbox_path = os.path.join(debug_dir, f"page_{page_num}_bbox{suffix}.jpeg")
    image.save(bbox_path, format="JPEG")


def _save_resize_debug(debug_dir: str, page_num: int,
                       original_width: int, original_height: int,
                       input_width: int, input_height: int) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    suffix = "" if (original_width, original_height) == (input_width, input_height) else "_resized"
    meta_path = os.path.join(debug_dir, f"page_{page_num}_resize{suffix}.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"original_width={original_width}\n")
        f.write(f"original_height={original_height}\n")
        f.write(f"input_width={input_width}\n")
        f.write(f"input_height={input_height}\n")
        if original_width and original_height:
            f.write(f"scale_x={input_width / original_width:.6f}\n")
            f.write(f"scale_y={input_height / original_height:.6f}\n")
        if input_width and input_height:
            f.write(f"restore_scale_x={original_width / input_width:.6f}\n")
            f.write(f"restore_scale_y={original_height / input_height:.6f}\n")


def _rescale_layout_bboxes(layout_html: str,
                           source_width: int,
                           source_height: int,
                           target_width: int,
                           target_height: int) -> str:
    """
    Convert data-bbox coordinates from the smart_resize input space back to the
    original rendered page image space using simple linear scaling.
    """
    if source_width <= 0 or source_height <= 0:
        return layout_html
    if source_width == target_width and source_height == target_height:
        return layout_html

    scale_x = source_width / target_width
    scale_y = source_height / target_height
    soup = BeautifulSoup(layout_html, "html.parser")

    for tag in soup.find_all(attrs={"data-bbox": True}):
        bbox = _parse_bbox(tag.get("data-bbox", ""))
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        x1 = int(round(x1 / scale_x))
        y1 = int(round(y1 / scale_y))
        x2 = int(round(x2 / scale_x))
        y2 = int(round(y2 / scale_y))

        x1 = max(0, min(x1, target_width))
        x2 = max(0, min(x2, target_width))
        y1 = max(0, min(y1, target_height))
        y2 = max(0, min(y2, target_height))
        tag["data-bbox"] = f"{x1} {y1} {x2} {y2}"

    return str(soup)


def _normalize_layout_html(layout_html: str) -> str:
    """
    Remove duplicated top-level blocks from the same page so downstream stages
    do not amplify repeated layout output from the VLM.
    """
    soup = BeautifulSoup(layout_html, "html.parser")
    container = soup.body if soup.body else soup
    seen_signatures = set()
    seen_image_bboxes = set()
    table_boxes = []

    for tag in soup.find_all(attrs={"data-bbox": True}):
        class_list = tag.get("class", [])
        if "table" not in class_list:
            continue
        bbox = _parse_bbox(tag.get("data-bbox", ""))
        if bbox:
            table_boxes.append(bbox)

    for child in list(container.children):
        if isinstance(child, str):
            if not child.strip():
                continue
            signature = ("text", " ".join(child.split()))
            if signature in seen_signatures:
                child.extract()
                continue
            seen_signatures.add(signature)
            continue

        if not isinstance(child, Tag):
            continue

        class_list = tuple(child.get("class", []))
        bbox = child.get("data-bbox", "")
        normalized_text = " ".join(child.get_text(" ", strip=True).split())
        parsed_bbox = _parse_bbox(bbox) if bbox else None

        if "image" in class_list and parsed_bbox:
            overlaps_table = any(
                _bbox_overlap_ratio(parsed_bbox, table_box) >= 0.72
                for table_box in table_boxes
            )
            if overlaps_table:
                child.decompose()
                continue

        if "image" in class_list and bbox:
            if bbox in seen_image_bboxes:
                child.decompose()
                continue
            seen_image_bboxes.add(bbox)

        signature = (child.name, class_list, bbox, normalized_text)
        if signature in seen_signatures:
            child.decompose()
            continue
        seen_signatures.add(signature)

    return str(soup)


def _inject_figure_tokens(layout_html: str) -> tuple[str, list[str]]:
    """
    Replace each image block in layout_html with a deterministic token so the
    markdown conversion step no longer decides how many figures exist.
    """
    soup = BeautifulSoup(layout_html, "html.parser")
    tokens = []
    elements = soup.find_all(attrs={"data-bbox": True})

    image_tags = []
    for el in elements:
        class_list = el.get("class", [])
        if "image" in class_list:
            image_tags.append(el)

    for idx, tag in enumerate(image_tags, start=1):
        token = f"{FIGURE_TOKEN_PREFIX}{idx}"
        placeholder = soup.new_tag("p")
        placeholder.string = token
        tag.replace_with(placeholder)
        tokens.append(token)

    return str(soup), tokens


def _restore_figure_placeholders(markdown_text: str, tokens: list[str]) -> str:
    """
    Convert deterministic figure tokens back into the placeholder format used
    by the downstream RagFlow integration.
    """
    if not markdown_text:
        return ""

    for token in tokens:
        markdown_text = re.sub(
            rf"(?m)^\s*{re.escape(token)}\s*$",
            "![FIG_PLACEHOLDER](#)",
            markdown_text,
        )
        markdown_text = markdown_text.replace(token, "![FIG_PLACEHOLDER](#)")

    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)
    return markdown_text.strip()


def _clean_html(full_predict: str) -> str:
    """Strip bbox/polygon attrs, normalise class names, return clean HTML block."""
    if "<body" not in full_predict.lower():
        full_predict = f"<html><body>{full_predict}</body></html>"
    soup = BeautifulSoup(full_predict, "html.parser")

    color_pat = re.compile(r"\bcolor:[^;]+;?")
    for tag in soup.find_all(style=True):
        new_style = color_pat.sub("", tag.get("style", "")).strip().rstrip(";")
        if new_style:
            tag["style"] = new_style
        else:
            del tag["style"]

    for attr in ("data-bbox", "data-polygon"):
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]

    fix_classes = {"formula.machine_printed", "formula.handwritten"}
    for tag in soup.find_all(class_=True):
        if isinstance(tag, Tag):
            tag["class"] = list(dict.fromkeys(
                "formula" if c in fix_classes else c for c in tag.get("class", [])))

    for div in soup.find_all("div", class_="image caption"):
        div.clear()
        div["class"] = ["image"]

    for cls in ("music sheet", "chemical formula", "chart"):
        for tag in soup.find_all(class_=cls):
            if isinstance(tag, Tag):
                tag.clear()
                tag.attrs.pop("format", None)

    output = []
    for child in soup.body.children:
        if isinstance(child, Tag):
            output.append(str(child) + "\n")
        elif isinstance(child, str) and child.strip():
            output.append(child.strip() + "\n")
    return f"```html\n<html><body>\n{''.join(output)}</body></html>\n```"


def _normalize_markdown_output(markdown_text: str) -> str:
    if not markdown_text:
        return ""
    markdown_text = re.sub(r"^```markdown\s*", "", markdown_text)
    markdown_text = re.sub(r"\s*```$", "", markdown_text)
    markdown_text = re.sub(
        r"```\nCreated with an evaluation copy of Aspose\..*?\n```",
        "", markdown_text, flags=re.DOTALL)
    markdown_text = re.sub(
        r"Evaluation Only\. Created with Aspose\..*?Pty Ltd\.",
        "", markdown_text, flags=re.MULTILINE)

    # Keep headings and figure placeholders as standalone blocks so downstream
    # section splitting can preserve finer document structure.
    markdown_text = re.sub(r'(?m)^\s*(#{1,6}\s+.+?)\s*$', r'\n\1\n', markdown_text)
    markdown_text = re.sub(
        r'\s*!\[FIG_PLACEHOLDER\]\(#\)\s*',
        '\n\n![FIG_PLACEHOLDER](#)\n\n',
        markdown_text,
    )
    # Apply only conservative, surface-level cleanup so we do not rewrite
    # document semantics. The goal is to remove obvious parser artifacts:
    # zero-width/bidi control chars, broken citation tails, and decorative
    # separator lines that often leak out of VLM markdown output.
    cleaned_lines = []
    in_fence = False
    for raw_line in markdown_text.splitlines():
        line = raw_line
        fence_match = re.match(r"^\s*```", line)
        if fence_match:
            cleaned_lines.append(line.rstrip())
            in_fence = not in_fence
            continue

        if in_fence:
            cleaned_lines.append(line.rstrip())
            continue

        line = _ZERO_WIDTH_AND_BIDI_RE.sub("", line)
        line = line.replace("\ufeff", "").replace("\ufffd", "")
        line = line.replace("\u00a0", " ").replace("\u3000", " ")
        for pattern in _NOISE_FRAGMENT_PATTERNS:
            line = pattern.sub("", line)

        line = re.sub(r"[ \t]{2,}", " ", line)
        line = re.sub(r"\s+([,.;:!?。！？；：、，)\]】》〉」』〕〗］）”’])", r"\1", line)
        line = re.sub(r"([([{（【《〈“‘])\s+", r"\1", line)
        line = re.sub(r"(?:(?<=^)|(?<=[\s，。！？；：,.!?;:]))i(?=(?:[\s，。！？；：,.!?;:]|$))", " ", line)
        line = line.rstrip()
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _NOISE_SEPARATOR_RE.match(stripped):
            cleaned_lines.append("")
            continue
        cleaned_lines.append(line)

    markdown_text = "\n".join(cleaned_lines)
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    return markdown_text.strip()


def _get_context(soup: BeautifulSoup, image_tag: Tag, max_chars=400) -> str:
    """Return up to max_chars of surrounding text for a figure tag."""
    texts = []
    prev, count = image_tag.previous_sibling, 0
    while prev is not None and count < 3:
        if isinstance(prev, Tag):
            t = prev.get_text(strip=True)
            if t:
                texts.insert(0, t)
                count += 1
        prev = prev.previous_sibling

    nxt, count = image_tag.next_sibling, 0
    while nxt is not None and count < 3:
        if isinstance(nxt, Tag):
            t = nxt.get_text(strip=True)
            if t:
                texts.append(t)
                count += 1
        nxt = nxt.next_sibling

    ctx = "\n".join(texts)
    return ctx[:max_chars] if len(ctx) > max_chars else ctx


# ---------------------------------------------------------------------------
# VLMDocParser
# ---------------------------------------------------------------------------

class VLMDocParser:
    """
    Parse a document (PDF / Word / PPT) via a two-stage VLM pipeline and return:
        markdown_str  – full document as Markdown with ![FIG_PLACEHOLDER](#) for figures
        figures_dict  – {"p{page}_{idx}": <bytes>}
        captions_dict – {"p{page}_{idx}": "caption text"}
    """

    def __init__(self, api_config: dict):
        self.vlm_key = api_config.get("vlm_api_key", "")
        self.vlm_url = api_config.get("vlm_api_url", "")
        self.vlm_model = api_config.get("vlm_modelid", "")
        self.md_key = api_config.get("markdown_api_key", "")
        self.md_url = api_config.get("markdown_api_url", "")
        self.md_model = api_config.get("markdown_modelid", "")
        self.debug_dir = api_config.get("debug_dir", "")

        # Reuse clients across calls to reduce connection churn in long-running
        # parsing tasks (many pages => many API requests).
        self._vlm_client = OpenAI(api_key=self.vlm_key, base_url=self.vlm_url) if self.vlm_key and self.vlm_url else None
        self._md_client = OpenAI(api_key=self.md_key, base_url=self.md_url) if self.md_key and self.md_url else None
        self.figure_order = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(self, binary: bytes, file_type: str = "pdf",
                 from_page: int = 0, to_page: int = 100000):
        """
        Parameters
        ----------
        binary     : raw file content
        file_type  : "pdf" | "word" | "ppt"
        from_page  : first page index (0-based, inclusive)
        to_page    : last page index (0-based, exclusive)

        Returns
        -------
        (markdown_str, figures_dict, captions_dict)
        """
        with tempfile.TemporaryDirectory() as work_dir:
            self.figure_order = []
            # Write the binary to a temp file
            suffix = {"pdf": ".pdf", "word": ".docx", "ppt": ".pptx"}.get(file_type, ".pdf")
            tmp_path = os.path.join(work_dir, f"input{suffix}")
            with open(tmp_path, "wb") as f:
                f.write(binary)

            page_images = self._render_pages(tmp_path, work_dir, file_type,
                                             from_page=from_page, to_page=to_page)
            if not page_images:
                logger.error("VLMDocParser: no pages rendered for file_type=%s", file_type)
                return "", {}, {}

            fig_dir = os.path.join(work_dir, "figures")
            all_markdowns = []
            captions_dict = {}
            figures_dict = {}

            for idx, img_path in enumerate(page_images):
                page_num = from_page + idx + 1
                try:
                    markdown_page, page_figs, page_caps, page_order = self._process_page(
                        img_path, fig_dir, page_num)
                    if markdown_page:
                        all_markdowns.append(markdown_page)
                    captions_dict.update(page_caps)
                    figures_dict.update(page_figs)
                    self.figure_order.extend(page_order)
                except Exception as exc:
                    logger.exception("VLMDocParser: page %d failed: %s", page_num, exc)
                    continue

        markdown_str = "\n\n".join(all_markdowns)
        return markdown_str, figures_dict, captions_dict

    # ------------------------------------------------------------------
    # Page rendering
    # ------------------------------------------------------------------

    def _render_pages(self, file_path: str, work_dir: str, file_type: str,
                      from_page: int = 0, to_page: int = 100000) -> list:
        pages_dir = os.path.join(work_dir, "pages")
        if file_type == "pdf":
            return _pdf_to_images(file_path, pages_dir,
                                  from_page=from_page, to_page=to_page)
        elif file_type == "word":
            return self._word_to_images(file_path, pages_dir)
        elif file_type == "ppt":
            return self._ppt_to_images(file_path, pages_dir)
        else:
            logger.warning("VLMDocParser: unsupported file_type '%s', treating as pdf", file_type)
            return _pdf_to_images(file_path, pages_dir)

    def _word_to_images(self, word_path: str, out_dir: str) -> list:
        try:
            import aspose.words as aw
        except ImportError:
            logger.error("aspose.words not installed; Word parsing unavailable")
            return []
        os.makedirs(out_dir, exist_ok=True)
        doc = aw.Document(word_path)
        paths = []
        for i in range(doc.page_count):
            page = doc.extract_pages(i, 1)
            img_path = os.path.join(out_dir, f"page_{i + 1}.jpeg")
            page.save(img_path)
            _compress_image(img_path)
            paths.append(img_path)
        return paths

    def _ppt_to_images(self, ppt_path: str, out_dir: str) -> list:
        try:
            import aspose.slides as slides
        except ImportError:
            logger.error("aspose.slides not installed; PPT parsing unavailable")
            return []
        os.makedirs(out_dir, exist_ok=True)
        prs = slides.Presentation(ppt_path)
        paths = []
        for i, slide in enumerate(prs.slides):
            img_path = os.path.join(out_dir, f"slide_{i + 1}.jpeg")
            slide.get_image(1, 1).save(img_path, slides.ImageFormat.JPEG)
            _compress_image(img_path)
            paths.append(img_path)
        slides.FontsLoader.clear_cache()
        return paths

    # ------------------------------------------------------------------
    # Per-page pipeline
    # ------------------------------------------------------------------

    def _process_page(self, img_path: str, fig_dir: str, page_num: int):
        """
        Run Pipeline A + B on one page image.
        Returns (markdown_str, figures_dict, captions_dict) for this page.
        """
        image = Image.open(img_path)
        width, height = image.size

        min_px = 512 * 28 * 28
        max_px = 2048 * 28 * 28
        input_height, input_width = _smart_resize_local(
            height, width, min_pixels=min_px, max_pixels=max_px, factor=32
        )

        # ---- Pipeline A: layout parsing --------------------------------
        layout_html = self._inference_layout(img_path, min_px, max_px)
        if not layout_html:
            logger.warning("VLMDocParser: page %d layout failed", page_num)
            return None, {}, {}, []

        if self.debug_dir:
            _save_resize_debug(
                self.debug_dir,
                page_num,
                original_width=width,
                original_height=height,
                input_width=input_width,
                input_height=input_height,
            )
            _save_layout_debug(img_path, layout_html, self.debug_dir, page_num)

        layout_html = _rescale_layout_bboxes(
            layout_html,
            source_width=input_width,
            source_height=input_height,
            target_width=width,
            target_height=height,
        )

        if self.debug_dir:
            _save_layout_debug(img_path, layout_html, self.debug_dir, page_num, name_suffix="rescaled")

        layout_html = _normalize_layout_html(layout_html)

        if self.debug_dir:
            _save_layout_debug(img_path, layout_html, self.debug_dir, page_num, name_suffix="normalized")

        # Crop figures from this page
        cropped_figures = _crop_figures_doc_extraction(img_path, layout_html, fig_dir, page_num)

        # Convert HTML → Markdown
        tokenized_layout_html, figure_tokens = _inject_figure_tokens(layout_html)
        cleaned_html = _clean_html(tokenized_layout_html)
        markdown_text = self._html_to_markdown_deterministic(cleaned_html)
        if markdown_text:
            markdown_text = _normalize_markdown_output(markdown_text)
            markdown_text = _restore_figure_placeholders(markdown_text, figure_tokens)

        # ---- Pipeline B: figure captions --------------------------------
        soup = BeautifulSoup(layout_html, "html.parser")
        image_blocks = _extract_image_blocks(layout_html)

        figures_dict = {}
        captions_dict = {}
        figure_order = []
        for img_idx, (_, img_tag) in enumerate(image_blocks):
            if img_idx >= len(cropped_figures):
                figure_order.append(None)
                continue

            fig_key, fig_path = cropped_figures[img_idx]
            if not os.path.exists(fig_path):
                figure_order.append(None)
                continue

            # Read figure bytes for later storage
            with open(fig_path, "rb") as fh:
                figures_dict[fig_key] = fh.read()
            figure_order.append(fig_key)

            ctx = _get_context(soup, img_tag)
            caption = self._inference_caption(fig_path, ctx) or "示意图"
            captions_dict[fig_key] = caption.strip()

        return markdown_text, figures_dict, captions_dict, figure_order

    # ------------------------------------------------------------------
    # VLM API calls
    # ------------------------------------------------------------------

    def _inference_layout(self, image_path: str, min_pixels: int, max_pixels: int):
        """Pipeline A: send full page image to VLM, receive HTML with bboxes."""
        sys_prompt = "You are a helpful document layout extractor."
        user_prompt = (
            "You are an AI specialized in recognizing and extracting the layout of document pages from images. "
            "Your task is to output HTML in the QwenVL Document Parser format.\n"
            "Requirements:\n"
            "1. Preserve the logical reading order and block structure (titles, paragraphs, lists, tables, etc.).\n"
            "2. For every figure or picture, output exactly one block:\n"
            "   <div class=\"image\" data-bbox=\"x1 y1 x2 y2\"></div>\n"
            "   DO NOT put any caption text inside the image div.\n"
            "3. If there is a separate figure caption text, keep it as a normal text block (e.g. <p>图1 ...</p>), "
            "   near its original position.\n"
            "4. Do not summarize or rephrase content; just reflect the structure you see.\n"
            "5. Make sure data-bbox covers the visual area of the image.\n"
            "6. Do not repeat the same image block, caption block, paragraph block, or section block multiple times.\n"
            "7. If one figure appears once on the page, output it exactly once."
        )
        b64 = _encode_image(image_path)

        def _call():
            client = self._vlm_client or OpenAI(api_key=self.vlm_key, base_url=self.vlm_url)
            resp = client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "min_pixels": min_pixels,
                         "max_pixels": max_pixels,
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": user_prompt},
                    ]},
                ],
                timeout=VLM_LAYOUT_TIMEOUT_SECONDS,
            )
            return resp.choices[0].message.content if resp and resp.choices else None

        return _retry(_call, name="VLM layout")

    def _inference_caption(self, fig_path: str, context: str,
                           min_pixels: int = 256 * 28 * 28,
                           max_pixels: int = 1024 * 28 * 28):
        """Pipeline B: caption a single cropped figure image."""
        sys_prompt = "You are an expert figure captioner for scientific and technical documents."
        user_prompt = (
            "You are given a figure image and some nearby text context from a technical document.\n"
            "Your task is to output ONE concise, high-quality figure caption.\n\n"
            "Priority:\n"
            "1. If the context already contains an explicit figure caption (lines starting with "
            "'图', 'Figure', 'Fig.', '表', 'Table', etc.), EXTRACT that caption verbatim. "
            "Do not translate or rewrite explicit captions.\n"
            "2. Otherwise, GENERATE a short informative caption based on the image and context, "
            "using the same language as the nearby context when possible.\n\n"
            "Requirements:\n"
            "- ALWAYS output a meaningful caption sentence.\n"
            "- DO NOT output 'null', 'N/A', 'none', or placeholder text.\n"
            "- Output ONLY the caption text itself.\n\n"
            f"Nearby text context:\n{context}\n"
        )
        b64 = _encode_image(fig_path)

        def _call():
            client = self._vlm_client or OpenAI(api_key=self.vlm_key, base_url=self.vlm_url)
            resp = client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "min_pixels": min_pixels,
                         "max_pixels": max_pixels,
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": user_prompt},
                    ]},
                ],
                timeout=90,
            )
            return resp.choices[0].message.content.strip() if resp and resp.choices else None

        return _retry(_call, name="VLM caption")

    def _html_to_markdown_deterministic(self, html_content: str):
        """Convert cleaned HTML to Markdown while preserving figure tokens."""
        sys_prompt = "You convert structured HTML into clean Markdown."
        user_prompt = (
            "Convert the provided HTML into Markdown.\n"
            "Requirements:\n"
            "1. Preserve the original reading order and document structure.\n"
            "2. Keep headings, paragraphs, lists and tables as faithful as possible.\n"
            "3. Do not summarize, rewrite or drop content.\n"
            "4. If the HTML contains tokens like VLM_FIG_TOKEN_1, keep them verbatim in the Markdown output.\n"
            "5. Do not create or remove figure tokens on your own.\n\n"
            f"HTML input:\n{html_content}"
        )

        def _call():
            client = self._md_client or OpenAI(api_key=self.md_key, base_url=self.md_url)
            resp = client.chat.completions.create(
                model=self.md_model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                    ]},
                ],
                timeout=VLM_MARKDOWN_TIMEOUT_SECONDS,
            )
            return resp.choices[0].message.content if resp and resp.choices else None

        return _retry(_call, name="HTML to Markdown")

    def _html_to_markdown(self, html_content: str):
        """Convert cleaned HTML to Markdown using a chat model."""
        sys_prompt = "你是一个熟练的文档转换助手。"
        user_prompt = (
            "请你将解析得到的html格式文档解析为顺序输出的markdown文档，"
            "同时过滤掉和正文（正文包括标题、作者，摘要，主要文段、参考文献和附录信息）无关的页眉页脚等信息。"
            "如果存在图像, HTML 中只有此 `<div class='image'>` 标签代表图像，请为每个图像生成一条 `![FIG_PLACEHOLDER](#)` 形式的图像占位符，"
            "不需要根据图像内容生成名称；后续会用其他模块来填充真正的题注。"
            "直接输出解析后的markdown文档，不需要任何其他说明。注意，不需要翻译。"
        )

        def _call():
            client = self._md_client or OpenAI(api_key=self.md_key, base_url=self.md_url)
            resp = client.chat.completions.create(
                model=self.md_model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"{user_prompt}\n\n以下是html文档：{html_content}"},
                    ]},
                ],
                timeout=VLM_MARKDOWN_TIMEOUT_SECONDS,
            )
            return resp.choices[0].message.content if resp and resp.choices else None

        return _retry(_call, name="HTML→Markdown")
