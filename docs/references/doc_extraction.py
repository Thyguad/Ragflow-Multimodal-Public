import os
import re
import json
import requests
import base64
import fitz
import time
import argparse
import pandas as pd
import aspose.words as aw
import aspose.slides as slides
from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO
from bs4 import BeautifulSoup, Tag
from pathlib import Path
from qwen_vl_utils import smart_resize

# 载入参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default='/home/baiyipeng/document_extraction/config.json',
    help='Path to config file'
)
args = parser.parse_args()

with open(args.config, 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

VLM_API_KEY = CONFIG["vlm_api_key"]
VLM_API_URL = CONFIG["vlm_api_url"]
VLM_MODELID = CONFIG["vlm_modelid"]
MARKDOWN_API_KEY = CONFIG["markdown_api_key"]
MARKDOWN_API_URL = CONFIG["markdown_api_url"]
MARKDOWN_MODELID = CONFIG["markdown_modelid"]
INPUT = CONFIG["input"]
OUTPUT_ROOT = CONFIG["output_root"]
# FONT_FOLDER = CONFIG["font_folder"]


# API重试机制
def retry_api_call(func, wait_sec=5, max_retries=5, name="API调用"):
    attempt = 1
    while attempt <= max_retries:
        try:
            result = func()
            if result:
                return result
            else:
                print(f"{name} 第 {attempt} 次返回无效结果，等待 {wait_sec}s 后重试...")
        except Exception as e:
            print(f"{name} 第 {attempt} 次发生异常：{e}")
        attempt += 1
        time.sleep(wait_sec)
    print(f"{name} 达到最大重试次数 {max_retries} 次，仍未成功。")
    return None


def get_file_type(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext == ".html":
        return "html"
    elif ext in [".xls", ".xlsx", ".csv"]:
        return "excel"
    elif ext in [".docx", ".doc"]:
        return "word"
    elif ext == ".txt":
        return "txt"
    elif ext in [".ppt", ".pptx"]:
        return "ppt"
    else:
        return "unknown"


# 压缩图片尺寸，符合VLM输入大小
def compress_image(img_path, max_width=2500, max_height=3000, quality=70):
    with Image.open(img_path) as image:
        orig_w, orig_h = image.width, image.height
        if orig_w > max_width or orig_h > max_height:
            ratio_w = max_width / orig_w
            ratio_h = max_height / orig_h
            resize_ratio = min(ratio_w, ratio_h, 1.0)
            new_size = (int(orig_w * resize_ratio), int(orig_h * resize_ratio))

            print(f"压缩 {img_path} 原尺寸: {orig_w}x{orig_h} → 新尺寸: {new_size}")
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            image.save(img_path, format="JPEG", quality=quality, optimize=True)
        else:
            print(f"无需压缩 {img_path} ({orig_w}x{orig_h})")


# 将pdf逐页转化为图像
def pdf_to_image(pdf_path, output_dir, zoom=3.0, max_width=2500, max_height=3000):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = os.path.join(output_dir, f"page_{page_index + 1}.jpeg")
        pix.save(img_path)

        compress_image(img_path, max_width=max_width, max_height=max_height, quality=70)

        file_size = os.path.getsize(img_path) / 1024 / 1024
        print(f"保存第 {page_index + 1} 页: {img_path}({file_size:.2f} MB)")
        image_paths.append(img_path)

    return image_paths


# word 转 image
def word_to_image(word_path, output_dir, max_width=2500, max_height=3000):
    os.makedirs(output_dir, exist_ok=True)
    word_path = str(word_path)
    doc = aw.Document(word_path)
    image_paths = []

    for i in range(doc.page_count):
        page = doc.extract_pages(i, 1)
        img_path = os.path.join(output_dir, f"page_{i + 1}.jpeg")
        page.save(img_path)

        compress_image(img_path, max_width=max_width, max_height=max_height, quality=70)

        file_size = os.path.getsize(img_path) / 1024 / 1024
        print(f"保存第 {i + 1} 页: {img_path} ({file_size:.2f} MB)")
        image_paths.append(img_path)

    return image_paths


# ppt 转 image
def ppt_to_image(ppt_path, output_dir, max_width=2500, max_height=3000):
    os.makedirs(output_dir, exist_ok=True)
    ppt_path = str(ppt_path)
    presentation = slides.Presentation(ppt_path)
    image_paths = []

    for i, slide in enumerate(presentation.slides):
        img_path = os.path.join(output_dir, f"slide_{i + 1}.jpeg")
        slide_image = slide.get_image(1, 1)
        slide_image.save(img_path, slides.ImageFormat.JPEG)

        compress_image(img_path, max_width=max_width, max_height=max_height, quality=70)

        file_size = os.path.getsize(img_path) / 1024 / 1024
        print(f"保存第 {i + 1} 页: {img_path} ({file_size:.2f} MB)")
        image_paths.append(img_path)

    slides.FontsLoader.clear_cache()
    return image_paths


# 绘制bounding box并根据html标签获取页面图像（用于裁剪 figures）
def draw_bbox(image_path, image_save_path, fig_dir, resized_width, resized_height, full_predict, page_num):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    draw = ImageDraw.Draw(image)
    soup = BeautifulSoup(full_predict, 'html.parser')
    elements = soup.find_all(attrs={"data-bbox": True})

    image_blocks = []  # [(bbox, tag)]
    other_blocks = []  # [(bbox, text)]

    for el in elements:
        bbox_str = el['data-bbox']
        # text = element.get_text(strip=True)
        x1, y1, x2, y2 = map(int, bbox_str.split())
        class_list = el.get('class', [])
        text = el.get_text(strip=True)
        if "image" in class_list:
            image_blocks.append(((x1, y1, x2, y2), el))
        elif el.name == 'ol':
            continue
        elif el.name == 'li' and el.parent.name == 'ol':
            other_blocks.append(((x1, y1, x2, y2), text))
        else:
            other_blocks.append(((x1, y1, x2, y2), text))
    # font = ImageFont.truetype("/home/baiyipeng/VLMPDF/NotoSansCJK-Regular.ttc", 20)

    # 保存所有 image_blocks
    for idx, (bbox, _) in enumerate(image_blocks):
        x1, y1, x2, y2 = bbox
        x1_r = int(x1 / scale_x)
        y1_r = int(y1 / scale_y)
        x2_r = int(x2 / scale_x)
        y2_r = int(y2 / scale_y)
        if x1_r > x2_r:
            x1_r, x2_r = x2_r, x1_r
        if y1_r > y2_r:
            y1_r, y2_r = y2_r, y1_r
        crop_img = image.crop((x1_r, y1_r, x2_r, y2_r))
        fig_name = f"p{page_num}_{idx + 1}.jpg"
        save_path = os.path.join(fig_dir, fig_name)
        crop_img.save(save_path)
        print(f"页面图像保存至: {save_path}")

    # 绘制所有bbox区域
    for bbox, _ in other_blocks + image_blocks:
        x1, y1, x2, y2 = bbox
        x1_r = int(x1 / scale_x)
        y1_r = int(y1 / scale_y)
        x2_r = int(x2 / scale_x)
        y2_r = int(y2 / scale_y)
        if x1_r > x2_r:
            x1_r, x2_r = x2_r, x1_r
        if y1_r > y2_r:
            y1_r, y2_r = y2_r, y1_r
        draw.rectangle([x1_r, y1_r, x2_r, y2_r], outline='red', width=2)
        # draw.text((x1_resized, y2_resized), text, fill='black', font=font)

    # 保存整页bbox图像
    save_bbox_path = os.path.join(image_save_path, f"page_{page_num}_bbox.jpeg")
    image.save(save_bbox_path)
    print(f"第 {page_num} 页bbox结果保存至: {save_bbox_path}")


# 清洗html文本
def clean_and_format_html(full_predict):
    if "<body" not in full_predict.lower():
        full_predict = f"<html><body>{full_predict}</body></html>"
    soup = BeautifulSoup(full_predict, 'html.parser')

    # Regular expression pattern to match 'color' styles in style attributes
    color_pattern = re.compile(r'\bcolor:[^;]+;?')

    # Find all tags with style attributes and remove 'color' styles
    for tag in soup.find_all(style=True):
        original_style = tag.get('style', '')
        new_style = color_pattern.sub('', original_style)
        if not new_style.strip():
            del tag['style']
        else:
            new_style = new_style.rstrip(';')
            tag['style'] = new_style

    # Remove 'data-bbox' and 'data-polygon' attributes from all tags
    for attr in ["data-bbox", "data-polygon"]:
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]

    classes_to_update = ['formula.machine_printed', 'formula.handwritten']
    # Update specific class names in div tags
    for tag in soup.find_all(class_=True):
        if isinstance(tag, Tag) and 'class' in tag.attrs:
            new_classes = [cls if cls not in classes_to_update else 'formula' for cls in tag.get('class', [])]
            tag['class'] = list(dict.fromkeys(new_classes))  # Deduplicate and update class names

    # Clear contents of divs with specific class names and rename their classes
    for div in soup.find_all('div', class_='image caption'):
        div.clear()
        div['class'] = ['image']

    classes_to_clean = ['music sheet', 'chemical formula', 'chart']
    # Clear contents and remove 'format' attributes of tags with specific class names
    for class_name in classes_to_clean:
        for tag in soup.find_all(class_=class_name):
            if isinstance(tag, Tag):
                tag.clear()
                if 'format' in tag.attrs:
                    del tag['format']

    # Manually build the output string
    output = []
    for child in soup.body.children:
        if isinstance(child, Tag):
            output.append(str(child))
            output.append('\n')  # Add newline after each top-level element
        elif isinstance(child, str) and not child.strip():
            continue  # Ignore whitespace text nodes
    complete_html = f"""```html\n<html><body>\n{" ".join(output)}</body></html>\n```"""
    return complete_html


# html分块（仅用于原生html文件）
def split_html(html, max_chunk_chars=8000):
    if "<body" not in html.lower():
        html = f"<html><body>{html}</body></html>"
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.body

    chunks = []
    current_chunk = ""
    current_len = 0

    for child in body.children:
        if isinstance(child, Tag):
            html_str = str(child)
            html_len = len(html_str)
            if current_len + html_len > max_chunk_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_len = 0
            current_chunk += html_str + "\n"
            current_len += html_len
        elif isinstance(child, str) and child.strip():
            html_str = child.strip()
            html_len = len(html_str)
            if current_len + html_len > max_chunk_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_len = 0
            current_chunk += html_str + "\n"
            current_len += html_len

    if current_chunk:
        chunks.append(current_chunk)

    full_chunks = [f"<html><body>\n{chunk}\n</body></html>" for chunk in chunks]
    print(f"分为{len(full_chunks)}块")
    return full_chunks


# base64 编码图片
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ========== Pipeline A：布局解析（VLM） ==========
def inference_layout(image_path, model_id=VLM_MODELID,
                     min_pixels=512 * 28 * 28, max_pixels=2048 * 28 * 28):
    """
    只做页面布局解析：
    - 文本块、标题、列表、表格等
    - 图像只输出 <div class="image" data-bbox="x1 y1 x2 y2"></div>
    - caption 文本保留为普通段落，不放在 image div 中
    """
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
        "5. Make sure data-bbox covers the visual area of the image."
    )

    base64_image = encode_image(image_path)
    def _call():
        client = OpenAI(
            api_key=VLM_API_KEY,
            base_url=VLM_API_URL,
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return completion.choices[0].message.content if completion and completion.choices else None

    return retry_api_call(_call, name="VLM布局解析")


# ========== Pipeline B：图像 + 题注生成 ==========
def inference_figure_caption(fig_image_path, context_text,
                             model_id=VLM_MODELID,
                             min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28):
    """
    针对单张图 + 上下文，生成/抽取题注：
    1. 若上下文有明确 '图1 ...' / 'Figure 1 ...'，优先抽取；
    2. 否则根据图像 + 上下文生成简短题注；
    """
    sys_prompt = "You are an expert figure captioner for scientific and technical documents."
    user_prompt = (
        "You are given a figure image and some nearby text context from a technical document.\n"
        "Your task is to output ONE concise, high-quality figure caption in the ORIGINAL language "
        "(Chinese stays Chinese, English stays English).\n\n"
        "Priority:\n"
        "1. If the context already contains an explicit figure caption (for example, lines starting with "
        "   '图1', '图2', 'Figure 1', 'Fig. 1', etc.), EXTRACT that line (or the most complete one) "
        "   and use it as the caption.\n"
        "2. If there is no explicit caption in the context, then COMBINE the visual information in the image "
        "   and the context to GENERATE a short, informative caption that reasonably describes the figure.\n\n"
        "Requirements:\n"
        "- ALWAYS output a meaningful caption sentence.\n"
        "- ALWAYS output the caption in English.\n"
        "- DO NOT output 'null', 'N/A', 'none', 'unknown', or any placeholder-like text.\n"
        "- Do NOT add any explanation, justification, or quotes. Output ONLY the caption text itself.\n\n"
        f"Here is the nearby text context:\n{context_text}\n"
    )

    base64_image = encode_image(fig_image_path)

    def _call():
        client = OpenAI(
            api_key=VLM_API_KEY,
            base_url=VLM_API_URL,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return completion.choices[0].message.content.strip() if completion and completion.choices else None

    return retry_api_call(_call, name="图像题注生成")


# HTML → Markdown
def html_to_markdown(html_content, model_id=MARKDOWN_MODELID):
    system_prompt = "你是一个熟练的文档转换助手。"
    user_prompt = (
        "请你将解析得到的html格式文档解析为顺序输出的markdown文档，"
        "同时过滤掉和正文（正文包括标题、作者，摘要，主要文段、参考文献和附录信息）无关的页眉页脚等信息。"
        "如果存在图像, HTML 中只有此 `<div class='image'>` 标签代表图像，请为每个图像生成一条 `![FIG_PLACEHOLDER](#)` 形式的图像占位符，"
        "不需要根据图像内容生成名称；后续会用其他模块来填充真正的题注。"
        "直接输出解析后的markdown文档，不需要任何其他说明。注意，不需要翻译。"
    )

    def _call():
        client = OpenAI(
            api_key=MARKDOWN_API_KEY,
            base_url=MARKDOWN_API_URL,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{user_prompt}, 以下是html文档：{html_content}"},
                ],
            },
        ]

        completion = client.chat.completions.create(model=model_id, messages=messages)
        return completion.choices[0].message.content

    return retry_api_call(_call, name="Markdown 转换")


# 从 HTML 中为某个 image 找上下文
def get_context_for_image(soup: BeautifulSoup, image_tag: Tag, max_chars=400):
    """
    从 HTML DOM 中，取 image_tag 前后若干文本节点，拼成上下文。
    简单策略：向前找3个兄弟块 + 向后找3个兄弟块。
    """
    texts = []

    # 向前找
    prev = image_tag.previous_sibling
    count = 0
    while prev is not None and count < 3:
        if isinstance(prev, Tag):
            t = prev.get_text(strip=True)
            if t:
                texts.insert(0, t)  # 前面的插到前面
                count += 1
        prev = prev.previous_sibling

    # 向后找
    nxt = image_tag.next_sibling
    count = 0
    while nxt is not None and count < 3:
        if isinstance(nxt, Tag):
            t = nxt.get_text(strip=True)
            if t:
                texts.append(t)
                count += 1
        nxt = nxt.next_sibling

    context = "\n".join(texts)
    if len(context) > max_chars:
        context = context[:max_chars]
    return context

def is_pdf_converted(pdf_path, imgs_dir):
    doc = fitz.open(pdf_path)
    expected_pages = len(doc)
    existing = len([f for f in os.listdir(imgs_dir) if f.startswith("page_")])
    return existing == expected_pages

# ========== 各类文件处理函数 ==========

def process_pdf(file_path, output_dir):
    pdf_name = Path(file_path).stem
    md_path = os.path.join(output_dir, f"{pdf_name}.md")  # markdown 输出路径

    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        print(f"跳过已处理文件: {file_path}")
        return

    imgs_save_dir = os.path.join(output_dir, "pages")
    imgs_bbox_save_dir = os.path.join(output_dir, "pages_bbox")
    caption_dir = os.path.join(output_dir, "captions.json")
    fig_dir = os.path.join(output_dir, "figures")

    os.makedirs(imgs_save_dir, exist_ok=True)
    os.makedirs(imgs_bbox_save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    captions_dict = {}
    all_markdowns = []
    if not is_pdf_converted(file_path, imgs_save_dir):
        image_paths = pdf_to_image(file_path, imgs_save_dir)
    else:
        print("SKIP CONVERT")
        image_paths = sorted([os.path.join(imgs_save_dir, f) for f in os.listdir(imgs_save_dir)])

    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        try:
            print(f"提取第 {page_num} 页")
            image = Image.open(img_path)
            width, height = image.size

            min_pixels = 512 * 28 * 28
            max_pixels = 2048 * 28 * 28
            input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels, factor=32)

            # Pipeline A：页面布局 + 文本
            layout_html_raw = inference_layout(img_path,
                                               min_pixels=min_pixels,
                                               max_pixels=max_pixels)
            if not layout_html_raw:
                print(f"第 {page_num} 页布局解析失败，跳过...")
                continue

            # 画 bbox & 裁剪图像
            draw_bbox(img_path, imgs_bbox_save_dir, fig_dir, width, height, layout_html_raw, page_num)

            # 清洗HTML → Markdown
            cleaned_html = clean_and_format_html(layout_html_raw)
            markdown = html_to_markdown(cleaned_html)
            markdown = re.sub(r'^```markdown\s*', '', markdown)
            markdown = re.sub(r'\s*```$', '', markdown)
            all_markdowns.append(markdown)
            # print(layout_html_raw)

            # Pipeline B：图像 + 题注
            soup = BeautifulSoup(layout_html_raw, 'html.parser')
            image_tags = soup.find_all('div', class_='image')

            for img_idx, img_tag in enumerate(image_tags):
                fig_key = f"p{page_num}_{img_idx + 1}"
                fig_path = os.path.join(fig_dir, f"{fig_key}.jpg")

                if not os.path.exists(fig_path):
                    print(f"警告：找不到裁剪图像 {fig_path}，题注置为 示意图")
                    captions_dict[fig_key] = "示意图"
                    continue

                context_text = get_context_for_image(soup, img_tag, max_chars=400)
                caption = inference_figure_caption(fig_path, context_text) or "示意图"
                captions_dict[fig_key] = caption.strip()

        except Exception as e:
            print(f"第 {page_num} 页处理失败：{e}")
            continue

    # 写入 markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdowns))

    # 写入 captions.json
    with open(caption_dir, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)

    print(f"处理完成：{pdf_name}")


def process_html(file_path, output_dir):
    html_name = Path(file_path).stem
    md_path = os.path.join(output_dir, f"{html_name}.md")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        print(f"跳过已处理文件: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    cleaned_html = clean_and_format_html(raw_html)
    chunks = split_html(cleaned_html, max_chunk_chars=8000)

    all_markdowns = []
    for i, chunk in enumerate(chunks):
        print(f"处理 HTML 第 {i + 1} 块...")
        try:
            markdown = html_to_markdown(chunk)
            markdown = re.sub(r'^```markdown\s*', '', markdown)
            markdown = re.sub(r'\s*```$', '', markdown)
            all_markdowns.append(markdown)
        except Exception as e:
            print(f"第 {i + 1} 块处理失败：{e}")
            continue

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdowns))
    print(f"文件处理完成：{html_name}")


def process_excel(file_path, output_dir):
    file_name = Path(file_path).stem
    md_path = os.path.join(output_dir, f"{file_name}.md")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        print(f"跳过已处理文件: {file_path}")
        return

    try:
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path)
            markdown = df.to_markdown(index=False)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"### 表：{file_name}\n\n{markdown}")
            print(f"CSV 文件处理完成：{file_name}")

        elif str(file_path).endswith('.xlsx') or str(file_path).endswith('.xls'):
            xls = pd.ExcelFile(file_path)
            all_markdowns = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                markdown = df.to_markdown(index=False)
                all_markdowns.append(f"### 表：{sheet_name}\n\n{markdown}")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_markdowns))
            print(f"Excel 文件处理完成：{file_name}")

        else:
            print(f"不支持的文件格式: {file_path}")
            return

    except Exception as e:
        print(f"处理 {file_path} 失败：{e}")


def process_word(file_path, output_dir):
    word_name = Path(file_path).stem
    md_path = os.path.join(output_dir, f"{word_name}.md")

    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        print(f"跳过已处理文件: {file_path}")
        return

    imgs_save_dir = os.path.join(output_dir, "pages")
    imgs_bbox_save_dir = os.path.join(output_dir, "pages_bbox")
    caption_dir = os.path.join(output_dir, "captions.json")
    fig_dir = os.path.join(output_dir, "figures")

    os.makedirs(imgs_save_dir, exist_ok=True)
    os.makedirs(imgs_bbox_save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    captions_dict = {}
    all_markdowns = []

    image_paths = word_to_image(file_path, imgs_save_dir)

    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        try:
            print(f"提取第 {page_num} 页")
            image = Image.open(img_path)
            width, height = image.size

            min_pixels = 256 * 28 * 28
            max_pixels = 1024 * 28 * 28
            _ = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)

            # Pipeline A：布局 + 文本
            layout_html_raw = inference_layout(img_path,
                                               min_pixels=min_pixels,
                                               max_pixels=max_pixels)
            if not layout_html_raw:
                print(f"第 {page_num} 页布局解析失败，跳过...")
                continue

            draw_bbox(img_path, imgs_bbox_save_dir, fig_dir,
                      width, height, layout_html_raw, page_num)

            cleaned_html = clean_and_format_html(layout_html_raw)
            markdown = html_to_markdown(cleaned_html)
            markdown = re.sub(r'^```markdown\s*', '', markdown)
            markdown = re.sub(r'\s*```$', '', markdown)

            # 去掉 Aspose 评估版水印文本
            markdown = re.sub(
                r'```\nCreated with an evaluation copy of Aspose\.Words\. To remove all limitations,\n'
                r'you can use Free Temporary License\nhttps://products\.aspose\.com/words/temporary-license/\n',
                '',
                markdown,
                flags=re.DOTALL
            )
            markdown = re.sub(
                r'Evaluation Only\. Created with Aspose\.Words\. Copyright.*?Pty Ltd\.',
                '',
                markdown,
                flags=re.MULTILINE
            )

            all_markdowns.append(markdown)

            # Pipeline B：图像 + 题注
            soup = BeautifulSoup(layout_html_raw, 'html.parser')
            image_tags = soup.find_all('div', class_='image')

            for img_idx, img_tag in enumerate(image_tags):
                fig_key = f"p{page_num}_{img_idx + 1}"
                fig_path = os.path.join(fig_dir, f"{fig_key}.jpg")

                if not os.path.exists(fig_path):
                    print(f"警告：找不到裁剪图像 {fig_path}，题注置为 示意图")
                    captions_dict[fig_key] = "示意图"
                    continue

                context_text = get_context_for_image(soup, img_tag, max_chars=400)
                caption = inference_figure_caption(fig_path, context_text) or "示意图"
                captions_dict[fig_key] = caption.strip()

        except Exception as e:
            print(f"第 {page_num} 页处理失败：{e}")
            continue

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdowns))

    with open(caption_dir, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)

    print(f"处理完成：{word_name}")


def process_txt(file_path, output_dir):
    txt_name = Path(file_path).stem
    md_path = os.path.join(output_dir, f"{txt_name}.md")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        print(f"跳过已处理文件: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            txt = f.read()
        markdown = txt.strip()
        markdown = re.sub(r'^```markdown\s*', '', markdown)
        markdown = re.sub(r'\s*```$', '', markdown)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"TXT 文件处理完成：{txt_name}")
    except Exception as e:
        print(f"处理 {file_path} 失败：{e}")


def process_ppt(file_path, output_dir):
    ppt_name = Path(file_path).stem
    md_path = os.path.join(output_dir, f"{ppt_name}.md")

    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        print(f"跳过已处理文件: {file_path}")
        return

    imgs_save_dir = os.path.join(output_dir, "pages")
    imgs_bbox_save_dir = os.path.join(output_dir, "pages_bbox")
    caption_dir = os.path.join(output_dir, "captions.json")
    fig_dir = os.path.join(output_dir, "figures")

    os.makedirs(imgs_save_dir, exist_ok=True)
    os.makedirs(imgs_bbox_save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    captions_dict = {}
    all_markdowns = []

    image_paths = ppt_to_image(file_path, imgs_save_dir)

    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        try:
            print(f"提取第 {page_num} 页")
            image = Image.open(img_path)
            width, height = image.size

            min_pixels = 256 * 28 * 28
            max_pixels = 1024 * 28 * 28
            _ = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)

            # Pipeline A：布局 + 文本
            layout_html_raw = inference_layout(img_path,
                                               min_pixels=min_pixels,
                                               max_pixels=max_pixels)
            if not layout_html_raw:
                print(f"第 {page_num} 页布局解析失败，跳过...")
                continue

            draw_bbox(img_path, imgs_bbox_save_dir, fig_dir,
                      width, height, layout_html_raw, page_num)

            cleaned_html = clean_and_format_html(layout_html_raw)
            markdown = html_to_markdown(cleaned_html)
            markdown = re.sub(r'^```markdown\s*', '', markdown)
            markdown = re.sub(r'\s*```$', '', markdown)
            markdown = re.sub(
                r'Evaluation Only\. Created with Aspose\.Words\. Copyright.*?Pty Ltd\.',
                '',
                markdown,
                flags=re.MULTILINE
            )
            all_markdowns.append(markdown)

            # Pipeline B：图像 + 题注
            soup = BeautifulSoup(layout_html_raw, 'html.parser')
            image_tags = soup.find_all('div', class_='image')

            for img_idx, img_tag in enumerate(image_tags):
                fig_key = f"p{page_num}_{img_idx + 1}"
                fig_path = os.path.join(fig_dir, f"{fig_key}.jpg")

                if not os.path.exists(fig_path):
                    print(f"警告：找不到裁剪图像 {fig_path}，题注置为 示意图")
                    captions_dict[fig_key] = "示意图"
                    continue

                context_text = get_context_for_image(soup, img_tag, max_chars=400)
                caption = inference_figure_caption(fig_path, context_text) or "示意图"
                captions_dict[fig_key] = caption.strip()

        except Exception as e:
            print(f"第 {page_num} 页处理失败：{e}")
            continue

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdowns))

    with open(caption_dir, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)

    print(f"处理完成：{ppt_name}")


def main():
    input_dir = INPUT
    output_root = OUTPUT_ROOT

    for file_path in Path(input_dir).iterdir():
        if not file_path.is_file():
            continue
        file_type = get_file_type(file_path)
        file_name = file_path.stem
        output_dir = os.path.join(output_root, file_name)

        try:
            if file_type == "pdf":
                print(f"开始处理 PDF 文件: {file_path.name}")
                process_pdf(file_path, output_dir)
            elif file_type == "html":
                print(f"开始处理 HTML 文件: {file_path.name}")
                process_html(file_path, output_dir)
            elif file_type == "excel":
                print(f"开始处理 excel 文件: {file_path.name}")
                process_excel(file_path, output_dir)
            elif file_type == "word":
                print(f"开始处理 WORD 文件: {file_path.name}")
                process_word(file_path, output_dir)
            elif file_type == "txt":
                print(f"开始处理 TXT 文件: {file_path.name}")
                process_txt(file_path, output_dir)
            elif file_type == "ppt":
                print(f"开始处理 PPT 文件: {file_path.name}")
                process_ppt(file_path, output_dir)
            else:
                print(f"跳过不支持的文件类型: {file_path.name}")
        except Exception as e:
            print(f"处理 {file_path.name} 失败：{e}")


if __name__ == "__main__":
    main()
