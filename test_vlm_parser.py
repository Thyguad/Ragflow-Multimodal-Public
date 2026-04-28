"""
测试脚本：验证 VLMDocParser 能否正常调用 VLM API 并解析文档。

用法：
    python test_vlm_parser.py --pdf path/to/test.pdf
    python test_vlm_parser.py --pdf path/to/test.pdf --pages 0 3   # 只解析第1-3页

在本仓库根目录运行时请使用：
    python test_vlm_parser.py --pdf path/to/test.pdf
"""

import argparse
import json
import os
import sys

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────
# 配置区：填入你的 API 信息
# ──────────────────────────────────────────────
API_CONFIG = {
    # 用于 Pipeline A（页面布局解析）和 Pipeline B（图像题注）的 VLM
    "vlm_api_key": os.getenv("RAGFLOW_VLM_API_KEY", ""),
    "vlm_api_url":  "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "vlm_modelid":  "qwen2.5-vl-72b-instruct",

    # 用于 HTML → Markdown 转换的文字模型
    "markdown_api_key": os.getenv("RAGFLOW_MD_API_KEY", os.getenv("RAGFLOW_VLM_API_KEY", "")),   # 可与 vlm_api_key 相同
    "markdown_api_url":  "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "markdown_modelid":  "qwen-plus",
}
# ──────────────────────────────────────────────


def step_banner(n, title):
    print(f"\n{'='*60}")
    print(f"  步骤 {n}：{title}")
    print(f"{'='*60}")


def test_api_connectivity():
    """步骤1：测试 VLM API 连通性（发送一条纯文字消息）"""
    step_banner(1, "API 连通性测试")
    from openai import OpenAI

    client = OpenAI(
        api_key=API_CONFIG["vlm_api_key"],
        base_url=API_CONFIG["vlm_api_url"],
    )
    try:
        resp = client.chat.completions.create(
            model=API_CONFIG["vlm_modelid"],
            messages=[{"role": "user", "content": "请回答：1+1=?，只输出数字"}],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"  VLM API 连通 ✓   返回：{answer!r}")
        return True
    except Exception as e:
        print(f"  VLM API 连接失败 ✗   错误：{e}")
        return False


def test_markdown_api_connectivity():
    """步骤2：测试 Markdown 转换模型连通性"""
    step_banner(2, "Markdown 模型 API 连通性测试")
    from openai import OpenAI

    client = OpenAI(
        api_key=API_CONFIG["markdown_api_key"],
        base_url=API_CONFIG["markdown_api_url"],
    )
    try:
        resp = client.chat.completions.create(
            model=API_CONFIG["markdown_modelid"],
            messages=[{"role": "user", "content": "你好，请回复 OK"}],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"  Markdown 模型 API 连通 ✓   返回：{answer!r}")
        return True
    except Exception as e:
        print(f"  Markdown 模型 API 连接失败 ✗   错误：{e}")
        return False


def test_vlm_with_image(pdf_path, from_page=0, to_page=1):
    """步骤3：截取一页 PDF 渲染为图片，发给 VLM 验证图像输入是否正常"""
    step_banner(3, "VLM 图像输入测试（取第1页）")
    import base64
    import tempfile

    import fitz
    from openai import OpenAI
    from PIL import Image

    # 渲染第一页
    doc = fitz.open(pdf_path)
    page = doc[from_page]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)

    with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f:
        tmp_path = f.name
    pix.save(tmp_path)

    # 压缩到合理大小
    img = Image.open(tmp_path)
    print(f"  页面图片尺寸：{img.width} × {img.height}")
    img.close()

    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)

    client = OpenAI(
        api_key=API_CONFIG["vlm_api_key"],
        base_url=API_CONFIG["vlm_api_url"],
    )
    try:
        resp = client.chat.completions.create(
            model=API_CONFIG["vlm_modelid"],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": 512 * 28 * 28,
                        "max_pixels": 2048 * 28 * 28,
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": "这张图片里有什么内容？请用一句话描述。"},
                ],
            }],
        )
        answer = resp.choices[0].message.content.strip()
        print(f"  VLM 图像理解 ✓")
        print(f"  返回内容：{answer[:200]}")
        return True
    except Exception as e:
        print(f"  VLM 图像输入失败 ✗   错误：{e}")
        print("  提示：如果错误含 'min_pixels'，说明该模型不支持此参数，")
        print("       可在 vlm_doc_parser.py 的 _inference_layout 中去掉该参数")
        return False


def test_full_pipeline(pdf_path, from_page=0, to_page=2):
    """步骤4：完整运行 VLMDocParser，解析指定页范围"""
    step_banner(4, f"完整解析测试（第 {from_page+1}~{to_page} 页）")
    from deepdoc.parser.vlm_doc_parser import VLMDocParser

    with open(pdf_path, "rb") as f:
        binary = f.read()

    parser = VLMDocParser(api_config=API_CONFIG)

    print("  正在解析，请稍候（每页需调用 VLM 2-3 次）…")
    try:
        markdown_str, figures_dict, captions_dict = parser(
            binary, file_type="pdf", from_page=from_page, to_page=to_page
        )
    except Exception as e:
        print(f"  解析失败 ✗   错误：{e}")
        import traceback
        traceback.print_exc()
        return False

    # ── 输出结果 ──────────────────────────────
    print(f"\n  Markdown 总长度：{len(markdown_str)} 字符")
    print(f"  提取图片数量：{len(figures_dict)}")
    print(f"  生成题注数量：{len(captions_dict)}")

    print("\n  ── Markdown 前 800 字 ──")
    print(markdown_str[:800])

    if figures_dict:
        print(f"\n  ── 图片列表 ──")
        for key, img_bytes in figures_dict.items():
            caption = captions_dict.get(key, "(无题注)")
            print(f"    {key:10s}  {len(img_bytes)//1024:4d} KB  题注: {caption}")

        # 把第一张图片保存到本地方便查看
        first_key = list(figures_dict.keys())[0]
        out_path = f"test_figure_{first_key}.jpg"
        with open(out_path, "wb") as f:
            f.write(figures_dict[first_key])
        print(f"\n  第一张图片已保存到：{out_path}")

    # 把完整 Markdown 保存
    md_out = "test_output.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(markdown_str)
    print(f"  完整 Markdown 已保存到：{md_out}")

    # 检查占位符是否有遗漏
    remaining = markdown_str.count("FIG_PLACEHOLDER")
    if remaining > 0:
        print(f"\n  ⚠ 警告：Markdown 中仍有 {remaining} 个 FIG_PLACEHOLDER 未被替换")
    else:
        print("\n  所有图片占位符已正确替换 ✓")

    return True


def main():
    arg_parser = argparse.ArgumentParser(description="测试 VLMDocParser 解析流程")
    arg_parser.add_argument("--pdf", required=True, help="测试用 PDF 文件路径")
    arg_parser.add_argument("--from-page", type=int, default=0, help="起始页（0-based，默认0）")
    arg_parser.add_argument("--to-page", type=int, default=2, help="结束页（不含，默认2，即解析前2页）")
    arg_parser.add_argument("--skip-connectivity", action="store_true", help="跳过 API 连通性检查")
    args = arg_parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"错误：找不到文件 {args.pdf}")
        sys.exit(1)

    print(f"\n测试文件：{args.pdf}")
    print(f"解析页范围：第 {args.from_page+1} ~ {args.to_page} 页")
    print(f"VLM 模型：{API_CONFIG['vlm_modelid']}")
    print(f"Markdown 模型：{API_CONFIG['markdown_modelid']}")

    if not args.skip_connectivity:
        ok1 = test_api_connectivity()
        ok2 = test_markdown_api_connectivity()
        if not (ok1 and ok2):
            print("\n⚠ API 连通性测试未全部通过，是否继续？(y/n) ", end="")
            if input().strip().lower() != "y":
                sys.exit(1)

        test_vlm_with_image(args.pdf, from_page=args.from_page)

    ok = test_full_pipeline(args.pdf, from_page=args.from_page, to_page=args.to_page)

    print(f"\n{'='*60}")
    print(f"  测试结果：{'全部通过 ✓' if ok else '存在失败项 ✗'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
