# RAGFlow-Multimodal

Multimodal Document QA for RAGFlow

This is a public release of a multimodal document question-answering system built on top of RAGFlow.

It focuses on complex document understanding, image asset extraction, optional text-image retrieval, image display planning, and offline evaluation scripts.

For the full Chinese documentation, see [README_zh.md](./README_zh.md).

## Project Name

The project is called **RAGFlow-Multimodal**.

It is not the upstream RAGFlow distribution. It is a public research and engineering fork that keeps the RAGFlow application foundation while adding a multimodal document QA workflow.

## What Changed Compared With Upstream RAGFlow

This release adds the following multimodal components on top of the original RAGFlow codebase:

- **VLM document parsing path**: `deepdoc/parser/vlm_doc_parser.py` and `rag/app/naive.py` add a VLM-based parser path for complex PDF and DOCX documents.
- **Image asset export**: parsed figures are written to `image_assets/<doc_uid>/` with `figures/*.jpg`, `captions.json`, and `manifest.json`.
- **Text-image ingestion bridge**: document ingestion keeps text chunks and image sidecar metadata aligned for later retrieval.
- **Optional image sidecar index**: `rag/image_index.py` builds and loads image/caption vectors for image candidate retrieval when image retrieval enhancement is enabled.
- **Image retrieval client**: `rag/multimodal_embedding_client.py` provides the image embedding interface used by the sidecar index.
- **Image display decision layer**: `rag/chat_image_planner.py` adds image gate and render-plan logic, deciding `none`, `single`, `compare`, or `gallery`.
- **Dialog integration**: `api/db/services/dialog_service.py` writes `reference.image_candidates`, `reference.image_gate`, and `reference.image_render_plan` into chat references.
- **Local image service**: `image_server.py` serves generated image assets through `/image_assets/<path>`.
- **Evaluation utilities**: `evaluate_rag/` contains text retrieval, image retrieval, image display decision, and example ground-truth files for public reproduction.

## What Is Included

- Multimodal document parsing with VLM-assisted page understanding
- Image asset export to `image_assets/<doc_uid>/`
- Caption and manifest generation for extracted figures
- Optional image sidecar indexing and late-fusion retrieval
- Image gate and render-plan logic for chat answers
- Frontend integration for multimodal retrieval settings and image display metadata
- General evaluation scripts under `evaluate_rag/`

## What Is Not Included

This public repository does not include private documents, real evaluation datasets, parsed artifacts, historical experiment results, API keys, private keys, local paths, or personal progress notes.

After cloning, you need to install dependencies, start the required services, configure model credentials, and upload your own PDF or DOCX files.

## Quick Start

Requirements:

- Python 3.10
- Node.js >= 18.20.4
- Docker or Docker Desktop
- uv
- npm
- At least 16 GB memory is recommended

Install dependencies:

```bash
git clone https://github.com/<your-user>/RAGFlow-Multimodal.git
cd RAGFlow-Multimodal

uv sync --python 3.10 --all-extras
cd web && npm install && cd ..
```

Configure the model credentials used by the multimodal pipeline:

```bash
export RAGFLOW_VLM_API_KEY=<your_vlm_key>
export RAGFLOW_MD_API_KEY=<your_markdown_model_key>
export RAGFLOW_IMAGE_EMBEDDING_API_KEY=<your_image_embedding_key>
export SERVER_IP=http://127.0.0.1:8000
```

Start services:

```bash
bash ./start_docker_services.sh
bash ./start_backend.sh
bash ./start_frontend.sh
```

Then open:

- Frontend: `http://localhost:9222`
- Backend: `http://127.0.0.1:9380`
- Image service: `http://127.0.0.1:8000/`

## Multimodal Usage

Image retrieval is not enabled by default. The code defaults `parser_config.enable_multimodal_image_retrieval` to `false`, and both image sidecar construction and chat-time image candidate recall check this switch before running.

In practice:

- `layout_recognize = "VLM"` enables VLM-assisted parsing and image asset export.
- `enable_multimodal_image_retrieval = true` enables the extra image retrieval enhancement path that builds/uses the image sidecar index and returns `reference.image_candidates`, `reference.image_gate`, and `reference.image_render_plan`.
- If this switch stays disabled, the system can still answer with the normal text RAG path, and VLM parsing can still export image assets, but online image recall/display planning will not run.

To test the full multimodal workflow with online image recall:

1. Create a knowledge base.
2. Use the `naive` parser.
3. Set `parser_config.layout_recognize = "VLM"`.
4. Enable image retrieval enhancement: `parser_config.enable_multimodal_image_retrieval = true`.
5. Upload your own PDF or DOCX document.
6. Wait until parsing finishes and check `image_assets/<doc_uid>/`.
7. Ask image-related questions in chat and inspect `reference.image_candidates`, `reference.image_gate`, and `reference.image_render_plan`.

## Evaluation

The public release keeps the evaluation code, but does not publish private datasets or experiment outputs.

Use your own samples under `data/eval_samples/` and run the scripts in `evaluate_rag/`.

## License

This project follows the license in [LICENSE](./LICENSE).
