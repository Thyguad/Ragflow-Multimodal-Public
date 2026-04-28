# RAG 评估脚本（检索 + 生成）

本目录提供 RAG 检索与生成质量评估脚本。公开仓库只保留评估代码和数据格式说明，不包含私有 ground truth、预测输出或实验报告。

可参考示例：

- 文本 Ground Truth 示例：[`examples/text_groundtruth_example.json`](./examples/text_groundtruth_example.json)
- 图片 Ground Truth 示例：[`examples/image_groundtruth_example.json`](./examples/image_groundtruth_example.json)

## 输入数据

### Ground truth

`--ground-truth` 指向一个 JSON 数组，每个元素包含：

- `qid`：问题唯一 ID
- `question`：问题
- `question_type`：例如 `single_hop` 或 `multi_hop`
- `ground_truth_evidences`：证据数组
- `reference_answer`：参考答案

每条 evidence 建议包含：

- `evidence_id`
- `doc_uid`：与检索结果中的 `document_name` 对齐
- `evidence_text`：短文本证据
- `relevance`：相关度等级，例如 0/1/2

### Predictions

`--predictions` 指向一个 JSON 数组，每个元素包含：

- `qid`
- `retrieved_chunks`：检索结果数组，脚本也兼容 `retrieval_chunks`
- `generated_answer`：生成答案，启用 ragas 生成评估时使用

每条 chunk 建议包含：

- `chunk_id`
- `document_name`
- `chunk_content`
- `rank`：排序名次，可选
- `score`：检索分数，可选

## 评估逻辑

检索评估会先按 `qid` 对齐 ground truth 与 predictions，再计算：

- `recall@k`
- `hitrate@k`
- `mrr`
- `ndcg@k`

匹配逻辑分两层：

- 文档级：`doc_uid` 与 `document_name` 归一化后相等。
- 文本级：优先做包含匹配；未命中时使用文本相似度和字符覆盖率做宽松匹配。

生成评估可选启用 ragas，包含：

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

## 运行方式

只评估检索：

```bash
python evaluate_rag/rag_eval_full.py \
  --ground-truth /path/to/ground_truth.json \
  --predictions /path/to/predictions.json \
  --ks 1,3,5,10 \
  --match-threshold 0.9 \
  --output /path/to/report.json
```

同时评估检索和生成：

```bash
export OPENAI_API_KEY="YOUR_LLM_API_KEY"

python evaluate_rag/rag_eval_full.py \
  --ground-truth /path/to/ground_truth.json \
  --predictions /path/to/predictions.json \
  --ks 1,3,5,10 \
  --match-threshold 0.9 \
  --enable-generation \
  --llm-base-url https://api.example.com/v1 \
  --llm-api-key "$OPENAI_API_KEY" \
  --generation-llm-model your-chat-model \
  --embedding-base-url https://embedding.example.com/v1 \
  --embedding-api-key "$OPENAI_API_KEY" \
  --generation-embedding-model your-embedding-model \
  --output /path/to/report.json
```

输出报告包含 `config`、`retrieval`，启用生成评估时还会包含 `generation`。
