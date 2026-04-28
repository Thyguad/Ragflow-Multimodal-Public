# 实验设置与评测策略说明

## 1. 实验目标

本实验面向电磁领域文档问答任务，评估系统在 `RAG` 场景下的两类核心能力：

- 检索能力：系统能否从知识库中召回并排序出支撑答案的关键证据。
- 生成能力：系统能否基于检索到的上下文生成与参考答案一致、且忠实于证据的回答。

本文重点说明 `baseline` 与 `VLM` 两种文档解析方案的评测方法。公开分支不包含私有 Ground Truth、预测结果或实验报告；请使用自己的文档与标注数据复现实验。

## 2. 数据与实验输入

### 2.1 Ground Truth 数据集

- Ground Truth 文件：由用户自行准备，例如 `data/eval_samples/text_eval/your_groundtruth.json`
- 文本样例文件：`evaluate_rag/examples/text_groundtruth_example.json`
- 图片样例文件：`evaluate_rag/examples/image_groundtruth_example.json`
- 数据规模：由用户数据集决定
- 问题类型：
  - `single_hop`
  - `multi_hop`

每条样本包含以下字段：

- `qid`：问题编号
- `question`：问题文本
- `question_type`：问题类型
- `ground_truth_evidences`：人工标注的支撑证据列表
- `reference_answer`：参考答案

其中，`ground_truth_evidences` 中的每条证据包含：

- `evidence_id`
- `doc_uid`
- `evidence_text`
- `relevance`

## 3. 评测流程

整体评测流程如下：

1. 将 Ground Truth 问题批量输入 RAGFlow 系统。
2. 保存系统返回的检索结果与生成结果。
3. 使用统一评测脚本对检索结果和生成结果进行离线评分。

相关脚本与文件如下：

- 评测脚本：`evaluate_rag/rag_eval_full.py`
- 检索与生成总报告：建议输出到本地私有目录，例如 `data/eval_samples/`

## 4. 检索评测设计

### 4.1 基本思想

检索评测采用 evidence-level 评测思路。即：

- 不再只判断某个 chunk 是否“看起来相关”
- 而是判断它是否真正命中了人工标注的证据

这样做的原因是，本文关注的不是粗粒度召回，而是系统是否能在前排结果中给出足以支撑答案的证据。

### 4.2 证据匹配规则

在 `evaluate_rag/rag_eval_full.py` 中，每条 evidence 允许通过以下三种方式与检索结果建立对应关系：

1. `strict`
   - evidence 文本与当前 chunk 文本直接匹配
   - 这是最强、最可信的命中方式

2. `same_doc_context`
   - 当前 chunk 本身未直接命中，但在同一文档的累计上下文中可以命中
   - 该信号适合辅助分析文档级覆盖情况

3. `answer_support`
   - 若生成答案与 evidence 或参考答案具有足够高的文本支持度，则认为该 evidence 获得弱支持
   - 该规则用于补偿表达改写、摘要压缩等情况

### 4.3 文本匹配方法

`strict` 与 `same_doc_context` 均采用统一的文本匹配策略，主要包含以下步骤：

- 文本归一化
- 句窗切分
- 序列相似度比较
- 字符覆盖率比较

当前固定参数为：

- `match_threshold = 0.82`
- `overlap_threshold = 0.75`

### 4.4 支持度阈值

`answer_support` 使用生成答案对证据的支持度作为补充判断依据，当前固定阈值为：

- `support_threshold = 0.70`

### 4.5 置信度校准（按 profile）

评测脚本支持 `retrieval_scoring_profile`，不同 profile 的置信度规则不同。

#### `primary_v2`（当前主口径）

- `strict = 1.0`
- `answer_support = 0.935`
- `same_doc_context = 0.0`

含义如下：

- `strict` 命中获得完整检索 credit
- `answer_support` 命中获得部分检索 credit
- `same_doc_context` 仅作为分析信号保留，不直接计入主检索得分

这样设计的原因是：

- `strict` 最能代表真实证据被直接检索到
- `answer_support` 有助于覆盖改写、压缩和生成表达变化带来的匹配误差
- `same_doc_context` 虽然说明文档级覆盖存在，但不能等价于“前排检索结果直接命中证据”

#### `legacy_strict_v1`（历史复现实验口径）

- `strict = 1.0`
- `answer_support = 1.0`
- `same_doc_context = 1.0`

## 5. 检索指标

本文保留常见的 evidence-level 检索指标：

- `recall@k`
- `hitrate@k`
- `ndcg@k`
- `mrr`

其中：

- `recall@k` 表示前 `k` 个结果中累计覆盖到的证据比例
- `hitrate@k` 表示前 `k` 个结果中是否出现有效证据命中
- `ndcg@k` 表示前 `k` 个结果中的排序质量
- `mrr` 表示最早有效证据出现的位置质量

需要说明的是，在当前实现中，上述指标并非简单二值计分，而是使用了前述置信度校准后的 credit。

## 6. 汇总策略

### 6.1 Micro Overall

`micro_overall`（兼容字段 `raw_overall`）表示对全部问题直接求平均，反映数据集自然分布下的整体表现。

### 6.2 Macro Overall

`macro_overall`（兼容字段 `overall`）表示先按 `question_type` 分组，再对各组结果做平衡平均。

这样做的原因是：

- 数据集中 `single_hop` 与 `multi_hop` 的题量并不完全相同
- 若直接整体平均，题量较多的问题类型会主导最终结论
- 宏平均更适合作为本文主口径，以平衡比较两类问题上的能力

### 6.3 Primary Summary

`primary_summary` 为 headline score，按 profile 定义：

- `primary_v2`：`mean(recall@3, hitrate@3, recall@5, hitrate@5)`
- `legacy_strict_v1`：`mean(recall@1, recall@3, hitrate@1, hitrate@3, ndcg@3, mrr)`

`primary_v2` 强调：

- 前 3 到 5 条结果是否已经覆盖足够答案支撑证据
- 多条上下文联合使用时的早期可用性
- 对单一 top1 偶然波动的鲁棒性

因此，论文主结果建议使用 `primary_v2`；`legacy_strict_v1` 仅用于历史结果复现和口径对照。

此外，报告中可并列给出 `structural_robustness` 作为辅助评分，用于衡量结构保持与多跳准备度。其定义为：

- `mean(gt_doc_coverage@3, gt_doc_coverage@5, multi_hop_multi_doc_rate@5, type_balance.score)`

其中：

- `gt_doc_coverage@k`：前 `k` 条结果覆盖标准证据所属文档的比例
- `multi_hop_multi_doc_rate@5`：多跳题前 5 条结果中出现至少 2 个文档来源的比例
- `type_balance.score`：`single_hop` 与 `multi_hop` 两类题上的主指标表现差距越小，得分越高

## 7. 生成评测设计

在开启 `--enable-generation` 时，脚本基于 `ragas` 对生成结果进行评估，主要指标包括：

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

生成评测的作用是判断：

- 回答是否忠实于检索上下文
- 回答是否与问题相关
- 检索上下文是否精确
- 检索上下文是否覆盖足够答案支撑信息

生成评测可按数据集规模调整上下文数量，例如：

- `--generation-max-contexts 8`

公开仓库不包含上下文 sweep 结果。建议在自己的数据集上比较不同 `--generation-max-contexts` 设置，再选择最终报告口径。

## 8. 当前采用该策略的原因

本文最终采用上述评测策略，主要基于以下考虑：

1. 仅使用严格的字符串级 evidence 命中会低估系统在改写、摘要和复杂版面解析场景下的真实能力。
2. 仅使用整体平均会掩盖 `single_hop` 与 `multi_hop` 的能力差异。
3. 仅关注 `recall@8` 等较大 `k` 值，会弱化“前排结果是否真正可用”这一更符合实际问答体验的因素。

因此，本文采用“evidence-level 匹配 + 置信度校准 + 题型平衡宏平均 + 前排质量主指标”的组合评测方案。

## 9. 使用建议

在论文正文中，建议按以下方式报告实验结果：

- 主结果：报告 `primary_summary` 与 `macro_overall`
- 辅助结果：报告 `micro_overall`
- 细粒度分析：报告 `single_hop` 与 `multi_hop` 分组结果
- 生成结果：报告 `faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`

若需强调评测公平性，可补充说明：

- `same_doc_context` 仅保留为分析信号，不直接获得检索得分
- `answer_support` 仅获得部分 credit，用于缓解表达改写带来的误判

## 10. 方法边界

需要指出的是，当前策略更适合作为本文场景下的实验主口径，而非通用信息检索 benchmark。其原因在于：

- 评测中引入了生成答案支持度
- 主结果采用了题型宏平均
- `primary_summary` 更强调前排结果质量而非纯粹 Top-k 覆盖

因此，本文在解释结果时，将其定位为：

- 面向复杂文档问答场景的综合评测方案
- 而非完全脱离生成链路的纯检索标准
