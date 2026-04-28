#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
import datetime
import json
import logging
import re
from collections import defaultdict

import json_repair

from api import settings
from api.db import LLMType
from rag.settings import TAG_FLD
from rag.utils import encoder, num_tokens_from_string



def chunks_format(reference):
    def get_value(d, k1, k2):
        return d.get(k1, d.get(k2))

    return [
        {
            "id": get_value(chunk, "chunk_id", "id"),
            "content": get_value(chunk, "content", "content_with_weight"),
            "document_id": get_value(chunk, "doc_id", "document_id"),
            "document_name": get_value(chunk, "docnm_kwd", "document_name"),
            "dataset_id": get_value(chunk, "kb_id", "dataset_id"),
            "image_id": get_value(chunk, "image_id", "img_id"),
            "positions": get_value(chunk, "positions", "position_int"),
            "url": chunk.get("url"),
            "similarity": chunk.get("similarity"),
            "vector_similarity": chunk.get("vector_similarity"),
            "term_similarity": chunk.get("term_similarity"),
            "doc_type": chunk.get("doc_type_kwd"),
        }
        for chunk in reference.get("chunks", [])
    ]


def llm_id2llm_type(llm_id):
    from api.db.services.llm_service import TenantLLMService

    llm_id, *_ = TenantLLMService.split_model_name_and_factory(llm_id)  # 提取模型ID和工厂名称,比如原始的llm_id为deepseek-chat@DeepSeek经处理后返回llm_id = deepseek-chat

    llm_factories = settings.FACTORY_LLM_INFOS
    for llm_factory in llm_factories:
        for llm in llm_factory["llm"]:
            if llm_id == llm["llm_name"]:
                # return llm["model_type"].strip(",")[-1]   # 这里写的有一些问题，不管是chat/image2txt返回的都是t
                return llm["model_type"].strip(",")


def message_fit_in(msg, max_length=4000):
    def count():
        nonlocal msg
        tks_cnts = []
        for m in msg:
            tks_cnts.append({"role": m["role"], "count": num_tokens_from_string(m["content"])})
        total = 0
        for m in tks_cnts:
            total += m["count"]
        return total

    c = count()
    if c < max_length:
        return c, msg

    msg_ = [m for m in msg if m["role"] == "system"]
    if len(msg) > 1:
        msg_.append(msg[-1])
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg

    ll = num_tokens_from_string(msg_[0]["content"])
    ll2 = num_tokens_from_string(msg_[-1]["content"])
    if ll / (ll + ll2) > 0.8:
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[: max_length - ll2])
        msg[0]["content"] = m
        return max_length, msg

    m = msg_[-1]["content"]
    m = encoder.decode(encoder.encode(m)[: max_length - ll2])
    msg[-1]["content"] = m
    return max_length, msg


def kb_prompt(kbinfos, max_tokens):
    from api.db.services.document_service import DocumentService

    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            logging.warning(f"Not all the retrieval into prompt: {i + 1}/{len(knowledges)}")
            break

    docs = DocumentService.get_by_ids([ck["doc_id"] for ck in kbinfos["chunks"][:chunks_num]])
    docs = {d.id: d.meta_fields for d in docs}

    doc2chunks = defaultdict(lambda: {"chunks": [], "meta": []})
    for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
        cnt = f"---\nID: {i}\n" + (f"URL: {ck['url']}\n" if "url" in ck else "")
        cnt += ck["content_with_weight"]
        doc2chunks[ck["docnm_kwd"]]["chunks"].append(cnt)
        doc2chunks[ck["docnm_kwd"]]["meta"] = docs.get(ck["doc_id"], {})

    knowledges = []
    for nm, cks_meta in doc2chunks.items():
        txt = f"\nDocument: {nm} \n"
        for k, v in cks_meta["meta"].items():
            txt += f"{k}: {v}\n"
        txt += "Relevant fragments as following:\n"
        for i, chunk in enumerate(cks_meta["chunks"], 1):
            txt += f"{chunk}\n"
        knowledges.append(txt)
    return knowledges


def citation_prompt():
    return """

# Citation requirements:
- Inserts CITATIONS in format '##i$$ ##j$$' where i,j are the ID of the content you are citing and encapsulated with '##' and '$$'.
- Inserts the CITATION symbols at the end of a sentence, AND NO MORE than 4 citations.
- DO NOT insert CITATION in the answer if the content is not from retrieved chunks.
- DO NOT use standalone Document IDs (e.g., '#ID#').
- Under NO circumstances any other citation styles or formats (e.g., '~~i==', '[i]', '(i)', etc.) be used.
- Citations ALWAYS the '##i$$' format.
- Any failure to adhere to the above rules, including but not limited to incorrect formatting, use of prohibited styles, or unsupported citations, will be considered a error, should skip adding Citation for this sentence.

--- Example START ---
<SYSTEM>: Here is the knowledge base:

Document: Elon Musk Breaks Silence on Crypto, Warns Against Dogecoin ...
URL: https://blockworks.co/news/elon-musk-crypto-dogecoin
ID: 0
The Tesla co-founder advised against going all-in on dogecoin, but Elon Musk said it’s still his favorite crypto...

Document: Elon Musk's Dogecoin tweet sparks social media frenzy
ID: 1
Musk said he is 'willing to serve' D.O.G.E. – shorthand for Dogecoin.

Document: Causal effect of Elon Musk tweets on Dogecoin price
ID: 2
If you think of Dogecoin — the cryptocurrency based on a meme — you can’t help but also think of Elon Musk...

Document: Elon Musk's Tweet Ignites Dogecoin's Future In Public Services
ID: 3
The market is heating up after Elon Musk's announcement about Dogecoin. Is this a new era for crypto?...

      The above is the knowledge base.

<USER>: What's the Elon's view on dogecoin?

<ASSISTANT>: Musk has consistently expressed his fondness for Dogecoin, often citing its humor and the inclusion of dogs in its branding. He has referred to it as his favorite cryptocurrency ##0$$ ##1$$.
Recently, Musk has hinted at potential future roles for Dogecoin. His tweets have sparked speculation about Dogecoin's potential integration into public services ##3$$.
Overall, while Musk enjoys Dogecoin and often promotes it, he also warns against over-investing in it, reflecting both his personal amusement and caution regarding its speculative nature.

--- Example END ---

"""


def keyword_extraction(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer.
Task: extract the most important keywords/phrases of a given piece of text content.
Requirements:
  - Summarize the text content, and give top {topn} important keywords/phrases.
  - The keywords MUST be in language of the given piece of text content.
  - The keywords are delimited by ENGLISH COMMA.
  - Keywords ONLY in output.

### Text Content
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


def question_proposal(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer.
Task:  propose {topn} questions about a given piece of text content.
Requirements:
  - Understand and summarize the text content, and propose top {topn} important questions.
  - The questions SHOULD NOT have overlapping meanings.
  - The questions SHOULD cover the main content of the text as much as possible.
  - The questions MUST be in language of the given piece of text content.
  - One question per line.
  - Question ONLY in output.

### Text Content
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


def full_question(tenant_id, llm_id, messages, language=None):
    from api.db.services.llm_service import LLMBundle

    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    conv = []
    for m in messages:
        if m["role"] not in ["user", "assistant"]:
            continue
        conv.append("{}: {}".format(m["role"].upper(), m["content"])) # 对话解析成列表
    conv = "\n".join(conv) # 用换行符拼接成字符串
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    prompt = f"""
Role: A helpful assistant

Task and steps:
    1. Generate a full user question that would follow the conversation.
    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is {today}. For example: 'yesterday' would be converted to {yesterday}.

Requirements & Restrictions:
  - If the user's latest question is completely, don't do anything, just return the original question.
  - DON'T generate anything except a refined question."""
    if language:
        prompt += f"""
  - Text generated MUST be in {language}."""
    else:
        prompt += """
  - Text generated MUST be in the same language of the original user's question.
"""
    prompt += f"""

######################
-Examples-
######################

# Example 1
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
###############
Output: What's the name of Donald Trump's mother?

------------
# Example 2
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
ASSISTANT:  Mary Trump.
User: What's her full name?
###############
Output: What's the full name of Donald Trump's mother Mary Trump?

------------
# Example 3
## Conversation
USER: What's the weather today in London?
ASSISTANT:  Cloudy.
USER: What's about tomorrow in Rochester?
###############
Output: What's the weather in Rochester on {tomorrow}?

######################
# Real Data
## Conversation
{conv}
###############
    """
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": "Output: "}], {"temperature": 0.2})
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    return ans if ans.find("**ERROR**") < 0 else messages[-1]["content"]

def cross_languages(tenant_id, llm_id, query, languages=[]):
    from api.db.services.llm_service import LLMBundle

    if llm_id and llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)

    sys_prompt = """
Act as a streamlined multilingual translator. Strictly output translations separated by ### without any explanations or formatting. Follow these rules:

1. Accept batch translation requests in format:
[source text]
=== 
[target languages separated by commas]

2. Always maintain:
- Original formatting (tables/lists/spacing)
- Technical terminology accuracy
- Cultural context appropriateness

3. Output format:
[original language text]
###
[language1 translation] 
### 
[language2 translation]

**Examples:**
Input:
Hello World! Let's discuss AI safety.
===
Chinese, French, Jappanese

Output:
Hello World! Let's discuss AI safety.
###
你好世界！让我们讨论人工智能安全问题。
###
Bonjour le monde ! Parlons de la sécurité de l'IA.
###
こんにちは世界！AIの安全性について話し合いましょう。
"""
    user_prompt=f"""
Input:
{query}
===
{', '.join(languages)}

Output:
"""

    ans = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {"temperature": 0.2})
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    if ans.find("**ERROR**") >= 0:
        return query
    questions="\n".join([a for a in re.sub(r"(^Output:|\n+)", "", ans, flags=re.DOTALL).split("===") if a.strip()])
    print("翻译后的问题：", questions)
    return questions


def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    prompt = f"""
Role: You're a text analyzer.

Task: Tag (put on some labels) to a given piece of text content based on the examples and the entire tag set.

Steps::
  - Comprehend the tag/label set.
  - Comprehend examples which all consist of both text content and assigned tags with relevance score in format of JSON.
  - Summarize the text content, and tag it with top {topn} most relevant tags from the set of tag/label and the corresponding relevance score.

Requirements
  - The tags MUST be from the tag set.
  - The output MUST be in JSON format only, the key is tag and the value is its relevance score.
  - The relevance score must be range from 1 to 10.
  - Keywords ONLY in output.

# TAG SET
{", ".join(all_tags)}

"""
    for i, ex in enumerate(examples):
        prompt += """
# Examples {}
### Text Content
{}

Output:
{}

        """.format(i, ex["content"], json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False))

    prompt += f"""
# Real Data
### Text Content
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.5})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    try:
        obj = json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        try:
            result = kwd.replace(prompt[:-1], "").replace("user", "").replace("model", "").strip()
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            obj = json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e
    res = {}
    for k, v in obj.items():
        try:
            res[str(k)] = int(v)
        except Exception:
            pass
    return res


def vision_llm_describe_prompt(page=None) -> str:
    prompt_en = """
INSTRUCTION:
Transcribe the content from the provided PDF page image into clean Markdown format.
- Only output the content transcribed from the image.
- Do NOT output this instruction or any other explanation.
- If the content is missing or you do not understand the input, return an empty string.

RULES:
1. Do NOT generate examples, demonstrations, or templates.
2. Do NOT output any extra text such as 'Example', 'Example Output', or similar.
3. Do NOT generate any tables, headings, or content that is not explicitly present in the image.
4. Transcribe content word-for-word. Do NOT modify, translate, or omit any content.
5. Do NOT explain Markdown or mention that you are using Markdown.
6. Do NOT wrap the output in ```markdown or ``` blocks.
7. Only apply Markdown structure to headings, paragraphs, lists, and tables, strictly based on the layout of the image. Do NOT create tables unless an actual table exists in the image.
8. Preserve the original language, information, and order exactly as shown in the image.
"""

    if page is not None:
        prompt_en += f"\nAt the end of the transcription, add the page divider: `--- Page {page} ---`."

    prompt_en += """
FAILURE HANDLING:
- If you do not detect valid content in the image, return an empty string.
"""
    return prompt_en


def vision_llm_figure_describe_prompt() -> str:
    prompt = """
You are an expert visual data analyst. Analyze the image and provide a comprehensive description of its content. Focus on identifying the type of visual data representation (e.g., bar chart, pie chart, line graph, table, flowchart), its structure, and any text captions or labels included in the image.

Tasks:
1. Describe the overall structure of the visual representation. Specify if it is a chart, graph, table, or diagram.
2. Identify and extract any axes, legends, titles, or labels present in the image. Provide the exact text where available.
3. Extract the data points from the visual elements (e.g., bar heights, line graph coordinates, pie chart segments, table rows and columns).
4. Analyze and explain any trends, comparisons, or patterns shown in the data.
5. Capture any annotations, captions, or footnotes, and explain their relevance to the image.
6. Only include details that are explicitly present in the image. If an element (e.g., axis, legend, or caption) does not exist or is not visible, do not mention it.

Output format (include only sections relevant to the image content):
- Visual Type: [Type]
- Title: [Title text, if available]
- Axes / Legends / Labels: [Details, if available]
- Data Points: [Extracted data]
- Trends / Insights: [Analysis and interpretation]
- Captions / Annotations: [Text and relevance, if available]

Ensure high accuracy, clarity, and completeness in your analysis, and includes only the information present in the image. Avoid unnecessary statements about missing elements.
"""
    return prompt



# ## 居中实现，图片和题注居中显示  250627
# def get_prompt_for_image_insertion(content: str, image_list_text: str, server_ip) -> str:
#     """
#     图文结构增强任务的 prompt 构造函数（新版，支持结构化标签）
#     参数 image_list_text：多行字符串，每行一个图片题注，格式示例：
#         <image_assets/doc_name/figures/pX_Y:题注>
#     """

#     # 把多行字符串拆成列表
#     image_list = image_list_text.splitlines()
#     # 再重新用换行连接，保持格式
#     image_str = "\n".join(image_list)
#     prompt = content

#     prompt += f"""
#     [IMAGE_LIST]
#     {image_str}
#     [/IMAGE_LIST]

#     请结合以下图片及题注，合理插入与内容相关的图片。
#     请使用提示语“如下图所示：”，图片标签单独另起一行。
#     插入时去掉题注，由<image_assets/doc_name/figures/pX_Y:题注>转为<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">，并将<image_assets/doc_name/figures/pX_Y:题注>替换成<doc_name/pX_Y:题注>保留另起一行放入<img src="http://10.103.238.124:8000/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">后面    
#     注意在最终显示的时候不要显示doc_name这个英文字母，而是要显示其代表的具体文件名
    

#     ### 要求：
#     - **仅插入与回答内容高度相关的图**，无关图不要插入；
#     - 如果是用户自己上传的文档的话，就不用进行图片显示，也不用根据知识库回答
#     - 插入位置应贴合上下文语义，使用提示语“如下图所示：”；
#     - 标签格式统一为：<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">；
#     - 如果某段插入了图，请在段落中使用提示语“如下图所示：”,居中显示图片；
#     - 输出为完整连贯的段落文本。
#     - 图片居中显示。
#     - 为了防止显示Markdown格式的删除线，回答时禁止出现波浪线字符，即禁止出现"~"与"~~"字符，也禁止出现~~2==等字符
#     - 为了防止显示HTML格式的删除线，回答时禁止出现"<s>"、"</s>"与"<del>"字符。
#     - 如果没有匹配到合适的<image_assets/doc_name/figures/pX_Y:题注>的话不要根据问题伪造，只匹配已经存在的
#     - 请在你的思维链中，每一步都力求简洁，快速聚焦于关键信息，避免冗余的细节和旁枝末节。整个思考过程不宜过长，目标是快速、准确地解决问题。
#     - 如果 [IMAGE_LIST] 中没有任何有效图片，请直接忽略插图任务，不生成任何图片标签或提示语，也不要造图。

#     ### 示例输出：
#     在模型结构中，输入图像首先经过特征提取模块，**如下图所示：**
#     \n
#     <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
#         <img src="{server_ip}/image_assets/document1/figures/p1_1.jpg" alt="图片" width="300">
#         # <figcaption>image_assets/doc_name/figures/pX_Y:题注</figcaption>
#         <figcaption>doc_name/pX_Y:题注</figcaption>
#     </div>
#     \n
#     注释doc_name代表具体的文件名，而不是doc_name这个单词
   
#     doc_name/pX_Y:题注 显示的时候应该用括号括住，比如《doc_name/pX_Y:题注》

#     随后，提取的特征将被送入融合层进行整合。
#     此外，对于不同分辨率的输入，还需进行多尺度处理。本图未展示该过程，故不做展开。本图未展示该过程，故不做展开。
    
#     """
#     return prompt


def get_prompt_for_image_insertion_si(content: str, image_list_text: str, server_ip) -> str:
    """
    图文结构增强任务的 prompt 构造函数（新版，支持结构化标签）
    参数 image_list_text：多行字符串，每行一个图片题注，格式示例：
        <image_assets/doc_name/figures/pX_Y:caption>
    """

    # 把多行字符串拆成列表
    image_list = image_list_text.splitlines()
    # 再重新用换行连接，保持格式
    image_str = "\n".join(image_list)
    prompt = content

    prompt += f"""
    [IMAGE_LIST]
    {image_str}
    [/IMAGE_LIST]

    ### 任务
    请结合以下图片及题注，合理插入与内容相关的图片。
    回答的语言要为僧伽罗语。
    请使用提示语“如下图所示：”或"පහත දර්ශන ලද රූපයේ පරිදි:"，回答的语言要为僧伽罗语，图片标签单独另起一行。
    插入时去掉题注，由<image_assets/doc_name/figures/pX_Y:caption>转为<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">，并将<image_assets/doc_name/figures/pX_Y:caption>替换成<doc_name/pX_Y:caption>保留另起一行放入<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">后面    
    注意在最终显示的时候不要显示doc_name这个英文字母，而是要显示其代表的具体文件名
    

    ### 要求：
    - **仅插入与回答内容高度相关的图**，无关图不要插入；
    - **请判断回答内容与提问是否相关，对于不相关的内容，不需要进行插图操作，不需要进行插图操作**；
    - **不要自己随便生成题注，从IMAGE_LIST中选取合适的图像题注**；
    - 标签格式统一为：<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">；
    - 如果某段插入了图，请在段落中使用提示语“如下图所示：”或"පහත දර්ශන ලද රූපයේ පරිදි:",回答的语言要为僧伽罗语，居中显示图片；
    - 输出为完整连贯的段落文本，回答的语言要为僧伽罗语。
    - 请在你的思维链中，每一步都力求简洁，快速聚焦于关键信息，避免冗余的细节和旁枝末节。整个思考过程不宜过长，目标是快速、准确地解决问题。

    ### 示例输出：
    在模型结构中，输入图像首先经过特征提取模块，**如下图所示：**
    \n
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="{server_ip}/image_assets/document1/figures/p1_1.jpg" alt="图片" width="300">
        # <figcaption>image_assets/doc_name/figures/pX_Y:caption</figcaption>
        <figcaption>doc_name/pX_Y:caption</figcaption>
    </div>
    \n
    随后，提取的特征将被送入融合层进行整合。此外，对于不同分辨率的输入，还需进行多尺度处理。本图未展示该过程，故不做展开。本图未展示该过程，故不做展开。
    
    注意doc_name代表具体的文件名，而不是doc_name这个单词
   
    doc_name/pX_Y:ප්‍රස්තාවනා 显示的时候应该用括号括住，比如《doc_name/pX_Y:ප්‍රස්තාවනා》，并使用僧伽罗语输出题注。
    """
    return prompt

def get_prompt_for_image_insertion_tai(content: str, image_list_text: str, server_ip) -> str:
    """
    图文结构增强任务的 prompt 构造函数（新版，支持结构化标签）
    参数 image_list_text：多行字符串，每行一个图片题注，格式示例：
        <image_assets/doc_name/figures/pX_Y:caption>
    """

    # 把多行字符串拆成列表
    image_list = image_list_text.splitlines()
    # 再重新用换行连接，保持格式
    image_str = "\n".join(image_list)
    prompt = content

    prompt += f"""
    [IMAGE_LIST]
    {image_str}
    [/IMAGE_LIST]

    ### 任务
    请结合以下图片及题注，合理插入与内容相关的图片。
    回答的语言要为泰语。
    请使用提示语“如下图所示：”或"ดังภาพ:"，回答的语言要为泰语，图片标签单独另起一行。
    插入时去掉题注，由<image_assets/doc_name/figures/pX_Y:caption>转为<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">，并将<image_assets/doc_name/figures/pX_Y:caption>替换成<doc_name/pX_Y:caption>保留另起一行放入<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">后面    
    注意在最终显示的时候不要显示doc_name这个英文字母，而是要显示其代表的具体文件名
    

    ### 要求：
    - **仅插入与回答内容高度相关的图**，无关图不要插入；
    - **请判断回答内容与提问是否相关，对于不相关的内容，不需要进行插图操作，不需要进行插图操作**；
    - **不要自己随便生成题注，从IMAGE_LIST中选取合适的图像题注**；
    - 标签格式统一为：<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">；
    - 如果某段插入了图，请在段落中使用提示语“如下图所示：”或"ดังภาพ:",回答的语言要为泰语，居中显示图片；
    - 输出为完整连贯的段落文本，回答的语言要为泰语。
    - 请在你的思维链中，每一步都力求简洁，快速聚焦于关键信息，避免冗余的细节和旁枝末节。整个思考过程不宜过长，目标是快速、准确地解决问题。

    ### 示例输出：
    在模型结构中，输入图像首先经过特征提取模块，**如下图所示：**
    \n
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="{server_ip}/image_assets/document1/figures/p1_1.jpg" alt="图片" width="300">
        # <figcaption>image_assets/doc_name/figures/pX_Y:caption</figcaption>
        <figcaption>doc_name/pX_Y:caption</figcaption>
    </div>
    \n
    随后，提取的特征将被送入融合层进行整合。此外，对于不同分辨率的输入，还需进行多尺度处理。本图未展示该过程，故不做展开。本图未展示该过程，故不做展开。
    
    注意doc_name代表具体的文件名，而不是doc_name这个单词
   
    doc_name/pX_Y:คำอธิบาย 显示的时候应该用括号括住，比如《doc_name/pX_Y:คำอธิบาย》，并使用泰语输出题注。
    """
    return prompt

## 居中实现，图片和题注居中显示  250711
def get_prompt_for_image_insertion(content: str, image_list_text: str, server_ip) -> str:
    """
    图文结构增强任务的 prompt 构造函数（新版，支持结构化标签）
    参数 image_list_text：多行字符串，每行一个图片题注，格式示例：
        <image_assets/doc_name/figures/pX_Y:caption>
    """

    # 把多行字符串拆成列表
    image_list = image_list_text.splitlines()
    # 再重新用换行连接，保持格式
    image_str = "\n".join(image_list)
    prompt = content

    prompt += f"""
    [IMAGE_LIST]
    {image_str}
    [/IMAGE_LIST]

    ### 任务
    请结合以下图片及题注，合理插入与内容相关的图片。
    回答的语言要为英文。
    请使用提示语“如下图所示：”或"As shown in the figure below:"，回答的语言要为英文，图片标签单独另起一行。
    插入时去掉题注，由<image_assets/doc_name/figures/pX_Y:caption>转为<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">，并将<image_assets/doc_name/figures/pX_Y:caption>替换成<doc_name/pX_Y:caption>保留另起一行放入<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">后面    
    注意在最终显示的时候不要显示doc_name这个英文字母，而是要显示其代表的具体文件名
    

    ### 要求：
    - **仅插入与回答内容高度相关的图**，无关图不要插入；
    - **请判断回答内容与提问是否相关，对于不相关的内容，不需要进行插图操作，不需要进行插图操作**；
    - **不要自己随便生成题注，从IMAGE_LIST中选取合适的图像题注**；
    - 标签格式统一为：<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">；
    - 如果某段插入了图，请在段落中使用提示语“如下图所示：”或"As shown in the figure below:",回答的语言要为英文，居中显示图片；
    - 输出为完整连贯的段落文本，回答的语言要为英文。
    - 请在你的思维链中，每一步都力求简洁，快速聚焦于关键信息，避免冗余的细节和旁枝末节。整个思考过程不宜过长，目标是快速、准确地解决问题。

    ### 示例输出：
    在模型结构中，输入图像首先经过特征提取模块，**如下图所示：**
    \n
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="{server_ip}/image_assets/document1/figures/p1_1.jpg" alt="图片" width="300">
        # <figcaption>image_assets/doc_name/figures/pX_Y:caption</figcaption>
        <figcaption>doc_name/pX_Y:caption</figcaption>
    </div>
    \n
    随后，提取的特征将被送入融合层进行整合。此外，对于不同分辨率的输入，还需进行多尺度处理。本图未展示该过程，故不做展开。本图未展示该过程，故不做展开。
    
    注意doc_name代表具体的文件名，而不是doc_name这个单词
   
    doc_name/pX_Y:caption 显示的时候应该用括号括住，比如《doc_name/pX_Y:captain》，并使用英文输出题注。
    """
    return prompt


def get_prompt_for_image_insertion_zh(content: str, image_list_text: str, server_ip) -> str:
    """
    图文结构增强任务的 prompt 构造函数（新版，支持结构化标签）
    参数 image_list_text：多行字符串，每行一个图片题注，格式示例：
        <image_assets/doc_name/figures/pX_Y:题注>
    """

    # 把多行字符串拆成列表
    image_list = image_list_text.splitlines()
    # 再重新用换行连接，保持格式
    image_str = "\n".join(image_list)
    prompt = content

    prompt += f"""
    [IMAGE_LIST]
    {image_str}
    [/IMAGE_LIST]

    ### 任务
    请结合以下图片及题注，合理插入与内容相关的图片。
    回答的语言要为中文。
    请使用提示语“如下图所示：”，回答的语言要为中文，图片标签单独另起一行。
    插入时去掉题注，由<image_assets/doc_name/figures/pX_Y:题注>转为<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">，并将<image_assets/doc_name/figures/pX_Y:题注>替换成<doc_name/pX_Y:题注>保留另起一行放入<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">后面    
    注意在最终显示的时候不要显示doc_name这个英文字母，而是要显示其代表的具体文件名
    

    ### 要求：
    - **仅插入与回答内容高度相关的图**，无关图不要插入；
    - **请判断回答内容与提问是否相关，对于不相关的内容，不需要进行插图操作，不需要进行插图操作**；
    - **不要自己随便生成题注，从IMAGE_LIST中选取合适的图像题注**；
    - 标签格式统一为：<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">；
    - 如果某段插入了图，请在段落中使用提示语“如下图所示：”,回答的语言要为中文，居中显示图片；
    - 输出为完整连贯的段落文本，回答的语言要为中文。
    - 请在你的思维链中，每一步都力求简洁，快速聚焦于关键信息，避免冗余的细节和旁枝末节。整个思考过程不宜过长，目标是快速、准确地解决问题。

    ### 示例输出：
    在模型结构中，输入图像首先经过特征提取模块，**如下图所示：**
    \n
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <img src="{server_ip}/image_assets/document1/figures/p1_1.jpg" alt="图片" width="300">
        # <figcaption>image_assets/doc_name/figures/pX_Y:题注</figcaption>
        <figcaption>doc_name/pX_Y:题注</figcaption>
    </div>
    \n

    随后，提取的特征将被送入融合层进行整合。此外，对于不同分辨率的输入，还需进行多尺度处理。本图未展示该过程，故不做展开。本图未展示该过程，故不做展开。
    
    注意doc_name代表具体的文件名，而不是doc_name这个单词
   
    doc_name/pX_Y:题注 显示的时候应该用括号括住，比如《doc_name/pX_Y:题注》
    """
    print("这是查找图片的prompt",prompt)
    return prompt

# def get_prompt_for_image_insertion(content: str, image_list_text: str, server_ip) -> str:
#     """
#     图文结构增强任务的 prompt 构造函数（新版，支持结构化标签）
#     参数 image_list_text：多行字符串，每行一个图片题注，格式示例：
#         <image_assets/doc_name/figures/pX_Y:题注>
#     """

#     # 把多行字符串拆成列表
#     image_list = image_list_text.splitlines()
    
#     # 将每个图片路径转化为具体的URL形式，并生成题注
#     image_urls = []
#     for entry in image_list:
#         # 解析doc_name、pX_Y和题注
#         parts = entry.split(":")
#         path_part = parts[0]  # <image_assets/doc_name/figures/pX_Y
#         caption = parts[1] if len(parts) > 1 else ""  # 题注
        
#         # 提取 doc_name 和 pX_Y
#         doc_name = path_part.split("/")[1]
#         pX_Y = path_part.split("/")[2]
        
#         # 转换为图片URL形式
#         image_url = f"{server_ip}/image_assets/{doc_name}/figures/{pX_Y}.jpg"
#         image_urls.append((image_url, caption))
    
#     # 拼接为图片部分的字符串
#     image_str = "\n".join([f'<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">'
#                           f'<img src="{url}" alt="图片" width="300">' 
#                           f'<figcaption>{caption}</figcaption>'
#                           f'</div>' for url, caption in image_urls])

#     prompt = content

#     prompt += f"""
#     [IMAGE_LIST]
#     {image_str}
#     [/IMAGE_LIST]

#     ### 任务
#     请结合以下图片及题注，合理插入与内容相关的图片。
#     请使用提示语“如下图所示：”，图片标签单独另起一行。
#     插入时去掉题注，由<image_assets/doc_name/figures/pX_Y:题注>转为<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">，并将<image_assets/doc_name/figures/pX_Y:题注>替换成<doc_name/pX_Y:题注>保留另起一行放入<img src="http://10.103.238.124:8000/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">后面    
#     注意在最终显示的时候不要显示doc_name这个英文字母，而是要显示其代表的具体文件名

#     ### 要求：
#     - **仅插入与回答内容高度相关的图**，无关图不要插入；
#     - 标签格式统一为：<img src="{server_ip}/image_assets/doc_name/figures/pX_Y.jpg" alt="图片" width="300">；
#     - 如果某段插入了图，请在段落中使用提示语“如下图所示：”，居中显示图片；
#     - 输出为完整连贯的段落文本。
#     - 请在你的思维链中，每一步都力求简洁，快速聚焦于关键信息，避免冗余的细节和旁枝末节。整个思考过程不宜过长，目标是快速、准确地解决问题。

#     ### 示例输出：
#     在模型结构中，输入图像首先经过特征提取模块，**如下图所示：**
#     \n
#     <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
#         <img src="{server_ip}/image_assets/document1/figures/p1_1.jpg" alt="图片" width="300">
#         <figcaption>doc_name/p1_1:题注</figcaption>
#     </div>
#     \n
#     注意doc_name代表具体的文件名，而不是doc_name这个单词
    
#     doc_name/pX_Y:题注 显示的时候应该用括号括住，比如《doc_name/pX_Y:题注》

#     随后，提取的特征将被送入融合层进行整合。此外，对于不同分辨率的输入，还需进行多尺度处理。本图未展示该过程，故不做展开。本图未展示该过程，故不做展开。
#     """
#     return prompt
