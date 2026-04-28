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
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import trafilatura
import json
import binascii
import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from timeit import default_timer as timer

from langfuse import Langfuse

from agentic_reasoning import DeepResearcher
from api import settings
from api.db import LLMType, ParserType, StatusEnum
from api.db.db_models import DB, Dialog
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.langfuse_service import TenantLangfuseService
from api.db.services.llm_service import LLMBundle, TenantLLMService
from api.utils import current_timestamp, datetime_format
from rag.app.resume import forbidden_select_fields4resume
from rag.app.tag import label_question
from rag.nlp.search import index_name
from rag.prompts import chunks_format, citation_prompt, cross_languages, full_question, kb_prompt, keyword_extraction, \
    llm_id2llm_type, message_fit_in, get_prompt_for_image_insertion, get_prompt_for_image_insertion_zh, get_prompt_for_image_insertion_si, get_prompt_for_image_insertion_tai
from rag.utils import num_tokens_from_string, rmSpace
from rag.utils.tavily_conn import Tavily
from dotenv import load_dotenv
import os
from ddgs import DDGS
import json




class DialogService(CommonService):
    model = Dialog

    @classmethod
    def save(cls, **kwargs):
        """Save a new record to database.

        This method creates a new record in the database with the provided field values,
        forcing an insert operation rather than an update.

        Args:
            **kwargs: Record field values as keyword arguments.

        Returns:
            Model instance: The created record object.
        """
        sample_obj = cls.model(**kwargs).save(force_insert=True)
        return sample_obj

    @classmethod
    def update_many_by_id(cls, data_list):
        """Update multiple records by their IDs.

        This method updates multiple records in the database, identified by their IDs.
        It automatically updates the update_time and update_date fields for each record.

        Args:
            data_list (list): List of dictionaries containing record data to update.
                             Each dictionary must include an 'id' field.
        """
        with DB.atomic():
            for data in data_list:
                data["update_time"] = current_timestamp()
                data["update_date"] = datetime_format(datetime.now())
                cls.model.update(data).where(cls.model.id == data["id"]).execute()

    @classmethod
    @DB.connection_context()
    def get_list(cls, tenant_id, page_number, items_per_page, orderby, desc, id, name):
        chats = cls.model.select()
        if id:
            chats = chats.where(cls.model.id == id)
        if name:
            chats = chats.where(cls.model.name == name)
        chats = chats.where((cls.model.tenant_id == tenant_id) & (cls.model.status == StatusEnum.VALID.value))
        if desc:
            chats = chats.order_by(cls.model.getter_by(orderby).desc())
        else:
            chats = chats.order_by(cls.model.getter_by(orderby).asc())

        chats = chats.paginate(page_number, items_per_page)

        return list(chats.dicts())


def chat_solo(dialog, messages, stream=True):
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    msg = [{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if
           m["role"] != "system"]
    if stream:
        last_ans = ""
        delta_ans = ""
        for ans in chat_mdl.chat_streamly(prompt_config.get("system", ""), msg, dialog.llm_setting):
            answer = ans
            delta_ans = ans[len(last_ans):]
            if num_tokens_from_string(delta_ans) < 16:
                continue
            last_ans = answer
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "",
                   "created_at": time.time()}
            delta_ans = ""
        if delta_ans:
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "",
                   "created_at": time.time()}
    else:
        answer = chat_mdl.chat(prompt_config.get("system", ""), msg, dialog.llm_setting)
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, answer), "prompt": "",
               "created_at": time.time()}


def chat(dialog, messages, stream=True, **kwargs):
    # os.environ["IS_BIN_UPLOADED"] = "0"
    print("000-os.environ[\"IS_BIN_UPLOADED\"] = ", os.environ["IS_BIN_UPLOADED"])
    print("chatchatchatchatchatchatchatchat")
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    if not dialog.kb_ids:
        for ans in chat_solo(dialog, messages, stream):
            yield ans
        return

    chat_start_ts = timer()

    # 判断大模型类型 “图像转文本” 还是“chat聊天”模型
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    max_tokens = llm_model_config.get("max_tokens", 8192)

    check_llm_ts = timer()

    # 判断是否使用追踪器（一个用于日志追踪、分析的服务）
    langfuse_tracer = None
    langfuse_keys = TenantLangfuseService.filter_by_tenant(tenant_id=dialog.tenant_id)
    if langfuse_keys:
        langfuse = Langfuse(public_key=langfuse_keys.public_key, secret_key=langfuse_keys.secret_key,
                            host=langfuse_keys.host)
        if langfuse.auth_check():
            langfuse_tracer = langfuse
            langfuse.trace = langfuse_tracer.trace(name=f"{dialog.name}-{llm_model_config['llm_name']}")

    check_langfuse_tracer_ts = timer()

    # 判断知识库中的内容是否采用同一个embedding模型
    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)
    print("**************mmmmmm********************")
    print("dialog.kb_ids of kbs: ", dialog.kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))
    if len(embedding_list) != 1:
        yield {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}
        return {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}

    embedding_model_name = embedding_list[0]

    retriever = settings.retrievaler  # 从 settings 的属性获得检索器并且赋值

    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]  # 最近三条的用户的问题结合，列表解析
    # 从参数kwargs中获得doc_ids，作为当前检索或回答时需要用到的文档集合；处理附加的文档ID（如果传入），方便后续知识检索时限制范围。
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]  # 如果用户提问中提到了doc_ids，优先按照用户提示的，

    create_retriever_ts = timer()

    # 创建嵌入模型实例
    embd_mdl = LLMBundle(dialog.tenant_id, LLMType.EMBEDDING,
                         embedding_model_name)  # LLMBundle 就像是一个“语言模型工具箱”，里面装着不同种类的大语言模型组件
    if not embd_mdl:
        raise LookupError("Embedding model(%s) not found" % embedding_model_name)

    bind_embedding_ts = timer()

    print("dialog.tenant_id = ", dialog.tenant_id)  # 594b22123a0511f0b0815b6bf5900da2
    print("dialog.llm_id = ", dialog.llm_id)  # deepseek-chat@DeepSeek
    print("llm_id2llm_type(dialog.llm_id) = ", llm_id2llm_type(dialog.llm_id))  # llm_id2llm_type(dialog.llm_id) =  t
    print("LLMType.CHAT = ", LLMType.CHAT)  # chat

    # 创建聊天模型实例
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)
        toolcall_session, tools = kwargs.get("toolcall_session"), kwargs.get("tools")  # 如果有工具调用会话和工具，绑定它们到聊天模型
        if toolcall_session and tools:
            chat_mdl.bind_tools(toolcall_session, tools)

    bind_llm_ts = timer()

    prompt_config = dialog.prompt_config  # 获取提示词配置
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)  # 知识库集合中的字段映射关系
    # 语音合成模型初始化（TTS）
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # try to use sql if field mapping is good to go 执行基于SQL的检索并返回答案ans (应该不会执行)
    if field_map:
        logging.debug("Use SQL to retrieval:{}".format(questions[-1]))
        ans = use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl,
                      prompt_config.get("quote", True))  # prompt_config.get("quote", True)提示配置中获取是否启用引用功能，默认为 True。
        if ans:
            yield ans
            return

    # 参数检查与默认值的替换，判断是否有知识库
    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            continue
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Miss parameter: " + p["key"])  # 如果参数的键不在传入的参数字典kwargs中，且该参数不可选，就抛出异常
        if p["key"] not in kwargs:  # 如果参数键不在 kwargs（且是可选参数），则将提示词模板中对应的占位符 {key} 替换为空格。
            prompt_config["system"] = prompt_config["system"].replace("{%s}" % p["key"], " ")

    if len(questions) > 1 and prompt_config.get("refine_multiturn"):
        questions = [
            full_question(dialog.tenant_id, dialog.llm_id, messages)]  # 这一步很关键，利用prompt和chat生成上下文消息，最终多条上下文转化成一条问题
    else:
        questions = questions[-1:]

    # 将问题翻译成多种跨语言，并存入questions中，换行存储成多行文本
    print("prompt参数", prompt_config)
    if prompt_config.get("cross_languages"):
        questions = [cross_languages(dialog.tenant_id, dialog.llm_id, questions[0], prompt_config["cross_languages"])]
        print("翻译后的问题", questions)
    refine_question_ts = timer()

    # 构建重排序模型：用于对已有检索结果进行二次排序的机器学习模型，因为初次检索不精确（挺重要）
    rerank_mdl = None
    if dialog.rerank_id:
        rerank_mdl = LLMBundle(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)

    bind_reranker_ts = timer()
    generate_keyword_ts = bind_reranker_ts
    thought = ""  # 用以存储思考过程
    kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}  # 初始化一个字典kbinfos：信息总条数、文本块、文档聚合信息

    if "knowledge" not in [p["key"] for p in prompt_config["parameters"]]:
        knowledges = []
    else:
        if prompt_config.get("keyword", False):
            questions[-1] += keyword_extraction(chat_mdl, questions[-1])
            generate_keyword_ts = timer()

        tenant_ids = list(set([kb.tenant_id for kb in kbs]))  # 知识库所有租户的ID

        # 调用推理器或直接检索
        knowledges = []
        if prompt_config.get("reasoning", False):
            reasoner = DeepResearcher(
                chat_mdl,
                prompt_config,
                partial(retriever.retrieval, embd_mdl=embd_mdl, tenant_ids=tenant_ids, kb_ids=dialog.kb_ids, page=1,
                        page_size=dialog.top_n, similarity_threshold=0.2, vector_similarity_weight=0.3),
            )

            for think in reasoner.thinking(kbinfos, " ".join(questions)):
                if isinstance(think, str):
                    thought = think
                    knowledges = [t for t in think.split("\n") if t]
                elif stream:
                    yield think
        else:
            print("#####dialog参数", dialog.top_n, dialog.top_k)
            kbinfos = retriever.retrieval(
                " ".join(questions),
                embd_mdl,  # 嵌入模型实例
                tenant_ids,
                dialog.kb_ids,  # 知识库ID列表
                1,
                dialog.top_n,  # rerank最终返回的参考文档数量，期望返回的最大结果数（候选文档数量）
                dialog.similarity_threshold,  # 相似度阈值
                dialog.vector_similarity_weight,  # 向量相似度权重
                doc_ids=attachments,  # 限定检索的文档ID的集合
                top=dialog.top_k,  # 初次检索，返回最优的文档的数量
                aggs=False,  # 是否执行聚合操作，False不聚合
                rerank_mdl=rerank_mdl,  # 重排序模型实例，用于二次排序
                rank_feature=label_question(" ".join(questions), kbs),  # 排序特征（基于问题与知识库标签的评分）
            )
            # print("doc_aggs内容:", kbinfos.get("doc_aggs", []))
            if prompt_config.get("tavily_api_key"):
                tav = Tavily(prompt_config["tavily_api_key"])
                tav_res = tav.retrieve_chunks(" ".join(questions))
                kbinfos["chunks"].extend(tav_res["chunks"])
                kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
            if prompt_config.get("use_kg"):
                ck = settings.kg_retrievaler.retrieval(" ".join(questions), tenant_ids, dialog.kb_ids, embd_mdl,
                                                       LLMBundle(dialog.tenant_id, LLMType.CHAT))
                if ck["content_with_weight"]:
                    kbinfos["chunks"].insert(0, ck)

            # 构建知识提示
            knowledges = kb_prompt(kbinfos, max_tokens)

    logging.debug("{}->{}".format(" ".join(questions), "\n->".join(knowledges)))

    retrieval_ts = timer()
    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {"answer": empty_res, "reference": kbinfos, "prompt": "\n\n### Query:\n%s" % " ".join(questions),
               "audio_binary": tts(tts_mdl, empty_res)}
        return {"answer": prompt_config["empty_response"], "reference": kbinfos}

    kwargs["knowledge"] = "\n------\n" + "\n\n------\n\n".join(knowledges)

    gen_conf = dialog.llm_setting
    print("************0000*********")

    # print("kwargs: ",kwargs["knowledge"])

    print("************1111*********")
    # image_list_text = get_default_image_list_text()
    # kwargs["knowledge"]  = get_prompt_for_image_insertion(dialog.tenant_id, dialog.llm_id, kwargs["knowledge"] , image_list_text)
    # print("kwargs: ", kwargs["knowledge"])

    print("************1111*********")

    # 这一步很关键
    msg = [{"role": "system",
            "content": prompt_config["system"].format(**kwargs)}]  # 这一步用**kwargs的knowledges填充原先系统提示词部分的knowledge
    prompt4citation = ""
    if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
        prompt4citation = citation_prompt()
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if
                m["role"] != "system"])
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.95))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    prompt = msg[0]["content"]  # 这就是系统提示，包含有knowledge，所以后面检索可以不需要系统提示词

    print("**********wsmzy*************")
    # print("prompt: ",prompt)
    # image_list_text = get_default_image_list_text()
    # prompt = get_prompt_for_image_insertion(prompt,image_list_text)
    # print("prompt2: ", prompt)
    print("**********wsmzy*************")

    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(gen_conf["max_tokens"], max_tokens - used_token_count)

    def repair_bad_citation_formats(answer: str, kbinfos: dict, idx: set):
        max_index = len(kbinfos["chunks"])

        def safe_add(i):
            if 0 <= i < max_index:
                idx.add(i)
                return True
            return False

        def find_and_replace(pattern, group_index=1, repl=lambda i: f"##{i}$$", flags=0):
            nonlocal answer
            for match in re.finditer(pattern, answer, flags=flags):
                try:
                    i = int(match.group(group_index))
                    if safe_add(i):
                        answer = answer.replace(match.group(0), repl(i))
                except Exception:
                    continue

        find_and_replace(r"\(\s*ID:\s*(\d+)\s*\)")  # (ID: 12)
        find_and_replace(r"ID[: ]+(\d+)")  # ID: 12, ID 12
        find_and_replace(r"\$\$(\d+)\$\$")  # $$12$$
        find_and_replace(r"\$\[(\d+)\]\$")  # $[12]$
        find_and_replace(r"\$\$(\d+)\${2,}")  # $$12$$$$
        find_and_replace(r"\$(\d+)\$")  # $12$
        find_and_replace(r"(#{2,})(\d+)(\${2,})", group_index=2)  # 2+ # and 2+ $
        find_and_replace(r"(#{2,})(\d+)(#{1,})", group_index=2)  # 2+ # and 1+ #
        find_and_replace(r"##(\d+)#{2,}")  # ##12###
        find_and_replace(r"【(\d+)】")  # 【12】
        find_and_replace(r"ref\s*(\d+)", flags=re.IGNORECASE)  # ref12, ref 12, REF 12

        return answer, idx

    def decorate_answer(answer):
        # print("mzy = 0")
        nonlocal prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_ts, questions, langfuse_tracer
        # print("mzy = 1")
        refs = []
        # print("mzy = 2")
        ans = answer.split("</think>")
        # print("mzy = 3")
        think = ""
        # print("mzy = 4")
        if len(ans) == 2:
            # print("mzy = 5")
            think = ans[0] + "</think>"
            # print("mzy = 6")
            answer = ans[1]
            # print("mzy = 7")
        # print("mzy = 8")
        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            # print("mzy = 9")
            answer = re.sub(r"##[ij]\$\$", "", answer, flags=re.DOTALL)
            # print("mzy = 10")
            idx = set([])
            # print("mzy = 11")
            if not re.search(r"##[0-9]+\$\$", answer):
                # print("mzy = 12")
                answer, idx = retriever.insert_citations(
                    answer,
                    [ck["content_ltks"] for ck in kbinfos["chunks"]],
                    [ck["vector"] for ck in kbinfos["chunks"]],
                    embd_mdl,
                    tkweight=1 - dialog.vector_similarity_weight,
                    vtweight=dialog.vector_similarity_weight,
                )
                # print("mzy = 12")
            else:
                # print("mzy = 14")
                for match in re.finditer(r"##([0-9]+)\$\$", answer):
                    i = int(match.group(1))
                    if i < len(kbinfos["chunks"]):
                        idx.add(i)
            # print("mzy = 15")
            answer, idx = repair_bad_citation_formats(answer, kbinfos, idx)
            # print("mzy = 16")
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            # print("mzy = 17")
            recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            # print("mzy = 18")
            if not recall_docs:
                # print("mzy = 19")
                recall_docs = kbinfos["doc_aggs"]
            # print("mzy = 20")
            kbinfos["doc_aggs"] = recall_docs
            # print("mzy = 21")
            print("doc_aggs内容(筛选只有引用的，优化的):", kbinfos.get("doc_aggs", []))
            # image_list_str_test = generate_image_list_from_kbinfos(kbinfos)
            # print("image_list_str_test: ", image_list_str_test)
            # print("mzy = 22")
            refs = deepcopy(kbinfos)
            # print("mzy = 23")
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]
        # print("mzy = 24")
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model providers -> API-Key'"
        finish_chat_ts = timer()
        # print("mzy = 25")
        total_time_cost = (finish_chat_ts - chat_start_ts) * 1000
        check_llm_time_cost = (check_llm_ts - chat_start_ts) * 1000
        check_langfuse_tracer_cost = (check_langfuse_tracer_ts - check_llm_ts) * 1000
        create_retriever_time_cost = (create_retriever_ts - check_langfuse_tracer_ts) * 1000
        bind_embedding_time_cost = (bind_embedding_ts - create_retriever_ts) * 1000
        bind_llm_time_cost = (bind_llm_ts - bind_embedding_ts) * 1000
        refine_question_time_cost = (refine_question_ts - bind_llm_ts) * 1000
        bind_reranker_time_cost = (bind_reranker_ts - refine_question_ts) * 1000
        generate_keyword_time_cost = (generate_keyword_ts - bind_reranker_ts) * 1000
        retrieval_time_cost = (retrieval_ts - generate_keyword_ts) * 1000
        generate_result_time_cost = (finish_chat_ts - retrieval_ts) * 1000
        # print("mzy = 26")
        tk_num = num_tokens_from_string(think + answer)
        prompt += "\n\n### Query:\n%s" % " ".join(questions)
        prompt = (
            f"{prompt}\n\n"
            "## Time elapsed:\n"
            f"  - Total: {total_time_cost:.1f}ms\n"
            f"  - Check LLM: {check_llm_time_cost:.1f}ms\n"
            f"  - Check Langfuse tracer: {check_langfuse_tracer_cost:.1f}ms\n"
            f"  - Create retriever: {create_retriever_time_cost:.1f}ms\n"
            f"  - Bind embedding: {bind_embedding_time_cost:.1f}ms\n"
            f"  - Bind LLM: {bind_llm_time_cost:.1f}ms\n"
            f"  - Multi-turn optimization: {refine_question_time_cost:.1f}ms\n"
            f"  - Bind reranker: {bind_reranker_time_cost:.1f}ms\n"
            f"  - Generate keyword: {generate_keyword_time_cost:.1f}ms\n"
            f"  - Retrieval: {retrieval_time_cost:.1f}ms\n"
            f"  - Generate answer: {generate_result_time_cost:.1f}ms\n\n"
            "## Token usage:\n"
            f"  - Generated tokens(approximately): {tk_num}\n"
            f"  - Token speed: {int(tk_num / (generate_result_time_cost / 1000.0))}/s"
        )
        # print("mzy = 27")
        langfuse_output = "\n" + re.sub(r"^.*?(### Query:.*)", r"\1", prompt, flags=re.DOTALL)
        langfuse_output = {"time_elapsed:": re.sub(r"\n", "  \n", langfuse_output), "created_at": time.time()}
        # print("mzy = 28")
        # Add a condition check to call the end method only if langfuse_tracer exists
        if langfuse_tracer and "langfuse_generation" in locals():
            langfuse_generation.end(output=langfuse_output)
        # print("mzy = 29")
        return {"answer": think + answer, "reference": refs, "prompt": re.sub(r"\n", "  \n", prompt),
                "created_at": time.time()}

    if langfuse_tracer:
        langfuse_generation = langfuse_tracer.trace.generation(name="chat", model=llm_model_config["llm_name"],
                                                               input={"prompt": prompt,
                                                                      "prompt4citation": prompt4citation,
                                                                      "messages": msg})

    if stream:
        last_ans = ""
        answer = ""
        print("doc_aggs内容(初步检索试一下):", kbinfos.get("doc_aggs", []))
        print("question:", msg[-1])
        # 从你的消息中提取问题内容
        question_content = msg[-1]['content']
        print(f"question_content = {question_content}")
        # 设置标志位
        # FLAG = 1 if is_chinese(question_content) else 0
        if is_chinese(question_content):
            FLAG = 1
        elif is_sinhala(question_content):
            FLAG = 2
        elif is_tai(question_content):
            FLAG = 3
        else:
            FLAG = 0
        if FLAG != 1:
            new_system_content = {'role': 'system', 'content': 'You are a super intelligent assistant. Please summarize the knowledge base content to answer questions, providing detailed responses with data from the knowledge base. When none of the knowledge base content is relevant to the question, your response must include the phrase "The answer you are looking for is not found in the knowledge base!" At this point, you should provide an answer using common knowledge. Responses should consider the chat history. \n        Below is the knowledge base: \n        \n------ \n\n        The above is the knowledge base.'}
            msg[0] = new_system_content
            prompt = prompt.replace("知识库中未找到您要的答案！", "The answer you are looking for is not found in the knowledge base!")

        print(f"FLAG = {FLAG}")
        if prompt_config.get("web_search"):

            # 执行网络搜索，获取整合后的搜索结果
            web_search_context = run_web_search(
                original_query=question_content,
                llm_api_url='https://api.deepseek.com/v1/chat/completions',
                llm_api_key='YOUR_LLM_API_KEY',
                max_results_per_entity=3  # 控制结果数量，避免提示词过长
            )

            # 将网络信息整合到系统提示词中（你的原有逻辑）
            enhanced_system_content = msg[0]['content'] + f"\n\n=== 实时网络信息 ===\n{web_search_context}"
            msg[0]['content'] = enhanced_system_content

        image_list_text = generate_image_list_from_kbinfos(kbinfos)
        load_dotenv()
        server_ip = os.getenv('SERVER_IP')

        system_prompt = (
            "你是一个智能助手，请判断用户提出的问题是否属于电磁、通信、人工智能、计算机、机器学习、深度学习、信号处理、自然语言处理、环境测试条件或电气测试领域等领域范围。\n\n"
            "请先判断用户问题属于哪个学科领域，再判断是否与上述领域相关。相关的话只输出Yes,不相关输出No\n"
            "最终请只输出 **Yes** 或 **No**，不要包含空格、标点或解释说明。"
        )

        current_user_question = [{"role": "user", "content": msg[-1]["content"]}]
        for ans in chat_mdl.chat_streamly(system_prompt, current_user_question, dialog.llm_setting):
            answer_if = ans
        answer_if = answer_if

        answer_if = re.sub('<think>.*?</think>', '', answer_if, flags=re.DOTALL)
        answer_if = re.sub('<seed:think>.*?</seed:think>', '', answer_if, flags=re.DOTALL)

        answer_if = 'Yes'  # 回答所有领域
        print("answer_if = ", answer_if)
        if answer_if == 'Yes':
            # print("属于电磁领域知识")
            # print("image_list_text.strip() = ",image_list_text.strip())
            # print("image_list_text = ",image_list_text)
            # 如果 image_list_text 非空才调用 prompt 插图构造逻辑
            if image_list_text.strip():
                # 新增检索回答与问题是否相关判断，用于决定是否需要插图，否则会出现回答无关但仍然插图的情况
                image_sys_prompt = (
                "你要判断：当前知识库检索结果是否 ** 足以支持回答 ** 用户问题。\n"
                "我会给你用户问题 Q 和若干知识片段 K。\n\n"
                "请你在心里先尝试：能不能只依靠 K 来回答 Q。\n"
                "如果只是出现了类似的词（例如只提到了 Windows、版本号、操作系统环境等），"
                "但不能给出该概念的定义、含义、原理、步骤或核心结论，则必须回答 No。\n"
                "只有当你可以基于 K 写出一个比较完整、信息确实来自 K 的回答时，才回答 Yes。\n\n"
                "注意：\n"
                "1. 仅关键词重合（例如文档里只是说“支持 Windows 7/8/10”，"
                "而问题是“什么是 Windows 系统”）这种情况要判定为 No；\n"
                "2. 不能编造文档里没有的内容；\n"
                "3. 最终只输出 Yes 或 No，不要包含任何解释或其他符号。"
                )

                kb_preview = "\n\n".join(
                    [ck.get("content_ltks", "") for ck in kbinfos.get("chunks", [])[:5]]
                ) or "（无检索结果）"
                # print("回答的内容展示:::::",kb_preview)
                image_question_msg = [{
                    "role": "user",
                    "content": f"用户问题：\n{msg[-1]['content']}\n\n知识库检索内容预览：\n{kb_preview}"
                }]

                for ans_img in chat_mdl.chat_streamly(
                        image_sys_prompt, image_question_msg, dialog.llm_setting
                ):
                    image_if = ans_img

                image_if = re.sub('<think>.*?</think>', '', image_if, flags=re.DOTALL)
                image_if = re.sub('<seed:think>.*?</seed:think>', '', image_if, flags=re.DOTALL)
                image_if = 'Yes'
                print("image_if = ", image_if)
                if image_if == 'Yes':

                    if FLAG == 0:
                        msg[-1]["content"] = get_prompt_for_image_insertion(msg[-1]["content"], image_list_text,
                                                                            server_ip)
                    elif FLAG == 1:
                        msg[-1]["content"] = get_prompt_for_image_insertion_zh(msg[-1]["content"], image_list_text,
                                                                               server_ip)
                    elif FLAG == 2:
                        msg[-1]["content"] = get_prompt_for_image_insertion_si(msg[-1]["content"], image_list_text,
                                                                               server_ip)
                    elif FLAG == 3:
                        msg[-1]["content"] = get_prompt_for_image_insertion_tai(msg[-1]["content"], image_list_text,
                                                                               server_ip)
                else:
                    print("知识与问题无关，跳过插图任务")

            else:
                print(" image_list_text 为空，跳过插图任务")
            print("msg[-1]:", msg[-1])
            # print("msg[1:]:", msg[1:])
            html_intro = ('')
            ## bin文件上传显示逻辑
            if os.getenv('IS_BIN_UPLOADED') == '1':
                # ✅ 仅仅在回答前拼接固定化图文 HTML 内容
                pic_directory = os.getenv('BIN_FILE_SAVE_DIR')

                # 允许的图片后缀
                image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
                bin_file_name = os.getenv('BIN_FILE_NAME')
                bin_directory = os.path.dirname(pic_directory)
                # 初始化图片路径列表
                path_pic_file = []
                file_name_pic = []
                # 安全检查：路径存在且为文件夹
                if pic_directory and os.path.isdir(pic_directory):
                    # 遍历文件夹下所有文件
                    for filename in os.listdir(pic_directory):
                        # 只保留图片文件（后缀小写匹配）
                        if os.path.splitext(filename)[1].lower().strip() in image_extensions:
                            # 构造完整路径
                            file_name_pic.append(filename)
                            full_path = os.path.join(pic_directory, filename)
                            path_pic_file.append(full_path)
                else:
                    print("❌ pic_directory 无效或不存在")
                if len(path_pic_file) >= 2:

                    path_picture_1 = f"{server_ip}/{path_pic_file[0]}"
                    path_picture_2 = f"{server_ip}/{path_pic_file[1]}"

                    label_1 = f"《{file_name_pic[0]}》"
                    label_2 = f"《{file_name_pic[1]}》 "
                    html_intro = (
                        f'根据接收到的{bin_file_name}得到信号的相关信息'
                        f'所得到的信号解析结果与频谱图像为: \n'
                        f'<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">\n'
                        f'    <img src="{path_picture_1}" alt="图片" width="300">\n'
                        f'    <figcaption>{label_1}</figcaption>\n'
                        f'</div>\n\n'

                        f'<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">\n'
                        f'    <img src="{path_picture_2}" alt="图片" width="300">\n'
                        f'    <figcaption>{label_2}</figcaption>\n'
                        f'</div>\n\n'
                    )

                else:
                    print("❌ 图片文件不足两张，无法生成 html_intro")
                    # html_intro = ('')
            else:
                print("未上上传Bin附件")
                # html_intro = ('')

            ## 图片上传显示逻辑
            if os.getenv('IS_FIG_UPLOADED') == '1':
                path_pic_upload_file = []
                pic_upload_filename = []
                htmlpath_upload_pic = []

                pic_upload_base_dir = os.getenv('pic_upload_base_dir')
                for subfolder in os.listdir(pic_upload_base_dir):
                    subfolder_path = os.path.join(pic_upload_base_dir, subfolder)
                    if not os.path.isdir(subfolder_path):
                        continue  # 只处理子目录
                    for pic_upload_name in os.listdir(subfolder_path):
                        picitem_upload_path = os.path.join(subfolder_path, pic_upload_name)
                        path_pic_upload_file.append(picitem_upload_path)
                        pic_upload_filename.append(pic_upload_name)
                        htmlpath_upload_pic.append(f"{server_ip}/{picitem_upload_path}")

                for i in range(len(pic_upload_filename)):
                    html_intro_pic_upload = (
                        f'您上传的场景图{pic_upload_filename[i]}展示如下：'
                        f'<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">\n'
                        f'    <img src="{htmlpath_upload_pic[i]}" alt="图片" width="300">\n'
                        f'    <figcaption>{pic_upload_filename[i]}</figcaption>\n'
                        f'</div>\n\n'
                    )
                    html_intro += html_intro_pic_upload

            # print("msg0:", msg)

            # if len(msg) > 4:
            #     print(f"原始对话轮数为 {len(msg)/2}，下面进行截断")
            #     msg = msg[-5:]
            #     print(f"截断后，对话轮数为 {(len(msg) - 1) / 2}。")

            # 默认使用思考模式
            msg_with_think = msg.copy()
            msg_with_think[-1]["content"] += " "
            print("prompt4citation:", prompt4citation)
            print("msg_with_think:", msg_with_think)
            print("prompt:", prompt)
            length = len(prompt)
            print("length:", length)
            if prompt_config.get("web_search"):
                print(f"web_search已开启")

            # print("prompt4citation_250828 =  ",prompt4citation)
            # print("prompt_250828 = ",prompt) # 包含知识库的系统提示词

            # 请问：提示词是不是无限长？如果是能否在user_content中放的插入图片的提示词部分放入Prompt中？大模型调用的内部构造？
            for ans in chat_mdl.chat_streamly(prompt + prompt4citation, msg_with_think, gen_conf):
                # 检查是否包含 Error
                if "Error" in ans:
                    print("检测到 Error，切换到非思考模式")
                    # 切换到非思考模式
                    msgwiththink = msg.copy()
                    msgwiththink[-1]["content"] += " /nothink"
                    # print("prompt4citation:", prompt4citation)
                    # print("prompt:", prompt)
                    # print("msgwiththink:", msgwiththink)
                    length = len(prompt)
                    if length > 5000:
                        prompt = prompt[:5000]
                        # print("prompt:", prompt)
                        print("截取后的prompt长度:", len(prompt))
                    # 重新生成回答（非思考模式）
                    for ans in chat_mdl.chat_streamly(prompt + prompt4citation, msgwiththink, gen_conf):
                        if thought:
                            ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
                        answer = ans
                        delta_ans = answer[
                                    len(last_ans):]  # 提取出上次回答之后流式生成的新增部分    # 原先句子是delta_ans = ans[len(last_ans) :]
                        if num_tokens_from_string(delta_ans) < 16:  # 每新生成16个更新一次前端显示
                            continue
                        fist_8_0 = answer[:7]
                        last_8_0 = answer[-8:]
                        if fist_8_0 == '<think>' and last_8_0 != '</think>' and num_tokens_from_string(answer) > 8:
                            # 1.查找answer中</think>的位置
                            # 2.在answer的</think>后面插入html_intro
                            # 3.原先answer的</think>后面的内容后移就好
                            end_think_index = answer.find("</think>")
                            if end_think_index != -1:
                                before_think = answer[:end_think_index + len("</think>")]
                                after_think = answer[end_think_index + len("</think>"):]
                                answer = before_think + html_intro + after_think
                                # print("✅ 插入 html_intro 成功")
                        elif fist_8_0 != '<think>' and num_tokens_from_string(answer) > 8:
                            answer = html_intro + answer
                            # print("⚠️ 非 <think> 阶段，已将 html_intro 插入 answer 开头")
                        answer = "~~" + answer + "~~"
                        answer = clean_formatting(answer)
                        last_ans = ans  # 累计历史流式生成的回答
                        yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
                    break  # 重新生成回答后，跳出循环
                else:
                    if thought:
                        ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
                    answer = ans
                    delta_ans = answer[len(last_ans):]  # 提取出上次回答之后流式生成的新增部分    # 原先句子是delta_ans = ans[len(last_ans) :]
                    if num_tokens_from_string(delta_ans) < 16:  # 每新生成16个更新一次前端显示
                        continue
                    fist_8_0 = answer[:7]
                    last_8_0 = answer[-8:]
                    if fist_8_0 == '<think>' and last_8_0 != '</think>' and num_tokens_from_string(answer) > 8:
                        # 1.查找answer中</think>的位置
                        # 2.在answer的</think>后面插入html_intro
                        # 3.原先answer的</think>后面的内容后移就好
                        end_think_index = answer.find("</think>")
                        if end_think_index != -1:
                            before_think = answer[:end_think_index + len("</think>")]
                            after_think = answer[end_think_index + len("</think>"):]
                            answer = before_think + html_intro + after_think
                            # print("✅ 插入 html_intro 成功")
                    elif fist_8_0 != '<think>' and num_tokens_from_string(answer) > 8:
                        answer = html_intro + answer
                        # print("⚠️ 非 <think> 阶段，已将 html_intro 插入 answer 开头")
                    # 插入特殊字符清理函数，清洗特殊字符
                    # answer = "~~" + answer + "~~"
                    answer = clean_formatting(answer)
                    last_ans = ans  # 累计历史流式生成的回答
                    yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
            # print("woshimozhenyang")
            # print("Answer: ",answer)
            # print("test2")
            fist_8_0 = answer[:7]
            last_8_0 = answer[-8:]
            if fist_8_0 == '<think>' and last_8_0 != '</think>' and num_tokens_from_string(answer) > 8:
                end_think_index = answer.find("</think>")
                if end_think_index != -1:
                    before_think = answer[:end_think_index + len("</think>")]
                    after_think = answer[end_think_index + len("</think>"):]
                    answer = before_think + html_intro + after_think
            elif fist_8_0 != '<think>' and num_tokens_from_string(answer) > 8:
                answer = html_intro + answer
            answer = clean_formatting(answer)
            delta_ans = answer[len(last_ans):]
            # delta_ans = answer[len(last_ans)+len(html_intro) :]
            # print("answermmmmm=",answer)
            if delta_ans:
                yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
                print("wsmzy")
            # print("answer:",decorate_answer(thought + answer))
            # import sys
            # sys.exit("✅ 已完成流式输出，脚本退出")
            final_anwser = thought + answer
            print("===== RAW_BEFORE_DECORATE =====", flush=True)
            print(final_anwser, flush=True)
            res = decorate_answer(final_anwser)
            print("===== POST_AFTER_DECORATE =====", flush=True)
            print(res, flush=True)
            # print("Final_Answer: ", thought + answer)
            print("\n\n\n")
            print("mzy0.5 = ")
            yield res  # decorate_answer是插入引用
            print("test3")

        else:
            print("不属于电磁领域知识")
            ### 此处设计就不考虑历史了，替换msg[1:]即可
            # print("#### prompt + prompt4citation为",prompt + prompt4citation)
            prompt_out_MAG = ("你是一个智能助手，请依照公共知识回答用户问题")
            # current_user_question
            current_user_question2 = [{"role": "user", "content": msg[-1]["content"]}]
            for ans in chat_mdl.chat_streamly(prompt_out_MAG, current_user_question2, gen_conf):
                # for ans in chat_mdl.chat_streamly(prompt_out_MAG, msg[1:], gen_conf):
                if thought:
                    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
                answer = ans
                delta_ans = ans[len(last_ans):]
                if num_tokens_from_string(delta_ans) < 16:
                    continue
                last_ans = answer
                yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
            delta_ans = answer[len(last_ans):]
            if delta_ans:
                yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}

    else:
        answer = chat_mdl.chat(prompt + prompt4citation, msg[1:], gen_conf)
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        res = decorate_answer(answer)
        res["audio_binary"] = tts(tts_mdl, answer)
        yield res


def use_sql(question, field_map, tenant_id, chat_mdl, quota=True):
    sys_prompt = "You are a Database Administrator. You need to check the fields of the following tables based on the user's list of questions and write the SQL corresponding to the last question."
    user_prompt = """
Table name: {};
Table of database fields are as follows:
{}

Question are as follows:
{}
Please write the SQL, only SQL, without any other explanations or text.
""".format(index_name(tenant_id), "\n".join([f"{k}: {v}" for k, v in field_map.items()]), question)
    tried_times = 0

    def get_table():
        nonlocal sys_prompt, user_prompt, question, tried_times
        sql = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {"temperature": 0.06})
        sql = re.sub(r"^.*</think>", "", sql, flags=re.DOTALL)
        logging.debug(f"{question} ==> {user_prompt} get SQL: {sql}")
        sql = re.sub(r"[\r\n]+", " ", sql.lower())
        sql = re.sub(r".*select ", "select ", sql.lower())
        sql = re.sub(r" +", " ", sql)
        sql = re.sub(r"([;；]|```).*", "", sql)
        if sql[: len("select ")] != "select ":
            return None, None
        if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
            if sql[: len("select *")] != "select *":
                sql = "select doc_id,docnm_kwd," + sql[6:]
            else:
                flds = []
                for k in field_map.keys():
                    if k in forbidden_select_fields4resume:
                        continue
                    if len(flds) > 11:
                        break
                    flds.append(k)
                sql = "select doc_id,docnm_kwd," + ",".join(flds) + sql[8:]

        logging.debug(f"{question} get SQL(refined): {sql}")
        tried_times += 1
        return settings.retrievaler.sql_retrieval(sql, format="json"), sql

    tbl, sql = get_table()
    if tbl is None:
        return None
    if tbl.get("error") and tried_times <= 2:
        user_prompt = """
        Table name: {};
        Table of database fields are as follows:
        {}

        Question are as follows:
        {}
        Please write the SQL, only SQL, without any other explanations or text.


        The SQL error you provided last time is as follows:
        {}

        Error issued by database as follows:
        {}

        Please correct the error and write SQL again, only SQL, without any other explanations or text.
        """.format(index_name(tenant_id), "\n".join([f"{k}: {v}" for k, v in field_map.items()]), question, sql,
                   tbl["error"])
        tbl, sql = get_table()
        logging.debug("TRY it again: {}".format(sql))

    logging.debug("GET table: {}".format(tbl))
    if tbl.get("error") or len(tbl["rows"]) == 0:
        return None

    docid_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "doc_id"])
    doc_name_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "docnm_kwd"])
    column_idx = [ii for ii in range(len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

    # compose Markdown table
    columns = (
            "|" + "|".join(
        [re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"], tbl["columns"][i]["name"])) for i in
         column_idx]) + ("|Source|" if docid_idx and docid_idx else "|")
    )

    line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + ("|------|" if docid_idx and docid_idx else "")

    rows = ["|" + "|".join([rmSpace(str(r[i])) for i in column_idx]).replace("None", " ") + "|" for r in tbl["rows"]]
    rows = [r for r in rows if re.sub(r"[ |]+", "", r)]
    if quota:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    else:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    if not docid_idx or not doc_name_idx:
        logging.warning("SQL missing field: " + sql)
        return {"answer": "\n".join([columns, line, rows]), "reference": {"chunks": [], "doc_aggs": []},
                "prompt": sys_prompt}

    docid_idx = list(docid_idx)[0]
    doc_name_idx = list(doc_name_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1
    return {
        "answer": "\n".join([columns, line, rows]),
        "reference": {
            "chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
            "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in
                         doc_aggs.items()],
        },
        "prompt": sys_prompt,
    }


def tts(tts_mdl, text):
    if not tts_mdl or not text:
        return
    bin = b""
    for chunk in tts_mdl.tts(text):
        bin += chunk
    return binascii.hexlify(bin).decode("utf-8")


# def ask(question, kb_ids, tenant_id):
#     from api.db.db_models import Dialog
#
#     # 保留原有的检索逻辑
#     kbs = KnowledgebaseService.get_by_ids(kb_ids)
#     embedding_list = list(set([kb.embd_id for kb in kbs]))
#
#     is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
#     retriever = settings.retrievaler if not is_knowledge_graph else settings.kg_retrievaler
#
#     embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embedding_list[0])
#     chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
#     max_tokens = chat_mdl.max_length
#     tenant_ids = list(set([kb.tenant_id for kb in kbs]))
#     kbinfos = retriever.retrieval(question, embd_mdl, tenant_ids, kb_ids, 1, 12, 0.1, 0.3, aggs=False,
#                                   rank_feature=label_question(question, kbs))
#
#     # 创建临时dialog对象以使用chat函数
#     temp_dialog = Dialog()
#     temp_dialog.tenant_id = tenant_id
#     temp_dialog.llm_id = chat_mdl.llm_name
#     temp_dialog.kb_ids = kb_ids
#     temp_dialog.prompt_config = {
#         "system": """Role: You're a smart assistant. Your name is Miss R.
# Task: Summarize the information from knowledge bases and answer user's question.
# Requirements and restriction:
#   - DO NOT make things up, especially for numbers.
#   - If the information from knowledge is irrelevant with user's question, JUST SAY: Sorry, no relevant information provided.
#   - Answer with markdown format text.
#   - Answer in language of user's question.
#   - DO NOT make things up, especially for numbers.""",
#         "parameters": []
#     }
#     temp_dialog.llm_setting = {"temperature": 0.1}
#     temp_dialog.similarity_threshold = 0.1
#     temp_dialog.vector_similarity_weight = 0.3
#     temp_dialog.top_n = 12
#     temp_dialog.top_k = 1024
#
#     # 构造messages
#     messages = [{"role": "user", "content": question}]
#
#     final_response = None
#     for ans in chat(temp_dialog, messages, stream=True):
#         final_response = ans  # 只收集最后一个结果
#
#     if final_response:
#         yield final_response
#
def ask(question, kb_ids, tenant_id):
    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))

    is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
    retriever = settings.retrievaler if not is_knowledge_graph else settings.kg_retrievaler

    embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embedding_list[0])
    chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
    max_tokens = chat_mdl.max_length
    tenant_ids = list(set([kb.tenant_id for kb in kbs]))
    kbinfos = retriever.retrieval(question, embd_mdl, tenant_ids, kb_ids, 1, 12, 0.1, 0.3, aggs=False,
                                  rank_feature=label_question(question, kbs))
    knowledges = kb_prompt(kbinfos, max_tokens)
    prompt = """
    Role: You're a smart assistant. Your name is Miss R.
    Task: Summarize the information from knowledge bases and answer user's question.
    Requirements and restriction:
      - DO NOT make things up, especially for numbers.
      - If the information from knowledge is irrelevant with user's question, JUST SAY: Sorry, no relevant information provided.
      - Answer with markdown format text.
      - Answer in language of user's question.
      - DO NOT make things up, especially for numbers.

    ### Information from knowledge bases
    %s

    The above is information from knowledge bases.

    """ % "\n".join(knowledges)
    msg = [{"role": "user", "content": question}]

    def decorate_answer(answer):
        # print("mzy0 = ")
        nonlocal knowledges, kbinfos, prompt
        # print("mzy1 = ")
        answer, idx = retriever.insert_citations(answer, [ck["content_ltks"] for ck in kbinfos["chunks"]],
                                                 [ck["vector"] for ck in kbinfos["chunks"]], embd_mdl, tkweight=0.7,
                                                 vtweight=0.3)
        # print("mzy2 = ")
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        # print("mzy3 = ")
        recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        # print("mzy4 = ")
        if not recall_docs:
            # print("mzy5 = ")
            recall_docs = kbinfos["doc_aggs"]
            # print("mzy6 = ")
        kbinfos["doc_aggs"] = recall_docs
        # print("mzy7 = ")
        refs = deepcopy(kbinfos)
        # print("mzy8 = ")
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]
        # print("mzy9 = ")
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        # print("mzy10 = ")
        refs["chunks"] = chunks_format(refs)
        # print("mzy11 = ")
        return {"answer": answer, "reference": refs}

    answer = ""
    for ans in chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
        answer = ans
        yield {"answer": answer, "reference": {}}
    yield decorate_answer(answer)


def get_default_image_list_text():
    captions = [
        "天线阵列的增益方向图",
        "毫米波信道建模示意图",
        "MIMO系统中的信道矩阵结构",
        "信道估计误差与频率响应对比图",
        "频域下的子载波分布示意",
        "无线传播路径中的反射和绕射图",
        "CSI在不同时间步下的变化趋势",
        "OFDM系统帧结构图",
        "瑞利衰落与莱斯衰落的对比",
        "功率谱密度的典型分布",
        "空时编码结构示意",
        "天线极化与接收信号强度关系图",
        "时延扩展对系统性能的影响"
    ]
    return "\n".join([f"<image{i + 1}/{caption}>" for i, caption in enumerate(captions)])


import os
import random
import json


def generate_image_list_from_kbinfos(kbinfos):
    """
    根据 kbinfos 中 doc_aggs 的 doc_name 读取对应 captions.json 文件，形成图例字符串列表并拼接返回。

    参数：
        kbinfos: dict，包含 "doc_aggs" 字段，列表中每项含有 "doc_name" 字段。

    返回：
        多行字符串，每行格式如 <image_assets/doc_name/figures/pX_Y:题注>
    """
    image_entries = []

    doc_aggs = kbinfos.get("doc_aggs", [])
    for doc in doc_aggs:
        doc_name = doc.get("doc_name")
        if not doc_name:
            continue

        # 去掉 doc_name 里的 .md 后缀（如果存在）
        if doc_name.endswith(".md"):
            doc_name = doc_name[:-3]
        # 生成 captions.json 文件路径
        captions_path = os.path.join("./image_assets", doc_name, "captions.json")
        if not os.path.isfile(captions_path):
            # 如果文件不存在，跳过
            continue

        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                captions = json.load(f)
            # captions 是类似 {"p1_1": "null", "p2_1": "图1 ...", ...} 结构
            for key, caption in captions.items():
                image_path = os.path.join("./image_assets", doc_name, "figures", f"{key}.jpg")
                if not os.path.isfile(image_path):
                    # print("该文件不存在(跳过)： ",image_path)
                    continue  # 图像文件不存在，跳过该项

                # 拼成 <image_assets/doc_name/figures/pX_Y:题注> 形式
                image_entry = f"<image_assets/{doc_name}/figures/{key}:{caption}>"
                image_entries.append(image_entry)
        except Exception as e:
            print(f"读取文件 {captions_path} 出错: {e}")
            continue

    # 随机排序后取前30个
    # if len(image_entries) > 30:
    #     random.shuffle(image_entries)
    #     image_entries = image_entries[:30]

    # 用换行符拼接所有图例条目
    image_list_text = "\n".join(image_entries)
    # print(f"image_list_text :{image_list_text} ")
    return image_list_text


def is_chinese(text):
    # 检测字符串中是否包含中文字符
    return bool(re.search(r'[\u4e00-\u9fff]', text))
def is_sinhala(text):
    #
    return bool(re.search(r'[\u0D80-\u0DFF]', text))
def is_tai(text):
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

def clean_formatting(text):
    # 去除==...==模式的删除线
    # text = re.sub(r'=(.+?)=', r'\1', text)
    # text = re.sub(r'=(.+?)~', r'\1', text)
    # text = re.sub(r'~(.+?)=', r'\1', text)
    # text = re.sub(r'~(.+?)~', r'\1', text)
    # text = re.sub(r'##(.+?)##', r'\1', text)
    # text = re.sub(r'$$(.+?)$$', r'\1', text)
    # text = re.sub(r'##(.+?)$$', r'\1', text)
    # text = re.sub(r'$$(.+?)##', r'\1', text)
    # text = re.sub(r'==(.+?)==', r'\1', text)
    # text = re.sub(r'~~(.+?)~~', r'\1', text) # 删除删除线
    # text = re.sub(r'==(.*?)~~', r'\1', text)
    # text = re.sub(r'~~(.*?)==', r'\1', text)
    # # text = re.sub(r'__([^_]+?)__', r'\1', text) # 删除加粗
    # # text = re.sub(r'_([^_]+?)_', r'\1', text) # 删除斜体
    # text = re.sub(r'<del.*?>.*?</del>', '', text)  # 清除<del>标签内容
    # text = re.sub(r'<strike.*?>.*?</strike>', '', text)  # 清除<strike>标签内容
    # text = re.sub(r'##\d+\$\$\s*', '', text)
    # text = re.sub(r'~~\d+==\s*', '', text)
    return text


def extract_entities_with_llm(query, api_url, api_key=None):
    """
    使用大模型从复杂问题中提取最多3个关键实体（优先聚合型、用于联网搜索的核心实体）
    :param query: 原始查询问题
    :param api_url: 大模型API地址
    :param api_key: API密钥（可选）
    :return: 实体列表
    """
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 优化后的提示词：优先聚合型核心搜索实体，排除通用知识
    prompt = f"""请从以下问题中提取1-3个最适合用于联网搜索的核心实体，需满足：
1. 优先提取完整的事件/概念组合（如"2025诺贝尔物理学奖"而非拆分的"2025""诺贝尔物理学奖"）；
2. 仅提取需要联网获取最新/特定信息的内容，大模型通用知识无需提取；
3. 最多提取3个，保留最核心的联网查询实体，如"Will the Nobel Prize in Physics be related to electromagnetism?"提取"Nobel Prize in Physics"；
4. 无需要联网的实体则返回空列表。

问题：{query}

请严格以JSON格式返回（仅返回JSON，不要其他内容）：
{{"entities": ["实体1", "实体2", "实体3"]}}"""

    payload = {
        "model": "deepseek-chat",  # 根据实际API调整
        "messages": [
            {"role": "system", "content": "你是一个专业的搜索引擎关键词提取助手，擅长从用户问题中提取最适合联网搜索的实体组合。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,  # 降低随机性，提升提取精准度
        "max_tokens": 100    # 限制返回长度，避免冗余
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # 触发HTTP状态码错误

        result = response.json()
        # 适配不同API的响应格式
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"].strip()
        else:
            content = str(result)

        # 解析JSON并清洗实体
        entities_data = json.loads(content)
        entities = entities_data.get("entities", [])
        # 去空、去重、限制最多3个实体
        entities = [e.strip() for e in entities if e.strip()]
        entities = list(dict.fromkeys(entities))[:3]

        # 结果判断
        if entities:
            print(f"🎯 提取的核心搜索实体: {entities}")
            return entities
        else:
            print("⚠️ 未提取到核心搜索实体，使用原查询")
            return [query]

    except json.JSONDecodeError:
        print(f"❌ 实体提取失败：返回内容非有效JSON，内容片段：{content[:100]}...")
        return [query]
    except requests.exceptions.RequestException as e:
        print(f"❌ 实体提取请求失败: {str(e)[:50]}...")
        return [query]
    except Exception as e:
        print(f"❌ 实体提取未知错误: {str(e)[:50]}...")
        return [query]

# ---------------------- 1. 爬虫核心函数（保持不变）----------------------
def crawl_and_clean_page(url, title_keywords, timeout=1, retry=0):
    """
    根据URL爬取网页，并基于标题关键词清洗提取核心内容
    :param url: 目标网页链接
    :param title_keywords: 标题关键词（用于过滤无关内容）
    :param timeout: 请求超时时间
    :param retry: 重试次数
    :return: 清洗后的核心文本 / 失败提示
    """
    # 前置过滤：排除非网页链接
    parsed_url = urlparse(url)
    if not parsed_url.scheme in ('http', 'https'):
        return "❌ 非网页链接，跳过爬取"

    # 请求头：模拟浏览器，避免反爬
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://duckduckgo.com/"
    }

    # 重试机制：应对临时网络/反爬
    for attempt in range(retry + 1):
        try:
            # 发起请求（禁止重定向到非预期域名）
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                verify=False  # 忽略SSL证书错误（部分网站需开启）
            )
            response.raise_for_status()  # 触发4xx/5xx错误

            # 编码处理：避免乱码
            response.encoding = response.apparent_encoding or 'utf-8'

            # 提取核心正文（trafilatura自动过滤广告/导航/脚本）
            # 方法1：用trafilatura直接提取（推荐，准确率高）
            raw_html = response.text
            extracted_text = trafilatura.extract(
                raw_html,
                include_comments=False,  # 排除评论
                include_tables=False,    # 排除表格（按需开启）
                only_with_metadata=True, # 只保留有元数据的正文
                favor_precision=True     # 优先精准度
            )

            # 方法2：备用方案（BeautifulSoup手动提取，防止trafilatura失效）
            if not extracted_text:
                soup = BeautifulSoup(raw_html, 'lxml')
                # 移除无关标签
                for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'ad']):
                    tag.decompose()
                # 提取正文标签内容
                main_content = soup.find(['article', 'main', 'div'], class_=lambda c: c and 'content' in c.lower())
                extracted_text = main_content.get_text(strip=True) if main_content else soup.get_text(strip=True)

            # 基于标题关键词清洗：只保留包含核心关键词的段落
            if extracted_text and title_keywords:
                # 按段落分割，过滤无关内容
                paragraphs = [p.strip() for p in extracted_text.split('\n') if p.strip()]
                filtered_paragraphs = []
                for para in paragraphs:
                    # 段落包含至少1个标题关键词才保留
                    if any(kw.lower() in para.lower() for kw in title_keywords):
                        filtered_paragraphs.append(para)
                # 合并过滤后的段落
                cleaned_text = '\n\n'.join(filtered_paragraphs) if filtered_paragraphs else extracted_text
            else:
                cleaned_text = extracted_text or "⚠️  未提取到有效内容"

            # 长度控制：避免输出过长（保留前2000字符）
            if len(cleaned_text) > 1000:
                cleaned_text = cleaned_text[:1000] + "\n\n【内容过长，已截断】"

            return f"✅：{cleaned_text}"

        except requests.exceptions.RequestException as e:
            if attempt < retry:
                time.sleep(2 ** attempt)  # 指数退避重试
                continue
            return f"❌ 爬取失败：{str(e)[:100]}..."  # 截断错误信息
        except Exception as e:
            return f"❌ 解析失败：{str(e)[:100]}..."

# ---------------------- 2. DDGS搜索函数（原有逻辑） ----------------------
def improved_ddgs_search(query, max_results=10, region='wt-wt', timelimit=None):
    """改进的 DuckDuckGo 搜索，支持更多参数"""
    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            max_results=max_results,
            region=region,
            timelimit=timelimit
        )
        return list(results)

# ---------------------- 3. 封装的网络搜索主函数（新增） ----------------------
def run_web_search(original_query, llm_api_url, llm_api_key, max_results_per_entity=3):
    """
    执行完整的网络搜索流程，返回整合后的搜索结果文本
    :param original_query: 原始用户查询问题
    :param llm_api_url: 大模型API地址（用于实体提取）
    :param llm_api_key: 大模型API密钥
    :param max_results_per_entity: 每个实体的最大搜索结果数
    :return: 整合后的搜索结果文本（可直接嵌入系统提示词）
    """
    search_context_parts = []
    search_context_parts.append(f"📝 原始查询：{original_query}")

    # 步骤1：提取核心实体
    entities = extract_entities_with_llm(original_query, llm_api_url, llm_api_key)

    # 步骤2：遍历每个实体搜索
    for i, entity in enumerate(entities, 1):
        search_context_parts.append(f"\n🎯 搜索实体 {i}/{len(entities)}: {entity}")
        search_context_parts.append("-" * 60)

        try:
            # 执行DDGS搜索
            results = improved_ddgs_search(entity, max_results=max_results_per_entity, region='wt-wt')

            for j, r in enumerate(results, 1):
                title = r['title']
                url = r['href']
                original_summary = r['body']

                # 添加基础信息
                search_context_parts.append(f"\n  {j}. 标题: {title}")
                search_context_parts.append(f"     链接: {url}")
                search_context_parts.append(f"     原始摘要: {original_summary}")

                # 提取标题关键词，爬取并清洗内容
                title_keywords = [kw.strip() for kw in title.split() if len(kw.strip()) > 2]
                time.sleep(1)  # 反爬延迟
                crawled_result = crawl_and_clean_page(url, title_keywords)

                # 添加爬取结果
                search_context_parts.append(f"     核心内容: \n     {crawled_result}")
                search_context_parts.append("\n" + "     " + "-" * 50)

        except Exception as e:
            error_msg = f"❌ 该实体搜索失败: {e}"
            search_context_parts.append(error_msg)
            print(error_msg)

    # 合并所有部分为最终文本
    full_search_context = "\n".join(search_context_parts)
    return full_search_context
