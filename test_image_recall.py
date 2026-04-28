import importlib.util
import json
import shutil
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parent


class _DummyAtomic:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDB:
    @staticmethod
    def atomic():
        return _DummyAtomic()

    @staticmethod
    def connection_context():
        def decorator(func):
            return func

        return decorator


class _FakeDialog:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def save(self, force_insert=False):
        return self


class _FakeEncoder:
    @staticmethod
    def encode(text):
        return list(str(text).encode("utf-8"))

    @staticmethod
    def decode(tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


class _PlannerModel:
    def __init__(self, answer_factory=None, is_tools=True):
        self.is_tools = is_tools
        self.answer_factory = answer_factory
        self.tool_session = None

    def bind_tools(self, toolcall_session, tools):
        self.tool_session = toolcall_session
        self.tools = tools

    def chat(self, system_prompt, messages, llm_setting):
        if self.answer_factory:
            return self.answer_factory(self.tool_session)
        return json.dumps(
            {
                "show_images": False,
                "mode": "none",
                "items": [],
                "reason_code": "default_none",
                "confidence": 0.0,
            },
            ensure_ascii=False,
        )


def _register_module(monkeypatch, name: str, module):
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = types.ModuleType(parent_name)
            parent.__path__ = []
            _register_module(monkeypatch, parent_name, parent)
        setattr(parent, child_name, module)
    monkeypatch.setitem(sys.modules, name, module)


def _load_repo_module(monkeypatch, module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    _register_module(monkeypatch, module_name, module)
    spec.loader.exec_module(module)
    return module


def _install_stubbed_dependencies(monkeypatch):
    json_repair_mod = types.ModuleType("json_repair")
    _register_module(monkeypatch, "json_repair", json_repair_mod)

    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *args, **kwargs: None
    _register_module(monkeypatch, "dotenv", dotenv_mod)

    langfuse_mod = types.ModuleType("langfuse")
    langfuse_mod.Langfuse = object
    _register_module(monkeypatch, "langfuse", langfuse_mod)

    agentic_reasoning_mod = types.ModuleType("agentic_reasoning")
    agentic_reasoning_mod.DeepResearcher = object
    _register_module(monkeypatch, "agentic_reasoning", agentic_reasoning_mod)

    api_pkg = types.ModuleType("api")
    api_pkg.__path__ = []
    api_pkg.settings = SimpleNamespace(FACTORY_LLM_INFOS=[], kg_retrievaler=None)
    _register_module(monkeypatch, "api", api_pkg)

    api_db_pkg = types.ModuleType("api.db")
    api_db_pkg.__path__ = []
    api_db_pkg.LLMType = SimpleNamespace(CHAT="chat", IMAGE2TEXT="image2text", RERANK="rerank", TTS="tts", EMBEDDING="embedding")
    api_db_pkg.ParserType = SimpleNamespace()
    api_db_pkg.StatusEnum = SimpleNamespace(VALID=SimpleNamespace(value=1))
    _register_module(monkeypatch, "api.db", api_db_pkg)

    db_models_mod = types.ModuleType("api.db.db_models")
    db_models_mod.DB = _FakeDB
    db_models_mod.Dialog = _FakeDialog
    _register_module(monkeypatch, "api.db.db_models", db_models_mod)

    common_service_mod = types.ModuleType("api.db.services.common_service")
    common_service_mod.CommonService = object
    _register_module(monkeypatch, "api.db.services.common_service", common_service_mod)

    knowledgebase_service_mod = types.ModuleType("api.db.services.knowledgebase_service")
    knowledgebase_service_mod.KnowledgebaseService = object
    _register_module(monkeypatch, "api.db.services.knowledgebase_service", knowledgebase_service_mod)

    langfuse_service_mod = types.ModuleType("api.db.services.langfuse_service")
    langfuse_service_mod.TenantLangfuseService = object
    _register_module(monkeypatch, "api.db.services.langfuse_service", langfuse_service_mod)

    llm_service_mod = types.ModuleType("api.db.services.llm_service")
    llm_service_mod.LLMBundle = object
    llm_service_mod.TenantLLMService = type(
        "TenantLLMService",
        (),
        {"split_model_name_and_factory": staticmethod(lambda llm_id: (llm_id, None))},
    )
    _register_module(monkeypatch, "api.db.services.llm_service", llm_service_mod)

    api_utils_mod = types.ModuleType("api.utils")
    api_utils_mod.current_timestamp = lambda: 0
    api_utils_mod.datetime_format = lambda dt: str(dt)
    _register_module(monkeypatch, "api.utils", api_utils_mod)

    rag_pkg = types.ModuleType("rag")
    rag_pkg.__path__ = []
    _register_module(monkeypatch, "rag", rag_pkg)

    rag_settings_mod = types.ModuleType("rag.settings")
    rag_settings_mod.TAG_FLD = "tag_kwd"
    _register_module(monkeypatch, "rag.settings", rag_settings_mod)

    rag_utils_mod = types.ModuleType("rag.utils")
    rag_utils_mod.encoder = _FakeEncoder()
    rag_utils_mod.num_tokens_from_string = lambda text: len(str(text or ""))
    rag_utils_mod.rmSpace = lambda text: str(text or "").strip()
    _register_module(monkeypatch, "rag.utils", rag_utils_mod)

    rag_app_resume_mod = types.ModuleType("rag.app.resume")
    rag_app_resume_mod.forbidden_select_fields4resume = []
    _register_module(monkeypatch, "rag.app.resume", rag_app_resume_mod)

    rag_app_tag_mod = types.ModuleType("rag.app.tag")
    rag_app_tag_mod.label_question = lambda *args, **kwargs: {}
    _register_module(monkeypatch, "rag.app.tag", rag_app_tag_mod)

    rag_nlp_search_mod = types.ModuleType("rag.nlp.search")
    rag_nlp_search_mod.index_name = lambda tenant_id: tenant_id
    _register_module(monkeypatch, "rag.nlp.search", rag_nlp_search_mod)

    tavily_mod = types.ModuleType("rag.utils.tavily_conn")
    tavily_mod.Tavily = object
    _register_module(monkeypatch, "rag.utils.tavily_conn", tavily_mod)


@pytest.fixture
def loaded_modules(monkeypatch):
    _install_stubbed_dependencies(monkeypatch)
    _load_repo_module(monkeypatch, "rag.image_asset_utils", "rag/image_asset_utils.py")
    _load_repo_module(monkeypatch, "rag.multimodal_embedding_client", "rag/multimodal_embedding_client.py")
    _load_repo_module(monkeypatch, "rag.image_index", "rag/image_index.py")
    prompt_mod = _load_repo_module(monkeypatch, "rag.chat_image_prompts", "rag/chat_image_prompts.py")
    planner_mod = _load_repo_module(monkeypatch, "rag.chat_image_planner", "rag/chat_image_planner.py")
    return prompt_mod, planner_mod


@pytest.fixture
def managed_output_docs():
    created_dirs = []
    yield created_dirs
    for path in created_dirs:
        shutil.rmtree(path, ignore_errors=True)


def _create_output_doc(created_dirs, doc_name: str, entries, output_root: Path | None = None):
    root = output_root or (REPO_ROOT / "output")
    doc_dir = root / doc_name
    figures_dir = doc_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created_dirs.append(doc_dir)

    captions = {}
    for entry in entries:
        captions[entry["key"]] = entry["caption"]
        if entry.get("create_image", True):
            (figures_dir / f"{entry['key']}.jpg").write_bytes(entry.get("bytes", b"fake-image-data"))

    with open(doc_dir / "captions.json", "w", encoding="utf-8") as fout:
        json.dump(captions, fout, ensure_ascii=False, indent=2)

    return doc_dir


def test_select_image_candidate_doc_names_prefers_explicit_doc_mentions(loaded_modules):
    prompt_mod, _ = loaded_modules
    selected = prompt_mod.select_image_candidate_doc_names(
        {
            "doc_aggs": [
                {"doc_name": "文档A.pdf"},
                {"doc_name": "文档B.pdf"},
                {"doc_name": "文档C.pdf"},
            ]
        },
        "请结合《文档B》中的图2回答问题",
        max_docs=3,
    )
    assert selected == ["文档B"]


def test_select_image_candidate_doc_names_prefers_quoted_title_match(loaded_modules):
    prompt_mod, _ = loaded_modules
    selected = prompt_mod.select_image_candidate_doc_names(
        {
            "doc_aggs": [
                {"doc_name": "多模态检索系统设计.pdf"},
                {"doc_name": "视觉问答评测方法.pdf"},
                {"doc_name": "图文融合索引技术.pdf"},
            ]
        },
        "在《视觉问答评测方法》这篇文档中，图2主要展示了什么？",
        max_docs=3,
    )
    assert selected == ["视觉问答评测方法"]


def test_strip_legacy_answer_images_removes_markup_and_intro(loaded_modules):
    _, planner_mod = loaded_modules
    answer = """
如下图所示：

![图片](http://127.0.0.1:8000/image_assets/doc/figures/p1_1.jpg)

<div style="display:flex"><img src="http://127.0.0.1:8000/image_assets/doc/figures/p2_1.jpg" alt="图片" width="300"><figcaption>Figure 2</figcaption></div>

这是正文结论。
""".strip()
    stripped = planner_mod.strip_legacy_answer_images(answer)
    assert "如下图所示" not in stripped
    assert "http://127.0.0.1:8000/image_assets/doc/figures/p1_1.jpg" not in stripped
    assert "http://127.0.0.1:8000/image_assets/doc/figures/p2_1.jpg" not in stripped
    assert stripped == "这是正文结论。"


def test_evaluate_image_gate_accepts_specific_figure_match(loaded_modules):
    _, planner_mod = loaded_modules
    gate = planner_mod.evaluate_image_gate(
        "在文档中，图2主要展示了什么？",
        {
            "chunks": [{"content_with_weight": "图2展示了数据驱动的SEI流程。"}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p2_1", "caption": "图 2 基于数据驱动的 SEI 示意", "score": 0.72},
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "图 1 基于 DL 的 SEI 实现过程示意", "score": 0.69},
            ],
        },
    )
    assert gate["passed"] is True
    assert gate["mode_hint"] == "single"
    assert gate["reason_code"] == "specific_figure_match"
    assert [item["figure_key"] for item in gate["filtered_candidates"]] == ["p2_1"]


def test_evaluate_image_gate_rejects_broad_visual_when_candidates_are_weak(loaded_modules):
    _, planner_mod = loaded_modules
    gate = planner_mod.evaluate_image_gate(
        "请结合图片解释这篇文章的课程设计。",
        {
            "chunks": [{"content_with_weight": "文章介绍了课程由讲授、实验和CTF竞赛组成。"}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p4_1", "caption": "Figure 1: CTF competition.", "score": 0.46},
                {"doc_name": "docA", "figure_key": "p4_2", "caption": "Figure 2: Student enthusiasm.", "score": 0.40},
                {"doc_name": "docA", "figure_key": "p5_1", "caption": "Frame in an 802.11 packet.", "score": 0.38},
            ],
        },
    )
    assert gate["passed"] is False
    assert gate["mode_hint"] == "none"
    assert gate["reason_code"] == "broad_visual_not_confident"


def test_evaluate_image_gate_accepts_compare_when_two_candidates_are_strong(loaded_modules):
    _, planner_mod = loaded_modules
    gate = planner_mod.evaluate_image_gate(
        "请比较 Figure 1 和 Figure 2，并说明它们分别反映了什么。",
        {
            "chunks": [{"content_with_weight": "Figure 1 and Figure 2 are both discussed in the main body."}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "Figure 1 pipeline", "score": 0.66},
                {"doc_name": "docA", "figure_key": "p2_1", "caption": "Figure 2 results", "score": 0.61},
                {"doc_name": "docA", "figure_key": "p3_1", "caption": "Figure 3 appendix", "score": 0.28},
            ],
        },
    )
    assert gate["passed"] is True
    assert gate["mode_hint"] == "compare"
    assert [item["figure_key"] for item in gate["filtered_candidates"]] == ["p1_1", "p2_1"]


def test_evaluate_image_gate_penalizes_ui_screenshot_for_text_only_query(loaded_modules):
    _, planner_mod = loaded_modules
    gate = planner_mod.evaluate_image_gate(
        "这篇文档的研究背景是什么？",
        {
            "chunks": [{"content_with_weight": "文章主要讨论课程背景和设计目标。"}],
            "image_candidates": [
                {
                    "doc_name": "docA",
                    "figure_key": "p1_1",
                    "caption": 'A button labeled "Check for updates" with a bookmark icon',
                    "score": 0.74,
                },
                {
                    "doc_name": "docA",
                    "figure_key": "p4_1",
                    "caption": "Figure 1: Course competition overview.",
                    "score": 0.58,
                },
            ],
        },
    )
    assert gate["passed"] is False
    assert gate["reason_code"] == "text_only_not_visual"
    assert gate["gate_score_breakdown"]["best_visual_type"] == "normal_figure"


def test_plan_image_rendering_builds_single_mode_plan(loaded_modules):
    _, planner_mod = loaded_modules

    def answer_factory(tool_session):
        tool_session.tool_call(
            "search_image_candidates",
            {"question": "Figure 1 展示了什么？", "doc_scope": ["docA"], "top_k": 3},
        )
        return json.dumps(
            {
                "show_images": True,
                "mode": "single",
                "items": [{"doc_name": "docA", "figure_key": "p1_1"}],
                "reason_code": "explicit_figure_match",
                "confidence": 0.91,
            },
            ensure_ascii=False,
        )

    planner = _PlannerModel(answer_factory=answer_factory, is_tools=True)
    plan = planner_mod.plan_image_rendering(
        planner,
        {"temperature": 0.2},
        "Figure 1 展示了什么？",
        {
            "chunks": [{"content_with_weight": "Figure 1 introduces the pipeline."}],
            "image_candidates": [
                {
                    "doc_name": "docA",
                    "figure_key": "p1_1",
                    "caption": "Figure 1 pipeline",
                    "score": 0.93,
                },
                {
                    "doc_name": "docA",
                    "figure_key": "p2_1",
                    "caption": "Figure 2 ablation",
                    "score": 0.41,
                },
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
    )
    assert plan["show_images"] is True
    assert plan["mode"] == "single"
    assert plan["presentation"] == "focus"
    assert [item["figure_key"] for item in plan["items"]] == ["p1_1"]
    assert plan["items"][0]["image_url"].endswith("/image_assets/docA/figures/p1_1.jpg")


def test_plan_image_rendering_builds_compare_mode_plan(loaded_modules):
    _, planner_mod = loaded_modules

    def answer_factory(tool_session):
        tool_session.tool_call(
            "search_image_candidates",
            {"question": "比较 Figure 1 和 Figure 2", "doc_scope": ["docA"], "top_k": 3},
        )
        return json.dumps(
            {
                "show_images": True,
                "mode": "compare",
                "items": [
                    {"doc_name": "docA", "figure_key": "p1_1"},
                    {"doc_name": "docA", "figure_key": "p2_1"},
                ],
                "reason_code": "compare_two_figures",
                "confidence": 0.88,
            },
            ensure_ascii=False,
        )

    planner = _PlannerModel(answer_factory=answer_factory, is_tools=True)
    plan = planner_mod.plan_image_rendering(
        planner,
        {},
        "比较 Figure 1 和 Figure 2",
        {
            "chunks": [{"content_with_weight": "Figure 1 and Figure 2 are both discussed."}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "Figure 1 pipeline", "score": 0.93},
                {"doc_name": "docA", "figure_key": "p2_1", "caption": "Figure 2 results", "score": 0.91},
                {"doc_name": "docA", "figure_key": "p3_1", "caption": "Figure 3 appendix", "score": 0.12},
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
    )
    assert plan["show_images"] is True
    assert plan["mode"] == "compare"
    assert plan["presentation"] == "side_by_side"
    assert [item["figure_key"] for item in plan["items"]] == ["p1_1", "p2_1"]


def test_plan_image_rendering_rejects_invalid_or_low_confidence_plans(loaded_modules):
    _, planner_mod = loaded_modules

    def answer_factory(tool_session):
        tool_session.tool_call(
            "search_image_candidates",
            {"question": "请结合图片解释", "doc_scope": ["docA"], "top_k": 3},
        )
        return json.dumps(
            {
                "show_images": True,
                "mode": "gallery",
                "items": [{"doc_name": "docA", "figure_key": "p9_9"}],
                "reason_code": "invalid_selection",
                "confidence": 0.12,
            },
            ensure_ascii=False,
        )

    planner = _PlannerModel(answer_factory=answer_factory, is_tools=True)
    plan = planner_mod.plan_image_rendering(
        planner,
        {},
        "请结合图片解释",
        {
            "chunks": [{"content_with_weight": "The pipeline is explained in the main body."}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "Figure 1 pipeline", "score": 0.93},
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
    )
    assert plan["show_images"] is False
    assert plan["mode"] == "none"


def test_plan_image_rendering_skips_when_tools_are_unavailable(loaded_modules):
    _, planner_mod = loaded_modules
    planner = _PlannerModel(is_tools=False)
    plan = planner_mod.plan_image_rendering(
        planner,
        {},
        "Figure 1 展示了什么？",
        {
            "chunks": [{"content_with_weight": "Figure 1 introduces the pipeline."}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "Figure 1 pipeline", "score": 0.93},
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
    )
    assert plan["show_images"] is False
    assert plan["reason_code"] == "planner_tools_unavailable"


def test_plan_answer_and_render_prefers_agent_answer_text(loaded_modules):
    _, planner_mod = loaded_modules

    def answer_factory(tool_session):
        tool_session.tool_call(
            "search_image_candidates",
            {"question": "Figure 1 展示了什么？", "doc_scope": ["docA"], "top_k": 3},
        )
        return json.dumps(
            {
                "answer_text": "Figure 1 展示了课程中的一个贯穿整个学期的远程 CTF 竞赛。",
                "show_images": True,
                "mode": "single",
                "items": [{"doc_name": "docA", "figure_key": "p1_1"}],
                "reason_code": "explicit_figure_match",
                "confidence": 0.93,
            },
            ensure_ascii=False,
        )

    planner = _PlannerModel(answer_factory=answer_factory, is_tools=True)
    result = planner_mod.plan_answer_and_render(
        planner,
        {},
        "Figure 1 展示了什么？",
        {
            "chunks": [{"content_with_weight": "Figure 1 depicts the semester-long CTF competition."}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "Figure 1 pipeline", "score": 0.93},
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
    )
    assert "远程 CTF 竞赛" in result["answer_text"]
    assert result["render_plan"]["show_images"] is True
    assert result["render_plan"]["items"][0]["figure_key"] == "p1_1"


def test_plan_answer_and_render_passes_preferred_language(loaded_modules):
    _, planner_mod = loaded_modules

    captured = {}

    def answer_factory(tool_session):
        captured["query"] = tool_session.query
        return json.dumps(
            {
                "answer_text": "这是中文回答。",
                "show_images": True,
                "mode": "single",
                "items": [{"doc_name": "docA", "figure_key": "p1_1"}],
                "reason_code": "explicit_figure_match",
                "confidence": 0.93,
            },
            ensure_ascii=False,
        )

    planner = _PlannerModel(answer_factory=answer_factory, is_tools=True)
    result = planner_mod.plan_answer_and_render(
        planner,
        {},
        "Figure 1 shows what?",
        {
            "chunks": [{"content_with_weight": "Figure 1 depicts the semester-long CTF competition."}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "Figure 1 pipeline", "score": 0.93},
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
        preferred_language="Chinese",
    )
    assert result["answer_text"] == "这是中文回答。"


def test_build_candidate_summary_exposes_score_rank_gap_and_type(loaded_modules):
    _, planner_mod = loaded_modules
    summary = planner_mod._build_candidate_summary(
        [
            {
                "doc_name": "docA",
                "figure_key": "p1_1",
                "caption": 'A button labeled "Check for updates" with a bookmark icon',
                "score": 0.91,
                "rank": 1,
            },
            {
                "doc_name": "docA",
                "figure_key": "p4_1",
                "caption": "Figure 1: Our course included a remotely available CTF competition.",
                "score": 0.66,
                "rank": 2,
            },
        ]
    )
    assert '"rank": 1' in summary
    assert '"score": 0.91' in summary
    assert '"gap_to_next": 0.25' in summary
    assert '"visual_type": "ui_screenshot"' in summary
    assert '"visual_type": "figure_or_diagram"' in summary


def test_plan_answer_and_render_rewrites_not_found_answer_when_image_selected(loaded_modules):
    _, planner_mod = loaded_modules

    def answer_factory(tool_session):
        tool_session.tool_call(
            "search_image_candidates",
            {"question": "图2主要展示了什么？", "doc_scope": ["docA"], "top_k": 3},
        )
        return json.dumps(
            {
                "answer_text": "知识库中未找到您要的答案！",
                "show_images": True,
                "mode": "single",
                "items": [{"doc_name": "docA", "figure_key": "p2_1"}],
                "reason_code": "explicit_figure_match",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        )

    planner = _PlannerModel(answer_factory=answer_factory, is_tools=True)
    result = planner_mod.plan_answer_and_render(
        planner,
        {},
        "图2主要展示了什么？",
        {
            "chunks": [{"content_with_weight": "图2展示了数据驱动的SEI流程。"}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p2_1", "caption": "图 2 基于数据驱动的 SEI 示意", "score": 0.93},
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
    )
    assert "知识库中未找到您要的答案" not in result["answer_text"]
    assert "基于数据驱动的 SEI 示意" in result["answer_text"]


def test_plan_answer_and_render_removes_cannot_display_disclaimer(loaded_modules):
    _, planner_mod = loaded_modules

    def answer_factory(tool_session):
        tool_session.tool_call(
            "search_image_candidates",
            {"question": "Figure 1 展示了什么？", "doc_scope": ["docA"], "top_k": 3},
        )
        return json.dumps(
            {
                "answer_text": "知识库中未提供《docA》中 Figure 1 的图像文件，因此无法直接展示图片。\n\nFigure 1 展示了一个贯穿整个学期的远程 CTF 竞赛。",
                "show_images": True,
                "mode": "single",
                "items": [{"doc_name": "docA", "figure_key": "p1_1"}],
                "reason_code": "explicit_figure_match",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        )

    planner = _PlannerModel(answer_factory=answer_factory, is_tools=True)
    result = planner_mod.plan_answer_and_render(
        planner,
        {},
        "Figure 1 展示了什么？",
        {
            "chunks": [{"content_with_weight": "Figure 1 depicts the semester-long CTF competition."}],
            "image_candidates": [
                {"doc_name": "docA", "figure_key": "p1_1", "caption": "Figure 1 pipeline", "score": 0.93},
            ],
        },
        ["docA"],
        "http://127.0.0.1:8000",
    )
    assert "无法直接展示图片" not in result["answer_text"]
    assert "远程 CTF 竞赛" in result["answer_text"]
