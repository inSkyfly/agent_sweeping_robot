import time
from datetime import datetime

import streamlit as st

from agent.react_agent import ReactAgent
from rag.vector_store import ensure_knowledge_indexed
from utils.geo_weather import set_location_override

USER_IDS = [f"100{i}" for i in range(1, 11)]

QUICK_PROMPTS = [
    "小户型适合哪款扫地机器人？",
    "机器人吸力下降怎么办？",
    "我在成都，今天适合扫地吗？",
    "帮我生成使用报告",
]

TOOL_LABELS = {
    "rag_summarize": "知识库检索",
    "get_weather": "天气查询",
    "get_user_location": "位置定位",
    "get_user_id": "获取用户ID",
    "get_current_month": "获取当前月份",
    "fetch_external_data": "读取使用记录",
    "fill_context_for_report": "切换报告模式",
    "generate_usage_report": "报告子图生成",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif;
        }

        /* 仅隐藏菜单和页脚，保留顶栏侧边栏开关 */
        #MainMenu, footer { visibility: hidden; height: 0; }
        header[data-testid="stHeader"] {
            visibility: visible !important;
            background: rgba(241, 245, 249, 0.92);
            backdrop-filter: blur(8px);
        }
        button[data-testid="baseButton-header"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            display: flex !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            top: 0.85rem !important;
            left: 0.85rem !important;
            background: #0f172a !important;
            color: #f8fafc !important;
            border-radius: 10px !important;
            padding: 0.35rem 0.55rem !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25) !important;
        }
        [data-testid="stSidebarCollapsedControl"] svg {
            stroke: #f8fafc !important;
        }
        .block-container {
            padding-top: 2.8rem;
            padding-bottom: 2rem;
            max-width: 920px;
        }

        /* 侧边栏 */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] p strong {
            color: #cbd5e1 !important;
            font-size: 0.88rem !important;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stTextInput label {
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: #94a3b8 !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 10px !important;
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: rgba(239,68,68,0.15) !important;
            color: #fca5a5 !important;
            border: 1px solid rgba(239,68,68,0.35) !important;
            border-radius: 10px !important;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(239,68,68,0.28) !important;
            border-color: rgba(239,68,68,0.5) !important;
        }

        /* 品牌头部 */
        .brand-hero {
            background: linear-gradient(135deg, #0e7490 0%, #0891b2 45%, #06b6d4 100%);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 40px rgba(8,145,178,0.22);
            position: relative;
            overflow: hidden;
        }
        .brand-hero::after {
            content: '';
            position: absolute;
            right: -30px;
            top: -30px;
            width: 160px;
            height: 160px;
            background: rgba(255,255,255,0.12);
            border-radius: 50%;
        }
        .brand-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #fff;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .brand-sub {
            font-size: 0.92rem;
            color: rgba(255,255,255,0.88);
            margin: 0;
            line-height: 1.5;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }
        .badge {
            background: rgba(255,255,255,0.18);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255,255,255,0.25);
            color: #fff;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
        }

        /* 空状态 */
        .empty-state {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem;
            color: #64748b;
        }
        .empty-icon {
            font-size: 2.8rem;
            margin-bottom: 0.6rem;
        }
        .empty-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 0.35rem;
        }
        .empty-desc {
            font-size: 0.88rem;
            color: #94a3b8;
            margin-bottom: 0;
        }

        /* 聊天气泡增强 */
        div[data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            padding: 0.35rem 0 !important;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background: #fff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 0.85rem 1rem !important;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 0.85rem 1rem !important;
            box-shadow: 0 4px 16px rgba(8,145,178,0.06);
        }

        /* 工具链路卡片 */
        .tool-trace-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #0891b2;
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.5rem;
        }
        .tool-trace-name {
            font-weight: 600;
            color: #0e7490;
            font-size: 0.88rem;
        }
        .tool-trace-result {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.25rem;
            line-height: 1.45;
        }

        /* 快捷提问按钮 */
        div[data-testid="stHorizontalBlock"] .stButton > button {
            background: #fff !important;
            color: #0e7490 !important;
            border: 1px solid #bae6fd !important;
            border-radius: 999px !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            padding: 0.35rem 0.9rem !important;
            transition: all 0.2s ease;
            white-space: nowrap;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {
            background: #ecfeff !important;
            border-color: #0891b2 !important;
            color: #0e7490 !important;
            box-shadow: 0 4px 12px rgba(8,145,178,0.12);
        }

        /* 输入框 */
        div[data-testid="stChatInput"] {
            border-top: 1px solid #e2e8f0;
            padding-top: 0.8rem;
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
            border: 1px solid #cbd5e1 !important;
            background: #fff !important;
        }
        div[data-testid="stChatInput"] textarea:focus {
            border-color: #0891b2 !important;
            box-shadow: 0 0 0 3px rgba(8,145,178,0.15) !important;
        }

        /* Expander */
        div[data-testid="stExpander"] {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.2rem 0 1rem;">
                <div style="font-size:1.35rem;font-weight:700;color:#f8fafc;">🤖 智扫通</div>
                <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.2rem;">智能客服控制台</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**会话设置**")
        user_id = st.selectbox(
            "模拟用户 ID",
            options=USER_IDS,
            index=0,
            help="用于演示个性化使用报告",
        )
        st.session_state["user_id"] = user_id

        st.markdown("**位置设置**")
        manual_city = st.text_input(
            "所在城市",
            placeholder="留空则自动 IP 定位",
            help="IP 定位可能不准确，可手动填写城市",
        )
        set_location_override(manual_city)

        st.divider()
        st.markdown(
            """
            <div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">
            <b style="color:#cbd5e1;">能力概览</b><br>
            · 知识库问答<br>
            · 天气 & 定位<br>
            · 使用报告生成<br>
            · ReAct 工具链路
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        if st.button("清空对话", use_container_width=True):
            st.session_state["message"] = []
            st.session_state.pop("pending_prompt", None)
            st.rerun()

    return manual_city


def render_header() -> None:
    st.markdown(
        """
        <div class="brand-hero">
            <p class="brand-title">智扫通 · 扫地机器人智能客服</p>
            <p class="brand-sub">基于 RAG + ReAct 的垂直领域 Agent，支持知识问答、环境建议与个性化报告</p>
            <div class="badge-row">
                <span class="badge">RAG 混合检索</span>
                <span class="badge">多轮记忆</span>
                <span class="badge">工具编排</span>
                <span class="badge">报告子图</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-title">开始对话</div>
            <p class="empty-desc">试试下方快捷提问，或直接输入你的问题</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_traces(tool_traces: list[dict]) -> None:
    if not tool_traces:
        return
    with st.expander("🔍 思考过程 · 工具调用链路", expanded=False):
        for idx, trace in enumerate(tool_traces, start=1):
            name = trace.get("name", "unknown")
            label = TOOL_LABELS.get(name, name)
            args = trace.get("args", {})
            result = trace.get("result", "")
            st.markdown(
                f"""
                <div class="tool-trace-card">
                    <div class="tool-trace-name">{idx}. {label}</div>
                    <div class="tool-trace-result">参数: {args}</div>
                    {f'<div class="tool-trace-result">结果: {result}</div>' if result else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_history() -> None:
    for message in st.session_state["message"]:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_tool_traces(message.get("tool_traces", []))


def handle_user_prompt(prompt: str, manual_city: str) -> None:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    agent_context = {
        "user_id": st.session_state.get("user_id", "1001"),
        "current_month": datetime.now().strftime("%Y-%m"),
        "city": manual_city.strip(),
        "report": False,
    }

    response_parts: list[str] = []
    tool_traces: list[dict] = []
    pending_calls: dict[str, dict] = {}

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("正在思考..."):
            event_stream = st.session_state["agent"].execute_stream(
                st.session_state["message"],
                context=agent_context,
            )

            def capture_text(generator):
                for event in generator:
                    if event["type"] == "tool_call":
                        pending_calls[event["name"]] = {
                            "name": event["name"],
                            "args": event.get("args", {}),
                        }
                    elif event["type"] == "tool_result":
                        trace = pending_calls.pop(
                            event["name"],
                            {"name": event["name"], "args": {}},
                        )
                        trace["result"] = event.get("result", "")
                        tool_traces.append(trace)
                    elif event["type"] == "text":
                        response_parts.append(event["content"])
                        for char in event["content"]:
                            time.sleep(0.004)
                            yield char

            st.write_stream(capture_text(event_stream))

        if tool_traces:
            render_tool_traces(tool_traces)

    st.session_state["message"].append(
        {
            "role": "assistant",
            "content": "".join(response_parts),
            "tool_traces": tool_traces,
        }
    )
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="智扫通 · 智能客服",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    manual_city = render_sidebar()

    render_header()

    if "index_ready" not in st.session_state:
        with st.spinner("正在检查知识库..."):
            skipped, indexed = ensure_knowledge_indexed()
        st.session_state["index_ready"] = True
        if indexed > 0:
            st.toast(f"知识库索引完成：新增 {indexed} 个文件", icon="✅")
        elif skipped > 0:
            st.toast("知识库已是最新", icon="📚")

    if "agent" not in st.session_state:
        st.session_state["agent"] = ReactAgent()

    if "message" not in st.session_state:
        st.session_state["message"] = []

    if not st.session_state["message"]:
        render_empty_state()

        st.markdown("<p style='font-size:0.82rem;color:#64748b;margin:0.2rem 0 0.5rem;'>快捷提问</p>", unsafe_allow_html=True)
        cols = st.columns(len(QUICK_PROMPTS))
        for col, q in zip(cols, QUICK_PROMPTS):
            with col:
                if st.button(q, key=f"quick_{q}", use_container_width=True):
                    st.session_state["pending_prompt"] = q
                    st.rerun()
    else:
        render_chat_history()

    prompt = st.session_state.pop("pending_prompt", None) or st.chat_input("输入你的问题，例如：小户型怎么选？")
    if prompt:
        handle_user_prompt(prompt, manual_city)


if __name__ == "__main__":
    main()
