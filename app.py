import time
from datetime import datetime

import streamlit as st

from agent.react_agent import ReactAgent
from rag.vector_store import ensure_knowledge_indexed
from utils.geo_weather import set_location_override

USER_IDS = [f"100{i}" for i in range(1, 11)]

st.title("扫地机器人智能客服")

with st.sidebar:
    st.subheader("会话设置")
    user_id = st.selectbox(
        "模拟用户 ID",
        options=USER_IDS,
        index=0,
        help="用于演示个性化使用报告，对应 data/external/records.csv 中的用户数据",
    )
    st.session_state["user_id"] = user_id

    st.subheader("位置设置")
    manual_city = st.text_input(
        "所在城市",
        placeholder="留空则自动 IP 定位",
        help="IP 定位可能不准确，可在此手动填写，例如：深圳",
    )
    set_location_override(manual_city)

    if st.button("清空对话"):
        st.session_state["message"] = []
        st.rerun()

st.divider()

if "index_ready" not in st.session_state:
    with st.spinner("正在检查并索引知识库..."):
        skipped, indexed = ensure_knowledge_indexed()
    st.session_state["index_ready"] = True
    if indexed > 0:
        st.toast(f"知识库索引完成：新增 {indexed} 个文件")
    elif skipped > 0:
        st.toast("知识库已是最新")

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("tool_traces"):
            with st.expander("思考过程", expanded=False):
                for idx, trace in enumerate(message["tool_traces"], start=1):
                    st.markdown(f"**{idx}. {trace['name']}**")
                    st.code(f"参数: {trace.get('args', {})}", language="json")
                    if trace.get("result"):
                        st.text(f"结果摘要: {trace['result']}")

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
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

    with st.spinner("智能客服思考中..."):
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
                        time.sleep(0.01)
                        yield char

        st.chat_message("assistant").write_stream(capture_text(event_stream))

    full_response = "".join(response_parts)
    st.session_state["message"].append(
        {
            "role": "assistant",
            "content": full_response,
            "tool_traces": tool_traces,
        }
    )
    st.rerun()
