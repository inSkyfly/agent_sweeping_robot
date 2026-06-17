import time

import streamlit as st
from utils.geo_weather import set_location_override
from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")

with st.sidebar:
    st.subheader("位置设置")
    manual_city = st.text_input(
        "所在城市",
        placeholder="留空则自动 IP 定位",
        help="IP 定位可能不准确（如宽带归属地与实际所在地不同），可在此手动填写，例如：深圳",
    )
    set_location_override(manual_city)

st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
