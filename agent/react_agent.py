from datetime import datetime

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
    generate_usage_report,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

MAX_HISTORY_ROUNDS = 10


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
                generate_usage_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    @staticmethod
    def _trim_messages(messages: list[dict]) -> list[dict]:
        max_messages = MAX_HISTORY_ROUNDS * 2
        if len(messages) <= max_messages:
            return messages
        return messages[-max_messages:]

    @staticmethod
    def _build_context(context: dict | None = None) -> dict:
        base = {
            "report": False,
            "user_id": "1001",
            "current_month": datetime.now().strftime("%Y-%m"),
            "city": "",
        }
        if context:
            base.update(context)
        return base

    def execute_stream(self, messages: list[dict], context: dict | None = None):
        """流式执行 Agent，产出结构化事件供 UI 消费。"""
        trimmed = self._trim_messages(messages)
        ctx = self._build_context(context)

        seen_tool_calls: set[str] = set()
        seen_tool_results: set[str] = set()
        last_text = ""

        for chunk in self.agent.stream(
            {"messages": trimmed},
            stream_mode="values",
            context=ctx,
        ):
            for msg in chunk["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_id = tc.get("id") or f"{tc['name']}:{tc.get('args', {})}"
                        if tc_id not in seen_tool_calls:
                            seen_tool_calls.add(tc_id)
                            yield {
                                "type": "tool_call",
                                "name": tc["name"],
                                "args": tc.get("args", {}),
                            }

                if isinstance(msg, ToolMessage):
                    result_key = msg.tool_call_id or f"{msg.name}:{msg.content[:50]}"
                    if result_key not in seen_tool_results:
                        seen_tool_results.add(result_key)
                        yield {
                            "type": "tool_result",
                            "name": msg.name or "unknown",
                            "result": (msg.content or "")[:200],
                        }

            latest_message = chunk["messages"][-1]
            if (
                isinstance(latest_message, AIMessage)
                and latest_message.content
                and not latest_message.tool_calls
            ):
                text = latest_message.content.strip()
                if text and text != last_text:
                    new_part = text[len(last_text):] if text.startswith(last_text) else text
                    last_text = text
                    if new_part:
                        yield {"type": "text", "content": new_part}


if __name__ == "__main__":
    agent = ReactAgent()
    demo_messages = [{"role": "user", "content": "给我生成我的使用报告"}]

    for event in agent.execute_stream(demo_messages):
        if event["type"] == "text":
            print(event["content"], end="", flush=True)
