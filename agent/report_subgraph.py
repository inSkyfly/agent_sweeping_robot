from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from model.factory import chat_model
from utils.external_data import fetch_user_month_data
from utils.prompt_loader import load_report_prompts


class ReportState(TypedDict):
    user_id: str
    month: str
    raw_data: str
    report: str


def fetch_data_node(state: ReportState) -> dict:
    return {"raw_data": fetch_user_month_data(state["user_id"], state["month"])}


def generate_report_node(state: ReportState) -> dict:
    if not state["raw_data"]:
        return {"report": f"未找到用户 {state['user_id']} 在 {state['month']} 的使用记录。"}

    prompt = (
        f"{load_report_prompts()}\n\n"
        f"用户ID：{state['user_id']}\n"
        f"报告月份：{state['month']}\n"
        f"使用记录：{state['raw_data']}\n"
        "请根据以上数据生成报告。"
    )
    response = chat_model.invoke(prompt)
    return {"report": response.content}


_report_graph_builder = StateGraph(ReportState)
_report_graph_builder.add_node("fetch_data", fetch_data_node)
_report_graph_builder.add_node("generate_report", generate_report_node)
_report_graph_builder.add_edge(START, "fetch_data")
_report_graph_builder.add_edge("fetch_data", "generate_report")
_report_graph_builder.add_edge("generate_report", END)
report_graph = _report_graph_builder.compile()


def run_report_subgraph(user_id: str, month: str) -> str:
    result = report_graph.invoke(
        {
            "user_id": user_id,
            "month": month,
            "raw_data": "",
            "report": "",
        }
    )
    return result["report"]
