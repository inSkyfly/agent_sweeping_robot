from datetime import datetime
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool

from rag.rag_service import RagSummarizeService
from utils.external_data import fetch_user_month_data
from utils.geo_weather import get_location_by_ip, get_weather_by_city
from utils.logger_handler import logger

rag = RagSummarizeService()


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    try:
        return get_weather_by_city(city)
    except Exception as e:
        logger.error(f"[get_weather]查询失败：{e}")
        return f"获取城市「{city}」天气失败：{e}"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location(
    runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> str:
    city = runtime.context.get("city", "").strip()
    if city:
        return city
    try:
        return get_location_by_ip()
    except Exception as e:
        logger.error(f"[get_user_location]IP定位失败：{e}")
        return f"IP定位失败：{e}"


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id(
    runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> str:
    return str(runtime.context.get("user_id", "1001"))


@tool(description="获取当前月份，以纯字符串形式返回，格式为YYYY-MM")
def get_current_month(
    runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> str:
    return str(runtime.context.get("current_month", datetime.now().strftime("%Y-%m")))


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    data = fetch_user_month_data(user_id, month)
    if not data:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
    return data


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


@tool(description="生成指定月份的扫地机器人个性化使用报告。month 为空时使用当前月份。")
def generate_usage_report(
    month: str = "",
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> str:
    from agent.report_subgraph import run_report_subgraph

    runtime.context["report"] = True
    user_id = str(runtime.context.get("user_id", "1001"))
    if not month:
        month = str(runtime.context.get("current_month", datetime.now().strftime("%Y-%m")))
    return run_report_subgraph(user_id=user_id, month=month)
