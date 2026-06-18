import os

from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

external_data: dict = {}


def generate_external_data() -> None:
    if external_data:
        return

    external_data_path = get_abs_path(agent_conf["external_data_path"])
    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

    with open(external_data_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            arr: list[str] = line.strip().split(",")

            user_id: str = arr[0].replace('"', "")
            feature: str = arr[1].replace('"', "")
            efficiency: str = arr[2].replace('"', "")
            consumables: str = arr[3].replace('"', "")
            comparison: str = arr[4].replace('"', "")
            month: str = arr[5].replace('"', "")

            if user_id not in external_data:
                external_data[user_id] = {}

            external_data[user_id][month] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison,
            }


def fetch_user_month_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return str(external_data[user_id][month])
    except KeyError:
        return ""
