import json
import os
import re
from dataclasses import dataclass

import requests

from utils.logger_handler import logger

REQUEST_TIMEOUT = 8

# WMO 天气代码 → 中文描述（Open-Meteo）
WMO_WEATHER_DESC = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "多云",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}

WIND_DIRECTIONS = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]

_location_override: str | None = None
_last_ip_location: "IpLocation | None" = None


@dataclass
class IpLocation:
    city: str
    district: str = ""
    lat: float | None = None
    lon: float | None = None
    source: str = ""

    @property
    def label(self) -> str:
        city = self.city.replace("市", "")
        if self.district:
            district = self.district.replace("区", "").replace("县", "").replace("市", "")
            if district and district not in city:
                return f"{city}{district}"
        return city


def set_location_override(city: str | None) -> None:
    """手动指定城市，优先级高于 IP 定位（Web 界面侧边栏使用）。"""
    global _location_override
    _location_override = city.strip().replace("市", "") if city and city.strip() else None


def _normalize_city(name: str) -> str:
    return name.strip().replace("市", "").replace("省", "")


def _cache_location(location: IpLocation) -> IpLocation:
    global _last_ip_location
    _last_ip_location = location
    return location


def _lookup_ip9() -> IpLocation | None:
    resp = requests.get("https://ip9.com.cn/get", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("ret") != 200:
        return None

    data = payload.get("data") or {}
    city = _normalize_city(data.get("city", ""))
    if not city:
        return None

    lat = float(data["lat"]) if data.get("lat") else None
    lon = float(data["lng"]) if data.get("lng") else None
    district = (data.get("area") or "").strip()

    return IpLocation(city=city, district=district, lat=lat, lon=lon, source="ip9")


def _lookup_pconline() -> IpLocation | None:
    resp = requests.get(
        "https://whois.pconline.com.cn/ipJson.jsp?json=true",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.content.decode("gbk", errors="ignore")
    text = re.sub(r"^[^(]*\(", "", text).rstrip(");")
    data = json.loads(text)

    city = _normalize_city(data.get("city", ""))
    if not city:
        return None

    region = (data.get("region") or "").strip()
    return IpLocation(city=city, district=region, source="pconline")


def _lookup_amap() -> IpLocation | None:
    api_key = os.getenv("AMAP_API_KEY")
    if not api_key:
        return None

    resp = requests.get(
        "https://restapi.amap.com/v3/ip",
        params={"key": api_key},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        logger.warning(f"[get_location_by_ip] 高德定位失败：{data.get('info')}")
        return None

    city = _normalize_city(data.get("city", ""))
    if not city:
        return None

    return IpLocation(city=city, source="amap")


def _lookup_ip_api() -> IpLocation | None:
    resp = requests.get(
        "http://ip-api.com/json/?lang=zh-CN&fields=status,city,regionName,lat,lon",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        return None

    city = _normalize_city(data.get("city") or data.get("regionName", ""))
    if not city:
        return None

    return IpLocation(
        city=city,
        lat=data.get("lat"),
        lon=data.get("lon"),
        source="ip-api",
    )


def get_location_by_ip() -> str:
    """根据公网 IP 定位城市；支持手动覆盖。"""
    if _location_override:
        logger.info(f"[get_location_by_ip] 使用手动指定城市：{_location_override}")
        return _cache_location(IpLocation(city=_location_override, source="manual")).label

    providers = (_lookup_ip9, _lookup_pconline, _lookup_amap, _lookup_ip_api)
    errors: list[str] = []

    for provider in providers:
        name = provider.__name__.replace("_lookup_", "")
        try:
            location = provider()
            if location:
                logger.info(
                    f"[get_location_by_ip] {name} 定位成功：{location.label}"
                    f"（{location.lat},{location.lon}）"
                )
                return _cache_location(location).label
        except Exception as e:
            errors.append(f"{name}: {e}")
            logger.warning(f"[get_location_by_ip] {name} 定位失败：{e}")

    raise RuntimeError(
        "IP定位失败，请在页面侧边栏手动填写所在城市，或在问题中直接说明城市。"
        + (f" 详情：{' | '.join(errors)}" if errors else "")
    )


def _wind_direction_text(degree: float) -> str:
    index = int((degree + 22.5) / 45) % 8
    return WIND_DIRECTIONS[index]


def _wind_level(speed_kmh: float) -> str:
    if speed_kmh < 1:
        return "0级"
    if speed_kmh < 6:
        return "1级"
    if speed_kmh < 12:
        return "2级"
    if speed_kmh < 20:
        return "3级"
    if speed_kmh < 29:
        return "4级"
    if speed_kmh < 39:
        return "5级"
    if speed_kmh < 50:
        return "6级"
    return "7级及以上"


def _format_weather(location_label: str, current: dict, max_rain_prob: float) -> str:
    weather_code = int(current.get("weather_code", 0))
    weather_desc = WMO_WEATHER_DESC.get(weather_code, "未知")
    temp = current.get("temperature_2m", 0)
    humidity = current.get("relative_humidity_2m", 0)
    wind_speed = current.get("wind_speed_10m", 0)
    wind_dir = _wind_direction_text(current.get("wind_direction_10m", 0))
    wind_level = _wind_level(wind_speed)
    precipitation = current.get("precipitation", 0)
    rain_hint = "较高" if max_rain_prob >= 60 else "中等" if max_rain_prob >= 30 else "极低"

    return (
        f"城市{location_label}当前天气{weather_desc}，"
        f"气温{temp:.0f}摄氏度，空气湿度{humidity:.0f}%，"
        f"{wind_dir}风{wind_level}（风速{wind_speed:.0f}公里/小时），"
        f"当前降水量{precipitation:.1f}毫米，"
        f"未来6小时降雨概率{max_rain_prob:.0f}%（{rain_hint}）"
    )


def _fetch_weather_by_coords(lat: float, lon: float, location_label: str) -> str:
    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m,precipitation",
            "hourly": "precipitation_probability",
            "forecast_hours": 6,
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT,
    )
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()

    current = weather_data["current"]
    hourly_prob = weather_data.get("hourly", {}).get("precipitation_probability") or []
    max_rain_prob = max(hourly_prob) if hourly_prob else 0
    return _format_weather(location_label, current, max_rain_prob)


def get_weather_by_city(city: str) -> str:
    """根据城市名或 IP 缓存坐标查询实时天气。"""
    city = _normalize_city(city)
    if not city:
        raise ValueError("城市名称不能为空")

    cached = _last_ip_location
    if cached and cached.lat is not None and cached.lon is not None:
        cached_city = _normalize_city(cached.city)
        if city == cached_city or city == cached.label or city in cached.label:
            logger.info(f"[get_weather_by_city] 使用 IP 定位坐标查询：{cached.label}")
            return _fetch_weather_by_coords(cached.lat, cached.lon, cached.label)

    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5, "language": "zh", "format": "json"},
        timeout=REQUEST_TIMEOUT,
    )
    geo_resp.raise_for_status()
    results = geo_resp.json().get("results") or []
    if not results:
        raise RuntimeError(f"未找到城市「{city}」的地理信息，请检查城市名称")

    # 优先匹配中国区划，减少同名城市偏差
    location = results[0]
    for item in results:
        if item.get("country_code") == "CN":
            location = item
            if _normalize_city(item.get("name", "")) == city:
                break

    lat = location["latitude"]
    lon = location["longitude"]
    resolved_name = location.get("name", city)
    admin1 = location.get("admin1", "")
    location_label = f"{admin1}{resolved_name}" if admin1 else resolved_name

    return _fetch_weather_by_coords(lat, lon, location_label)
