import json

stations = [
    "苹果园",
    "古城",
    "八角游乐园",
    "八宝山",
    "玉泉路",
    "五棵松",
    "万寿路",
    "公主坟",
    "军事博物馆",
    "木樨地",
    "南礼士路",
    "复兴门",
    "西单",
    "天安门西",
    "天安门东",
    "王府井",
    "东单",
    "建国门",
    "永安里",
    "国贸",
    "大望路",
    "四惠",
    "四惠东",
]
metro = []
for i, s in enumerate(stations):
    n = []
    if i > 0:
        n.append(stations[i - 1])  # 左边一站
    if i < len(stations) - 1:
        n.append(stations[i + 1])  # 右边一站
    metro.append({"name": s, "lines": ["1号线"], "neighbors": n})
open("beijing_metro.json", "w", encoding="utf-8").write(
    json.dumps(metro, ensure_ascii=False)
)
