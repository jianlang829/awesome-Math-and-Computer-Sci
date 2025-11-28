import json
from collections import deque

# 1. 读图
with open("beijing_metro.json", encoding="utf-8") as f:
    metro = json.load(f)
G = {s["name"]: s["neighbors"] for s in metro}  # 邻接表

# 2. 度序列 & 欧拉回路判定
deg = {v: len(G[v]) for v in G}
odd_deg_cnt = sum(d & 1 for d in deg.values())
print("奇度站点数 =", odd_deg_cnt)
print("存在欧拉回路?", "Yes" if odd_deg_cnt == 0 else "No")


# 3. 最短路径（BFS，站数=边数）
def bfs_dist(start, goal):
    if start == goal:
        return [start]  # 同一站直接返回单元素列表

    q = deque([(start, 0)])
    seen = {start}
    while q:
        v, d = q.popleft()
        if v == goal:
            return d
        for nxt in G[v]:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return []  # 不连通


def bfs_path(start, goal):
    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft()
        v = path[-1]
        if v == goal:
            return path
        for nxt in G[v]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(path + [nxt])


# 4. 随手测两站
a, b = "西单", "大望路"
print(f"{a} → {b} 最少 {bfs_dist(a, b)} 站")

p = bfs_path("西单", "大望路")
if not p:
    print("找不到路径，请检查站点名或数据是否连通")
else:
    print("具体路线:", " → ".join(p))
