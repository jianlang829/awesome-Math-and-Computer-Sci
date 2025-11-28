import json, math, tkinter as tk

# ---------- 1. 读图 ----------
with open("beijing_metro.json", encoding="utf-8") as f:
    metro = json.load(f)
nodes = {d["name"]: d["neighbors"] for d in metro}
name2id = {name: i for i, name in enumerate(nodes)}
edges = [(name2id[u], name2id[v]) for u, neigh in nodes.items() for v in neigh if u < v]

# ---------- 2. 简单环状布局 ----------
n = len(nodes)
pos = {
    name2id[name]: (
        300 + 180 * math.cos(2 * i * math.pi / n),
        240 + 180 * math.sin(2 * i * math.pi / n),
    )
    for i, name in enumerate(nodes)
}

# ---------- 3. tkinter 画布 ----------
root = tk.Tk()
root.title("地铁拓扑图")
canvas = tk.Canvas(root, width=700, height=500, bg="white")
canvas.pack()

# 画边
for a, b in edges:
    x1, y1 = pos[a]
    x2, y2 = pos[b]
    canvas.create_line(x1, y1, x2, y2, fill="gray", width=2)

# 画点
deg = {name: len(neigh) for name, neigh in nodes.items()}
dot = {}  # 保存圆 id，后面绑悬停
for name, i in name2id.items():
    x, y = pos[i]
    color = "red" if (deg[name] % 2) else "blue"  # 奇度红色，偶度蓝色
    dot[i] = canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline="black")

# 悬停文字
text_id = None


def show_name(event):
    global text_id
    cx, cy = event.x, event.y
    for name, i in name2id.items():
        x, y = pos[i]
        if (x - cx) ** 2 + (y - cy) ** 2 < 64:  # 8px 内
            if text_id:
                canvas.delete(text_id)
            text_id = canvas.create_text(
                cx + 10,
                cy - 10,
                text=f"{name}  {deg[name]}",
                anchor="nw",
                font=("Arial", 10, "bold"),
            )
            return
    if text_id:
        canvas.delete(text_id)
        text_id = None


canvas.bind("<Motion>", show_name)
tk.mainloop()
