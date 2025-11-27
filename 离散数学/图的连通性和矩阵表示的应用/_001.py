import networkx as nx
import numpy as np

# 1. 任意造一个无向图（也可以从文件读）
G = nx.erdos_renyi_graph(n=8, p=0.05, seed=42)  # 随机图 

"""作者提示
seed影响不同矩阵，p影响图的连通分量（连通性），n影响几个点（对应矩阵为n阶矩阵）
你用的参数 n=8, p=0.25，而 (ln 8)/8 ≈ 0.26，正好踩在临界区上方，所以大部分 seed 都会给出连通分量个数 = 1。
要想轻松看到 2 个甚至 3 个分量，只要把 p 调小或者人为断开即可。"""

A = nx.adjacency_matrix(G).todense()  # 邻接矩阵
print("邻接矩阵:\n", A)

# 2. 连通性判定（对应教材“连通分量”）
n_cc = nx.number_connected_components(G)
print("连通分量个数 =", n_cc)

# 3. 把每个分量染不同颜色画出来
import matplotlib.pyplot as plt

color = [0] * len(G)
for i, comp in enumerate(nx.connected_components(G)):
    for v in comp:
        color[v] = i
nx.draw(G, node_color=color, with_labels=True, cmap=plt.cm.Set3)
plt.title("Connected Components")
plt.savefig(
    "_0011.png",  # 文件名
    dpi=300,  # 分辨率
    bbox_inches="tight",  # 去掉多余白边
    pad_inches=0.1,  # 白边留多少英寸
    facecolor="white",  # 背景色
    transparent=True,  # 是否透明背景
)
plt.close()
