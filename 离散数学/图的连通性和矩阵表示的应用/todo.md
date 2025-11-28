下面给出一条“从理论 → 代码 → 可视化 → 小实验报告”的完整路线，让你把“连通性 + 矩阵表示”这两块离散数学知识真正在计算机上跑一遍。  
全程用 Python（含 NetworkX + NumPy + Matplotlib），不依赖黑箱工具，每一步都对应回课本概念，10 行代码就能出第一张图，30 行就能做完整实验。

--------------------------------------------------
一、准备环境（一条命令）
```bash
pip install networkx numpy matplotlib scipy
```
--------------------------------------------------
二、快速热身：把“图”变成矩阵，再判断连通性
```python
import networkx as nx
import numpy as np

# 1. 任意造一个无向图（也可以从文件读）
G = nx.erdos_renyi_graph(n=8, p=0.25, seed=42)   # 随机图
A = nx.adjacency_matrix(G).todense()              # 邻接矩阵
print("邻接矩阵:\n", A)

# 2. 连通性判定（对应教材“连通分量”）
n_cc = nx.number_connected_components(G)
print("连通分量个数 =", n_cc)

# 3. 把每个分量染不同颜色画出来
import matplotlib.pyplot as plt
color = [0]*len(G)
for i, comp in enumerate(nx.connected_components(G)):
    for v in comp: color[v] = i
nx.draw(G, node_color=color, with_labels=True, cmap=plt.cm.Set3)
plt.title("Connected Components")
plt.show()
```
跑完你就看到：  
- 终端里 8×8 的 0-1 矩阵（邻接矩阵）  
- 弹出的窗口里不同颜色的小块（直观对应“连通分量”概念）

--------------------------------------------------
三、进阶：有向图的强连通分量（SCC）+ 可达矩阵
```python
# 1. 造一个有向图
D = nx.scale_free_graph(30, seed=1)   # 30 个顶点的有向网络
D = nx.DiGraph(D)                     # 转成普通有向图

# 2. 强连通分量
scc = list(nx.strongly_connected_components(D))
print("强连通分量列表:", scc)

# 3. 可达矩阵（概念对应教材“可达性”）
n = D.number_of_nodes()
reach = np.zeros((n, n), dtype=int)
for u in D:
    reachable = nx.single_source_shortest_path_length(D, u)
    for v in reachable: reach[u][v] = 1
print("可达矩阵:\n", reach)

# 4. 可视化：把 SCC 缩成超级节点，画 DAG
scc_graph = nx.condensation(D)        # NetworkX 自带“缩点”函数
nx.draw(scc_graph, with_labels=True, node_color='coral')
plt.title("Condensation DAG (each node = a SCC)")
plt.show()
```
此时你已经把“强连通分量”“缩点”“可达矩阵”全部跑通，而且图是彩色的。

--------------------------------------------------
四、把“矩阵”再喂回线性代数——谱判定
课本上常提：  
- 无向图邻接矩阵的最大特征值 λ₁ 与连通性有关  
- 拉普拉斯矩阵第二小特征值 λ₂（代数连通度）>0 ⇔ 图连通

```python
L = nx.laplacian_matrix(G).todense()
eig = np.linalg.eigvalsh(L)
print("拉普拉斯特征值:", eig)
print("代数连通度 λ₂ =", eig[1])
if eig[1] > 1e-10:
    print("=> 图是连通的（λ₂>0）")
else:
    print("=> 图不连通")
```
这一段把“代数连通度”真正算了出来，和前面的 `nx.number_connected_components` 结果互相验证。

--------------------------------------------------
五、可扩展的小课程项目（任选其一）
1. 社交网络小实验  
   - 拿 Twitter/微博关注数据（或公开数据集 `ego-Facebook`）  
   - 用邻接矩阵存图，算 SCC，找出“大 V”是否处于 giant SCC  
   - 报告：画出分量大小分布直方图，写两句解释。

2. 道路网络鲁棒性测试  
   - 用 OpenStreetMap 导出你所在城市的道路（NetworkX 可直接读 `.osm`）  
   - 随机删掉 5% 的边（模拟事故），每删一次算 `n_cc` 和 `λ₂`  
   - 画曲线：横轴 = 删除比例，纵轴 = 连通分量数 / λ₂，观察突变点。

3. 迷宫生成与唯一解判定  
   - 用深度优先生成迷宫（网格图）  
   - 把迷宫看成无向图，保证它是连通图；再随机加一条边，检测是否出现环（邻接矩阵求迹）。  
   - 可视化迷宫路径 & 环。

--------------------------------------------------
六、30 行完整模板（可直接交作业）
```python
import networkx as nx, numpy as np, matplotlib.pyplot as plt
def report(G, name):
    A = nx.adjacency_matrix(G).todense()
    print(f"\n=== {name} ===")
    print("邻接矩阵:\n", A)
    if nx.is_directed(G):
        scc = list(nx.strongly_connected_components(G))
        print("强连通分量数:", len(scc))
        nx.draw(nx.condensation(G), node_color='coral')
    else:
        print("连通分量数:", nx.number_connected_components(G))
        nx.draw(G, node_color=[hash(tuple(comp))%10 for comp in nx.connected_components(G)], cmap=plt.cm.Set3)
    plt.show()

# 实验1：随机无向图
report(nx.erdos_renyi_graph(10, 0.3, seed=1), "Random Undirected")

# 实验2：随机有向图
report(nx.erdos_renyi_graph(15, 0.2, directed=True, seed=2), "Random Directed")
```
--------------------------------------------------
七、把结果写进实验报告（模板）
1. 理论回顾（50 字）：给出连通分量、强连通、邻接矩阵、可达矩阵定义。  
2. 实验目的（30 字）：用代码验证定义并可视化。  
3. 关键代码截图：把上面 30 行粘过去即可。  
4. 结果解释：  
   - 无向图分量颜色图 → 对应矩阵里 0 块解释。  
   - 有向图 SCC 缩点 DAG → 说明“偏序”关系。  
5. 结论（20 字）：矩阵与图算法结果一致，连通性判定正确。

--------------------------------------------------
至此，你把“离散数学”里的  
- 邻接矩阵 / 可达矩阵  
- 连通分量 / 强连通分量 / 缩点  
- 代数连通度  

全部在计算机上跑了一遍，还得到了彩色图片和数值报告。  
拿去当课程设计、实验报告、GitHub 小项目都可以，后续再换更大数据集或 C++/Java 实现也只要替换网络 IO 部分即可。祝玩得开心!