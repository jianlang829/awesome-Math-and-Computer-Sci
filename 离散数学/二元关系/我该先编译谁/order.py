import glob, networkx as nx, os, sys

DEP_DIR = "build"  # ← 学生只改这里
G = nx.DiGraph()
for f in glob.glob(f"{DEP_DIR}/*.d"):
    tgt, *deps = open(f).read().replace("\\\n", "").split()
    tgt = tgt[:-1]  # 去掉行末冒号
    for d in deps:
        G.add_edge(d, tgt)
print("编译顺序:", " ".join(nx.topological_sort(G)))
