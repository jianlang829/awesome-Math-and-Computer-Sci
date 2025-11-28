import pandas as pd, time, matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]  # ['Microsoft YaHei'] 微软雅黑 ['FangSong'] (仿宋) ['KaiTi'] (楷体)等
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题


t0 = time.time()
df = pd.read_csv("follows.csv", header=None, names=["a", "b"])
mutual = pd.merge(df, df.rename(columns={"a": "b", "b": "a"}), on=["a", "b"]).shape[0]
ratio = mutual / len(df)
print(f"对称率(互粉率): {ratio:.2%}  耗时: {time.time()-t0:.2f}s")

plt.pie([ratio, 1 - ratio], labels=["互粉", "单向"], autopct="%1.1f%%")
plt.title("微博互粉比例")
plt.savefig("mutual_rate.png", dpi=150)
plt.close()
