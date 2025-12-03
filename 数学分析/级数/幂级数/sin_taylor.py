"""sin_taylor.py: 用泰勒级数计算sin(x)的近似值"""

import math, time, matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]  # ['Microsoft YaHei'] 微软雅黑 ['FangSong'] (仿宋) ['KaiTi'] (楷体)等
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题


x = 0.5

t0 = time.time()
s = math.sin(x)
t1 = time.time()
print(f"math.sin: {s:.2f}", t1 - t0)

t0 = time.time()
s = 0.5 - 0.5**3 / 6 + 0.5**5 / 120
t1 = time.time()
print(f"泰勒3项:{s:.2f}", t1 - t0, "误差", abs(s - math.sin(x)))

# xs = [i * 0.01 for i in range(1000)]
# plt.plot(xs, [math.sin(x) for x in xs], label="math.sin")
# plt.plot(xs, [x - x**3 / 6 + x**5 / 120 for x in xs], "--", label="泰勒3项")
# plt.legend()
# plt.savefig(
#     "sin_taylor.png",  # 文件名
#     dpi=75,  # 分辨率
#     bbox_inches="tight",  # 去掉多余白边
#     pad_inches=0.1,  # 白边留多少英寸
#     facecolor="black",  # 背景色
#     transparent=True,  # 是否透明背景
# )
# plt.close()
