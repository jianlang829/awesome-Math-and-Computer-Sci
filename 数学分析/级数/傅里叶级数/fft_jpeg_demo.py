"""DCT 压缩示例"""

import numpy as np, matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]  # ['Microsoft YaHei'] 微软雅黑 ['FangSong'] (仿宋) ['KaiTi'] (楷体)等
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题

np.random.seed(0)

from scipy.fftpack import dct, idct


# 原始 8×8 块
block = np.random.rand(800, 800)
# DCT 变换
freq = dct(dct(block.T, norm="ortho").T, norm="ortho")
# 只保留左上角 4×4 系数
freq[400:, 400:] = 0
# 反变换重构
rec = idct(idct(freq.T, norm="ortho").T, norm="ortho")

plt.subplot(121)
plt.imshow(block, cmap="gray")
plt.title("原图")
plt.subplot(122)
plt.imshow(rec, cmap="gray")
plt.title("压缩后")
plt.savefig(
    "dct_compress_demo.png",  # 文件名
    dpi=75,  # 分辨率
    bbox_inches="tight",  # 去掉多余白边
    pad_inches=0.1,  # 白边留多少英寸
    facecolor="black",  # 背景色
    transparent=True,  # 是否透明背景
)
plt.close()
