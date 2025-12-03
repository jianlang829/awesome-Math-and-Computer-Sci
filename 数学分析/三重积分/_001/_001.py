# ========== 1. 导入 ==========
import numpy as np
import sympy as sp
from sympy import lambdify
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]  # ['Microsoft YaHei'] 微软雅黑 ['FangSong'] (仿宋) ['KaiTi'] (楷体)等
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题


# ========== 2. 定义积分区域 ==========
# 解析/数值通用的上下限
# 0 ≤ x ≤ 2
# 0 ≤ y ≤ 2 - x
# 0 ≤ z ≤ 2 - x
xlims = (1, 2)


def y_lower(x):
    return 0


def y_upper(x):
    return x


def z_lower(x, y):
    return 0


def z_upper(x, y):
    return y


# ========== 3. 被积函数 ==========
x, y, z = sp.symbols("x y z", real=True)
integrand_sympy = 1 / (x**2 + y**2)
integrand_numpy = lambda x, y, z: 1.0 / (x**2 + y**2)

# ========== 4. SymPy 尝试解析解 ==========
try:
    I_sympy = sp.integrate(
        integrand_sympy,
        (z, z_lower(x, y), z_upper(x, y)),
        (y, y_lower(x), y_upper(x)),
        (x, *xlims),
    )
    print("SymPy 解析解：", I_sympy)
    print("数值近似：", float(I_sympy.evalf()))
except Exception as e:
    print("SymPy 无法求出解析解，原因：", e)
    I_sympy = None

# ========== 5. 若解析失败，用 SciPy 数值积分 ==========
if I_sympy is None:
    res, err = integrate.tplquad(
        integrand_numpy,
        xlims[0],
        xlims[1],  # x 范围
        y_lower,
        y_upper,  # y 范围
        z_lower,
        z_upper,  # z 范围
    )
    print("SciPy 数值解：", res, "±", err)

# ========== 6. 可视化积分区域 ==========
fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor("black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.zaxis.label.set_color("white")

# 生成网格
n = 50
x_vals = np.linspace(0, 2, n)
y_vals = np.linspace(0, 2, n)
X, Y = np.meshgrid(x_vals, y_vals)

# 只保留满足 y ≤ 2 - x 的区域
mask = Y <= (X)
X = X[mask]
Y = Y[mask]

# 下表面 z = 0
Z_low = np.zeros_like(X)
# 上表面 z = 2 - x
Z_high = y
z_func = lambdify((x, y), Z_high, modules="numpy")
Z_high = z_func(X, Y)

# 画半透明“立体”
ax.plot_trisurf(X, Y, Z_low, color="C0", alpha=0.15, label="z=0")
ax.plot_trisurf(
    X.flatten(), Y.flatten(), Z_high.flatten(), color="C1", alpha=0.25, label="z=y"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("积分区域")
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.savefig(
    "_001.png",  # 文件名
    dpi=300,  # 分辨率
    bbox_inches="tight",  # 去掉多余白边
    pad_inches=0.1,  # 白边留多少英寸
    facecolor="black",  # 背景色
    transparent=True,  # 是否透明背景
)
plt.close()
