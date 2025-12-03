from scipy import integrate
import numpy as np


def integrand(z, y, x):
    return y * np.cos(x + z)  # 注意参数顺序：最内层变量放最前


# 积分限：最内层→最外层
# 积分限
a, b = 0, 1  # x 常数限
gfun = lambda x: 0  # y 下限
hfun = lambda x: 1 - x  # y 上限
qfun = lambda y, x: 0  # z 下限
rfun = lambda y, x: 1 - x - y  # z 上限

val, err = integrate.tplquad(integrand, a, b, gfun, hfun, qfun, rfun)


print(f"积分值 ≈ {val:.2f}")
print("估计误差 ≈", err)
