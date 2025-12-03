import numpy as np
from scipy.integrate import tplquad


# 被积函数
def integrand(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


# 积分限
r_limits = lambda theta, phi: [0, 1]
theta_limits = lambda phi: [0, np.pi]
phi_limits = lambda: [0, 2 * np.pi]

# 计算积分
result, error = tplquad(
    integrand,
    phi_limits()[0],
    phi_limits()[1],
    theta_limits,
    phi_limits,
    r_limits,
    theta_limits,
)

print(f"球体的质量为: {result}")
print(f"积分误差估计: {error}")
