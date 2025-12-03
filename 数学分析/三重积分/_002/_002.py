import time

def jx():
    start_time = time.time()

    import sympy as sp

    x, y, z = sp.symbols("x y z")
    ans = sp.integrate(
        x * sp.cos(y) * sp.cos(z), (x, 0, 1), (y, 0, sp.pi / 2), (z, 0, sp.pi / 2)
    )
    print(ans)
    end_time = time.time()
    cal_time = end_time - start_time
    print(f"耗时：{cal_time:.2f}")


def cal():
    start_time = time.time()
    from scipy.integrate import tplquad
    import numpy as np

    # 被积函数 f(x,y,z) = x cos y cos z
    f = lambda z, y, x: x * np.cos(y) * np.cos(z)

    # 积分限
    x_a, x_b = 0, 1
    y_a, y_b = lambda x: 0, lambda x: np.pi / 2
    z_a, z_b = lambda x, y: 0, lambda x, y: np.pi / 2

    val, abserr = tplquad(f, x_a, x_b, y_a, y_b, z_a, z_b)
    print(f"积分值 = {val:.2f}")
    print("估计误差 =", abserr)
    end_time = time.time()
    cal_time = end_time - start_time
    print(f"耗时：{cal_time:.2f}")


if __name__ == "__main__":
    user = input("1:cal 2:jx \n: ")
    if user == "1":
        cal()
    if user == "2":
        jx()
