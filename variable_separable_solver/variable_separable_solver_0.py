#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
variable_separable_solver.py
求解形如 dy/dx = f(x) * g(y) 的变量分离型一阶常微分方程（数值方法）
严格流程：分离变量（1/g(y) dy = f(x) dx）→ 两端数值积分（scipy.integrate.quad）
→ 反求 y(x)（用根求解 brentq）。
并使用 solve_ivp (RK45) 作为基准解比较，绘图并计算 RMSE。
依赖：numpy, matplotlib, scipy
安装：pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import sys

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # ['Microsoft YaHei'] 微软雅黑 ['FangSong'] (仿宋) ['KaiTi'] (楷体)等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题


# ----------------------- 参数设置模块（可修改区域） -----------------------
# 在这里直接修改 f_x(x) 和 g_y(y) 的表达式
# 必须保证这两个函数在求解区间上可积且数值上合理

# 示例选择（设置为 1 启用示例1，设置为 2 启用示例2）
EXAMPLE = 1  # 默认启用示例1；将其改为 2 可切换到示例2（并参考下方示例2替换说明）

# 示例1： dy/dx = x * y
def f_x(x):
    # 这里修改 f(x)
    return x

def g_y(y):
    # 这里修改 g(y)
    return y

# 如果要使用示例2（dy/dx = sin(x)*(1+y^2)），请把上面的 f_x/g_y 替换为：
# def f_x(x):
#     return np.sin(x)
# def g_y(y):
#     return 1 + y**2
# 或者将 EXAMPLE = 2，然后在下方参数配置里替换为示例2（已给出说明）

# 初始条件与求解参数（可修改）
x0 = 0.0         # 初始 x（或求解区间起点）
y0 = 1.0         # 初始 y（必须满足 g_y(y0) != 0）
x_end = 2.0      # 求解区间终点
n_steps = 200    # 迭代步数（步长 h 自动计算为 (x_end-x0)/n_steps ）
# -------------------------------------------------------------------------

# ----------------------- 合法性检查 -----------------------
if g_y(y0) == 0:
    print("错误：初始值 y0 导致 g(y0) == 0，会在分离变量时导致除以零。请更换 y0 或修改 g_y(y)。")
    sys.exit(1)

# 设置 x 网格
xs = np.linspace(x0, x_end, n_steps + 1)

# ----------------------- 核心求解函数模块 -----------------------
# 计算 G(x) = \int_{x0}^{x} f(s) ds
def G_of_x(x):
    # quad 返回 (value, err)
    val, err = quad(lambda s: f_x(s), x0, x, limit=200)
    return val

# 计算 F(y) = \int_{y0}^{y} 1/g(t) dt
def F_of_y(y):
    # integrand = 1 / g_y(t)
    def integrand(t):
        gt = g_y(t)
        if gt == 0:
            # 为避免除零抛出，返回大型数值（quad 在遇到真正奇异点可能报错）
            # 但更稳妥的做法是让 quad 发现并报错，由上层捕获
            raise ZeroDivisionError(f"g(y) 在 t={t} 处为零，积分发散或不可直接计算。")
        return 1.0 / gt
    # quad 支持上下限反向（会返回负值）
    val, err = quad(integrand, y0, y, limit=200)
    return val

# H(y; rhs) = F(y) - rhs，这里 rhs = G(x) 为常数
def H_of_y(y, rhs):
    return F_of_y(y) - rhs

# 在给定 x 网格上用“分离变量数值法”求 y(x)
def solve_by_separation(xs, y0):
    ys = np.zeros_like(xs)
    ys[0] = y0
    # 预计算 G(x) 对每个 x
    Gs = np.array([G_of_x(x) for x in xs])

    # 根求解区间（来自要求）：以 [y0 - 100, y0 + 100] 为初始搜索区间
    global_search_low = y0 - 100.0
    global_search_high = y0 + 100.0

    # 为每个 x（从索引1开始）求出 y，使得 F(y) = G(x)
    for i in range(1, len(xs)):
        rhs = Gs[i]  # 这是 F(y) 的目标值
        # 特殊情况：如果 rhs == 0（即 x == x0），则 y = y0
        if abs(rhs) < 1e-15:
            ys[i] = y0
            continue

        # 为了稳定求根，先在 global 区间内做采样，寻找符号变化的子区间
        N_samples = 400  # 采样点数（可调整，越多越稳健但慢）
        sample_points = np.linspace(global_search_low, global_search_high, N_samples)
        H_vals = []
        finite_flags = []
        for sp in sample_points:
            try:
                hv = H_of_y(sp, rhs)
                if np.isfinite(hv):
                    H_vals.append(hv)
                    finite_flags.append(True)
                else:
                    H_vals.append(np.nan)
                    finite_flags.append(False)
            except Exception:
                H_vals.append(np.nan)
                finite_flags.append(False)

        # 找到相邻点有符号变化的区间
        bracket_found = False
        y_root = None
        for j in range(N_samples - 1):
            if not (finite_flags[j] and finite_flags[j + 1]):
                continue
            v1 = H_vals[j]
            v2 = H_vals[j + 1]
            if v1 == 0.0:
                y_root = sample_points[j]
                bracket_found = True
                break
            if v2 == 0.0:
                y_root = sample_points[j + 1]
                bracket_found = True
                break
            if v1 * v2 < 0:
                a = sample_points[j]
                b = sample_points[j + 1]
                # 使用 brentq 在 (a, b) 上求根
                try:
                    y_root = brentq(lambda yy: H_of_y(yy, rhs), a, b, maxiter=200)
                    bracket_found = True
                    break
                except Exception as e:
                    # 如果 brentq 失败，继续尝试下一个符号变化区间
                    # print(f"brentq 在区间 ({a}, {b}) 失败：{e}")
                    continue

        if not bracket_found:
            # 如果没有在初始全局区间找到符号变化，尝试扩展搜索区间（线性扩展两倍）
            expand_factor = 2.0
            tries = 0
            max_tries = 3
            expanded_low = global_search_low
            expanded_high = global_search_high
            while not bracket_found and tries < max_tries:
                expanded_low = y0 + (expanded_low - y0) * expand_factor
                expanded_high = y0 + (expanded_high - y0) * expand_factor
                sample_points = np.linspace(expanded_low, expanded_high, N_samples)
                H_vals = []
                finite_flags = []
                for sp in sample_points:
                    try:
                        hv = H_of_y(sp, rhs)
                        if np.isfinite(hv):
                            H_vals.append(hv)
                            finite_flags.append(True)
                        else:
                            H_vals.append(np.nan)
                            finite_flags.append(False)
                    except Exception:
                        H_vals.append(np.nan)
                        finite_flags.append(False)
                for j in range(N_samples - 1):
                    if not (finite_flags[j] and finite_flags[j + 1]):
                        continue
                    v1 = H_vals[j]
                    v2 = H_vals[j + 1]
                    if v1 == 0.0:
                        y_root = sample_points[j]
                        bracket_found = True
                        break
                    if v2 == 0.0:
                        y_root = sample_points[j + 1]
                        bracket_found = True
                        break
                    if v1 * v2 < 0:
                        a = sample_points[j]
                        b = sample_points[j + 1]
                        try:
                            y_root = brentq(lambda yy: H_of_y(yy, rhs), a, b, maxiter=200)
                            bracket_found = True
                            break
                        except Exception:
                            continue
                tries += 1

        if not bracket_found:
            raise RuntimeError(f"在为 x={xs[i]:.6g} 求解 y 时，无法在 [{global_search_low}, {global_search_high}]（及扩展区间）中找到使 F(y)-G(x) 有符号变化的区间，无法用 brentq 求根。"
                               " 可能原因：g(y) 在区间内有奇点，或解超出搜索区间范围。请扩大搜索区间或检查 g(y)。")

        ys[i] = y_root

    return ys

# ----------------------- 基准验证模块（RK45） -----------------------
def solve_by_rk45(x0, y0, x_end, xs):
    def rhs(x, y):
        return f_x(x) * g_y(y)
    sol = solve_ivp(rhs, (x0, x_end), [y0], method='RK45', t_eval=xs, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError("solve_ivp (RK45) 未能成功求解： " + str(sol.message))
    return sol.y[0]

# ----------------------- 结果可视化与误差计算模块 -----------------------
def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def visualize_and_report(xs, ys_sep, ys_rk, example=1, analytic_solution=None):
    # 计算并打印 RMSE（保留10位小数）
    rmse_val = compute_rmse(ys_sep, ys_rk)
    print(f"分离变量数值解 与 RK45 数值解 的 RMSE = {rmse_val:.10f}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys_sep, label="分离变量数值解", lw=2)
    plt.plot(xs, ys_rk, '--', label="RK45 数值解", lw=2)
    if example == 1 and analytic_solution is not None:
        ys_analytic = analytic_solution(xs)
        plt.plot(xs, ys_analytic, ':', label="解析解", lw=2)
        # 额外计算并打印 分离变量数值解 与 解析解 的误差（示例1的要求）
        abs_err = np.abs(ys_sep - ys_analytic)
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)
        print(f"分离变量数值解 与 解析解 的 最大绝对误差 = {max_err:.10f}")
        print(f"分离变量数值解 与 解析解 的 平均绝对误差 = {mean_err:.10f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("变量分离法 与 RK45 解的比较")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figure_2.png")

# ----------------------- 主程序（执行流程） -----------------------
def main():
    # 如果需要按示例2切换 f_x/g_y，可在此处替换函数定义（演示说明）
    # if EXAMPLE == 2:
        # 示例2：dy/dx = sin(x) * (1 + y^2)
        # 说明：要切换到示例2，请解除下面两行的注释（或者直接在上方参数区域替换 f_x 和 g_y）
        # global f_x, g_y
        # f_x = lambda x: np.sin(x); g_y = lambda y: 1 + y**2

    print("开始用分离变量法计算（数值积分 + 根求解）...")
    ys_sep = solve_by_separation(xs, y0)
    print("分离变量法计算完成。")

    print("开始用 solve_ivp (RK45) 作为基准解...")
    ys_rk = solve_by_rk45(x0, y0, x_end, xs)
    print("RK45 计算完成。")

    # 若为示例1，构造解析解函数
    analytic_solution = None
    if EXAMPLE == 1:
        # 解析解：dy/dx = x*y -> ln(y) - ln(y0) = (x^2 - x0^2)/2
        def analytic_sol(x_arr):
            return y0 * np.exp((x_arr ** 2 - x0 ** 2) / 2.0)
        analytic_solution = analytic_sol

    # 可视化并输出误差
    visualize_and_report(xs, ys_sep, ys_rk, example=EXAMPLE, analytic_solution=analytic_solution)

if __name__ == "__main__":
    main()