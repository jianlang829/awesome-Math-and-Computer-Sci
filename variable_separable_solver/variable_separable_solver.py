# -*- coding: utf-8 -*-
'''tree /f
文件夹 PATH 列表
卷序列号为 D204-C902
C:.
│  .gitignore
│  LICENSE
│  README.md
│  __init__.py
│
├─homogeneous_solver
│  │  homogeneous_solver.py
│  │  homogeneous_solver_0.py
│  │
│  └─__pycache__
│          homogeneous_solver.cpython-312.pyc
│
├─plots
│      Figure_20251114_152229.png
│
└─variable_separable_solver
    │  Figure_1.png
    │  Figure_2.png
    │  q1.md
    │  separate.py
    │  variable_separable_solver.py
    │  variable_separable_solver_0.py
    │  variable_separable_solver_00.py
    │
    └─__pycache__
            variable_separable_solver.cpython-312.pyc'''
'''python -m variable_separable_solver.variable_separable_solver'''
'''
variable_separable_solver.py
求解形如 dy/dx = f(x) * g(y) 的通用变量分离型一阶常微分方程(数值方法，积分 + 求根，非解析方法)
严格流程:分离变量(1/g(y) dy = f(x) dx)→ 两端数值积分(scipy.integrate.quad)
→ 反求 y(x)(用根求解 brentq)。
并使用 solve_ivp (RK45) 作为基准解比较，绘图并计算 RMSE。
方法在数学原理上就是变量分离法，只是因为没有解析解（或没用到解析解），才用 “数值积分 + 数值求根” 来落地 —— 这是变量分离法在实际计算中的常见做法（尤其是复杂方程）
为什么不用解析解，在实际中很多复杂方程没有解析解，只能用数值解
依赖:numpy, matplotlib, scipy

安装:pip install --upgrade numpy matplotlib scipy
作者:jianlang829,lang306
'''

import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import sys

'''在使用 Python 的 Matplotlib 库进行数据可视化时，你可能会遇到中文显示为方框或乱码的问题。这是因为 Matplotlib 默认字体配置不支持中文。本教程将提供多种方法，从简单快捷到永久配置，帮你彻底解决 Matplotlib 中文乱码问题，让你的图表完美展示中文信息。
参考致谢:https://zhuanlan.zhihu.com/p/30790786209'''
# 设置全局字体为 SimHei (黑体) 或其他中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # ['Microsoft YaHei'] 微软雅黑 ['FangSong'] (仿宋) ['KaiTi'] (楷体)等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# ----------------------- 参数设置模块(可修改区域) -----------------------
# 在这里直接修改 f_x(x) 和 g_y(y) 的表达式
# 必须保证这两个函数在求解区间上可积且数值上合理

# 示例选择(设置为 1 启用示例1，设置为 2 启用示例2)
EXAMPLE = 1  # 默认启用示例1；将其改为 2 可切换到示例2(并参考下方示例2替换说明)

# 示例1: dy/dx = x * y
def _default_f_x_example1(x):
    # 这里修改 f(x)
    return x

def _default_g_y_example1(y):
    # 这里修改 g(y)
    return y

# 示例2: dy/dx = sin(x) * (1 + y^2)
def _default_f_x_example2(x):
    return np.sin(x)

def _default_g_y_example2(y):
    return 1 + y**2

# 初始条件与求解参数(可修改)
# 默认值仅用于直接运行时；当被其他程序调用时，可以通过参数覆盖这些默认值
_default_x0 = 0.0         # 初始 x(或求解区间起点)
_default_y0 = 1.0         # 初始 y(必须满足 g_y(y0) != 0)
_default_x_end = 2.0      # 求解区间终点
_default_n_steps = 200

# -------------------------------------------------------------------------

# ----------------------- 合法性检查 -----------------------
# 合法性检查将在具体求解函数中对传入的 g_y 和 y0 做检查（以便作为模块调用时能灵活使用）
# -------------------------------------------------------------------------

# ----------------------- 核心求解函数(可被外部调用) -----------------------
def solve_by_separation(f_x, g_y, x0=_default_x0, y0=_default_y0,
                        x_end=_default_x_end, n_steps=_default_n_steps,
                        N_samples=400, quad_limit=200,
                        expand_factor=2.0, max_tries=3):
    '''
    在给定 x 网格上用“分离变量数值法”求 y(x)
    - f_x: 可调用对象 f(x)
    - g_y: 可调用对象 g(y)
    - x0, y0, x_end: 初始条件与终点
    - n_steps: 将区间分为 n_steps 步 (生成 n_steps+1 个点)
    - N_samples: 在搜索区间上采样的点数(用于找到符号变化区间)
    - quad_limit: quad 的 limit 参数(最大细分数)
    - expand_factor,max_tries: 在未找到区间时的扩展策略 (保持与原逻辑一致)
    返回: xs, ys (numpy 数组)
    保留原有注释与逻辑，但将 G_of_x, F_of_y, H_of_y 封装为闭包使用传入的 f_x/g_y/x0/y0
    '''
    # 初始合法性检查
    if g_y(y0) == 0:
        raise ValueError("错误:初始值 y0 导致 g(y0) == 0，会在分离变量时导致除以零。请更换 y0 或修改 g_y(y)。")

    # 设置 x 网格
    xs = np.linspace(x0, x_end, n_steps + 1)
    ys = np.zeros_like(xs)
    ys[0] = y0

    # --- 封装 G_of_x, F_of_y, H_of_y 为闭包函数，使用传入的 f_x,g_y,x0,y0 ---
    def G_of_x(x):
        '''计算 G(x) = \int_{x0}^{x} f(s) ds'''
        val, err = quad(lambda s: f_x(s), x0, x, limit=quad_limit)
        return val

    def F_of_y(y):
        '''计算 F(y) = \int_{y0}^{y} 1/g(t) dt'''
        def integrand(t):
            gt = g_y(t)
            if gt == 0:
                raise ZeroDivisionError(f"g(y) 在 t={t} 处为零，积分发散或不可直接计算。")
            return 1.0 / gt
        val, err = quad(integrand, y0, y, limit=quad_limit)
        return val

    def H_of_y(y, rhs):
        '''H(y; rhs) = F(y) - rhs'''
        return F_of_y(y) - rhs

    # 预计算 G(x) 对每个 x
    Gs = np.array([G_of_x(x) for x in xs])

    # 为每个 x(从索引1开始)求出 y，使得 F(y) = G(x)
    for i in range(1, len(xs)):
        rhs = Gs[i]  # 这是 F(y) 的目标值

        # 特殊情况:如果 rhs == 0(即 x == x0)，则 y = y0
        if abs(rhs) < 1e-15:
            ys[i] = y0
            continue

        # 动态区间：基于上一步结果的比例区间 [ys[i-1]*0.5, ys[i-1]*1.5]
        search_low = ys[i-1] * 0.5
        search_high = ys[i-1] * 1.5

        N = N_samples
        sample_points = np.linspace(search_low, search_high, N)
        H_vals = []
        finite_flags = []

        # print(f"x={xs[i]:.6g}，rhs={rhs:.1e}")
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
        for j in range(N - 1):
            if not (finite_flags[j] and finite_flags[j + 1]):
                continue
            v1 = H_vals[j]
            v2 = H_vals[j + 1]

            # 如果其中一个点的H(y)接近0，直接视为根
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

        # 如果没有在初始动态区间找到符号变化，尝试以 x0 为基准扩展区间(与原逻辑保持一致)
        if not bracket_found:
            tries = 0
            expanded_low = search_low
            expanded_high = search_high
            while not bracket_found and tries < max_tries:
                expanded_low = x0 + (expanded_low - x0) * expand_factor
                expanded_high = x0 + (expanded_high - x0) * expand_factor
                sample_points = np.linspace(expanded_low, expanded_high, N)
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
                for j in range(N - 1):
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
            raise RuntimeError(
                f"在为 x={xs[i]:.6g} 求解 y 时，无法在 [{search_low}, {search_high}](及扩展区间)中找到使 F(y)-G(x) 有符号变化的区间，无法用 brentq 求根。"
                " 可能原因:g(y) 在区间内有奇点，或解超出搜索区间范围。请扩大搜索区间或检查 g(y)。"
            )

        ys[i] = y_root
        
    # print(ys)
    return xs, ys

# ----------------------- 基准验证模块(RK45) -----------------------
def solve_by_rk45(f_x, g_y, x0, y0, x_end, xs):
    def rhs(x, y):
        return f_x(x) * g_y(y)
    sol = solve_ivp(rhs, (x0, x_end), [y0], method='RK45', t_eval=xs)
    if not sol.success:
        raise RuntimeError("solve_ivp (RK45) 未能成功求解: " + str(sol.message))
    return sol.y[0]

# ----------------------- 结果可视化与误差计算模块 -----------------------
def compute_rmse(a, b):
    '''compute_rmse(a, b)：计算均方根误差'''
    return np.sqrt(np.mean((a - b) ** 2))

def visualize_and_report(xs, ys_sep, ys_rk, example=1, analytic_solution=None, show_plot=True):
    # 计算并打印 RMSE(保留10位小数)
    rmse_val = compute_rmse(ys_sep, ys_rk)
    print(f"分离变量数值解 与 RK45 数值解 的 RMSE = {rmse_val:.10f}")

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys_sep, label="分离变量数值解", lw=2)
        plt.plot(xs, ys_rk, '--', label="RK45 数值解", lw=2)
        if example == 1 and analytic_solution is not None:
            ys_analytic = analytic_solution(xs)
            plt.plot(xs, ys_analytic, ':', label="解析解", lw=2)
            # 额外计算并打印 分离变量数值解 与 解析解 的误差(示例1的要求)
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 确保文件夹存在，不存在则创建
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/Figure_{timestamp}.png")
    else:
        # 即使不显示图形，仍在示例1时输出解析解误差信息（如果 analytic_solution 提供）
        if example == 1 and analytic_solution is not None:
            ys_analytic = analytic_solution(xs)
            abs_err = np.abs(ys_sep - ys_analytic)
            max_err = np.max(abs_err)
            mean_err = np.mean(abs_err)
            print(f"分离变量数值解 与 解析解 的 最大绝对误差 = {max_err:.10f}")
            print(f"分离变量数值解 与 解析解 的 平均绝对误差 = {mean_err:.10f}")

# ----------------------- 对外统一接口 -----------------------
def solve_variable_separable(f_x=None, g_y=None,
                             x0=_default_x0, y0=_default_y0,
                             x_end=_default_x_end, n_steps=_default_n_steps,
                             example=EXAMPLE,
                             N_samples=400, quad_limit=200,
                             expand_factor=2.0, max_tries=3,
                             show_plot=True):
    '''
    模块化对外接口：
    - 如果 f_x/g_y 未提供，则根据 example 选择内置示例函数（1 或 2）
    - 返回一个字典，包含 xs, ys_sep, ys_rk, analytic_solution (若有), runtime_sec
    - 保持原有数值求解逻辑（积分 + 求根）
    '''
    # 选择函数
    if f_x is None or g_y is None:
        if example == 1:
            f_x = _default_f_x_example1
            g_y = _default_g_y_example1
        elif example == 2:
            f_x = _default_f_x_example2
            g_y = _default_g_y_example2
        else:
            raise ValueError("未知的 example 值。仅支持 1 或 2，或请提供自定义 f_x,g_y。")

    start_time = time.time()
    print("开始用分离变量法计算(数值积分 + 根求解)...")
    xs, ys_sep = solve_by_separation(f_x, g_y, x0=x0, y0=y0, x_end=x_end, n_steps=n_steps,
                                     N_samples=N_samples, quad_limit=quad_limit,
                                     expand_factor=expand_factor, max_tries=max_tries)
    print("分离变量法计算完成。")

    print("开始用 solve_ivp (RK45) 作为基准解...")
    ys_rk = solve_by_rk45(f_x, g_y, x0, y0, x_end, xs)
    print("RK45 计算完成。")

    analytic_solution = None
    if example == 1:
        # 解析解:dy/dx = x*y -> ln(y) - ln(y0) = (x^2 - x0^2)/2
        def analytic_sol(x_arr):
            return y0 * np.exp((x_arr ** 2 - x0 ** 2) / 2.0)
        analytic_solution = analytic_sol

    visualize_and_report(xs, ys_sep, ys_rk, example=example, analytic_solution=analytic_solution, show_plot=show_plot)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"总用时: {runtime:.6f} 秒")

    return {
        "xs": xs,
        "ys_sep": ys_sep,
        "ys_rk": ys_rk,
        "analytic_solution": analytic_solution,
        "runtime_sec": runtime
    }

# ----------------------- 主程序(执行流程) -----------------------
def main():
    '''
    main 函数的作用是按顺序执行求解流程，默认展示示例1的求解结果
    保持原有注释与行为：当直接运行此脚本时，执行示例1求解
    '''
    # 如果需要按示例2切换 f_x/g_y，可在此处替换函数定义(演示说明)
    # if EXAMPLE == 2:
    #     global f_x, g_y
    #     f_x = lambda x: np.sin(x); g_y = lambda y: 1 + y**2

    res = solve_variable_separable(example=EXAMPLE,
                                   x0=_default_x0, y0=_default_y0, x_end=_default_x_end,
                                   n_steps=_default_n_steps,
                                   N_samples=400, quad_limit=200,
                                   expand_factor=2.0, max_tries=3,
                                   show_plot=True)
    # 若需要进一步处理 res，可在此处扩展
    # print(res.keys())

if __name__ == "__main__":
    main()