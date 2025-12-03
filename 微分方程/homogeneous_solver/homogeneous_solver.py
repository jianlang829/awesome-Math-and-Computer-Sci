#!/usr/bin/env python3
'''chinese'''
'''python -m homogeneous_solver.homogeneous_solver'''
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

"""
homogeneous_solver.py

求解齐次微分方程形式:
    dy/dx = g(y/x)

使用替换 u = y/x (即 y = u*x) 我们得到:
    x * du/dx = g(u) - u
或
    du/dx = (1/x) * (g(u) - u)

这是可分离变量形式:
    du/dx = f(x) * g_sep(u)
其中
    f(x) = 1/x
    g_sep(u) = g(u) - u

本模块负责进行上述变换并调用 variable_separable_solver.py 中的
 solve_variable_separable 来求解转换后的可分离方程。
求出 u(x) 后，模块负责将解回代为 y(x) = x * u(x)，并保持返回格式与
 solve_variable_separable 函数的输入输出兼容。

对外接口和运行方式：
- solve_homogeneous(g, *, x0, y0, x_end, n_steps=..., **kwargs)
  g 可以是对 u 的可调用函数，也可以是以 'u' 为变量的字符串表达式（可使用 numpy np 和 math）。
- main() 提供一个简单的命令行演示。
- from homogeneous_solver.homogeneous_solver import solve_homogeneous

注意: x0 不能为 0，因为 f(x)=1/x 在 x=0 处有奇点。
依赖库：sympy（需提前 pip install sympy）
"""

from typing import Any, Callable, Tuple
import numpy as np
import math
import argparse
import types
import copy
import os
import time  # 用于程序运行计时
from datetime import datetime
import matplotlib.pyplot as plt  # 用于绘图：解析解与数值解对比
import sympy as sp  # 用于解析解符号推导

# 导入可分离变量求解器 - 假设它与本文件在同一目录或在 PYTHONPATH 中可用。
try:
    from variable_separable_solver.variable_separable_solver import solve_variable_separable
except Exception as e:
    raise ImportError(
        "Could not import solve_variable_separable from variable_separable_solver.py. "
        "Make sure that file is present and on PYTHONPATH."
    ) from e


def _make_callable_g(g) -> Callable[[float], float]:
    """
    确保 g 是一个以 u 为参数的可调用函数。如果 g 是字符串，则构造一个安全的 eval
    可调用对象，用于在变量 'u' 上计算该表达式，并允许使用 numpy (np) 与 math 模块。

    警告: 此处使用了 eval，并去除了内置函数以降低风险 —— 适用于命令行快速使用场景，
    但并非在所有威胁模型下都能提供完全沙箱保护。
    """
    if callable(g):
        return g
    if isinstance(g, str):
        expr = g.strip()
        # 构建安全的全局执行环境（移除内置，允许 np 和 math）
        safe_globals = {"__builtins__": None, "np": np, "math": math}
        # 预编译表达式以便于更好的性能和错误定位
        code = compile(expr, "<g_expr>", "eval")

        def g_func(u):
            # 支持 numpy 数组和标量作为输入
            local = {"u": u}
            return eval(code, safe_globals, local)

        return g_func
    raise TypeError("g must be a callable or a string expression in variable 'u'.")


def _convert_solution_u_to_y(result: Any) -> Any:
    """
    将 solve_variable_separable 返回的 u(x) 解转换为对应的 y(x) = x * u(x)。

    本函数尝试尽可能保留原有返回值的类型与结构：
    - 如果返回值是 (xs, us) 的二元组/列表，则返回 (xs, ys)。
    - 如果返回值是字典并包含 'x'/'y' 或 'xs'/'ys' 键，则进行相应转换。
    - 如果返回值是具有属性 'x'/'y' 或 'xs'/'ys' 的对象，尝试浅拷贝并替换属性。
    - 否则返回一个包含 'x' 与 'y' 的字典（仅当能识别原结构时）。

    注意: 我们假定在二元结构中索引 0 是 x 序列，索引 1 是 u 序列。
    """
    # 辅助函数：按元素相乘，兼容 list/tuple/ndarray
    def times_x(xs, us):
        xs_arr = np.asarray(xs)
        us_arr = np.asarray(us)
        # 数组长度一致性检查，避免越界
        if xs_arr.shape != us_arr.shape:
            # 尝试广播，若失败则抛出异常
            try:
                ys_arr = xs_arr * us_arr
            except Exception:
                raise ValueError("xs 与 us 形状不匹配，无法逐元素相乘")
        else:
            ys_arr = xs_arr * us_arr
        # 若原始 u 为 list，则返回 list；否则返回 ndarray
        if isinstance(us, list):
            return ys_arr.tolist()
        return ys_arr

    # 情形 1: tuple 或 list 且长度 >= 2，假定 index0 = xs, index1 = us
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        xs = result[0]
        us = result[1]
        ys = times_x(xs, us)
        # 保持返回类型（tuple 或 list）
        out = list(result)
        out[1] = ys
        return tuple(out) if isinstance(result, tuple) else out

    # 情形 2: dict 类型
    if isinstance(result, dict):
        res = dict(result)  # 浅拷贝以避免改动原字典
        # 常见键对转换 ('x','y') 或 ('xs','ys')
        if "x" in res and "y" in res:
            # 注意：此处 'y' 实际上存储的是 u 值，需要转换为 y
            res["y"] = times_x(res["x"], res["y"])
            return res
        if "xs" in res and "ys_sep" in res:
            res["ys"] = times_x(res["xs"], res["ys_sep"])
            return res
        # 兜底策略：寻找 'x' 或 'xs'，并把第一个非 x 的数组式键作为依赖变量转换
        xs_key = None
        for k in ("x", "xs"):
            if k in res:
                xs_key = k
                break
        if xs_key is not None:
            # 查找第一个不是 xs_key 的数组样值键并尝试转换
            for k, v in res.items():
                if k == xs_key:
                    continue
                try:
                    res[k] = times_x(res[xs_key], v)
                    return res
                except Exception:
                    pass
        # 无法确定要转换的键，返回浅拷贝（保持原样）
        return res

    # 情形 3: 对象（具有 __dict__ 或 __slots__）
    if hasattr(result, "__dict__") or hasattr(result, "__slots__"):
        # 尝试浅拷贝对象，避免修改原对象
        try:
            out_obj = copy.copy(result)
        except Exception:
            out_obj = result  # 无法拷贝则直接在原对象上操作（作为最后手段）

        # 尝试 ('x','y') 或 ('xs','ys') 属性对
        if hasattr(out_obj, "x") and hasattr(out_obj, "y"):
            xs = getattr(out_obj, "x")
            us = getattr(out_obj, "y")
            try:
                setattr(out_obj, "y", times_x(xs, us))
                return out_obj
            except Exception:
                pass
        if hasattr(out_obj, "xs") and hasattr(out_obj, "ys"):
            xs = getattr(out_obj, "xs")
            us = getattr(out_obj, "ys")
            try:
                setattr(out_obj, "ys", times_x(xs, us))
                return out_obj
            except Exception:
                pass
        # 无法安全转换，返回对象本身（未修改）
        return out_obj

    # 无法识别的返回类型
    raise TypeError(
        "solve_variable_separable returned an unsupported result type; "
        "expected tuple/list (xs, ys), dict, or object with x/y attributes."
    )


# ---------- 解析解模块：尝试使用符号计算推导闭式解析解 ----------
def analytical_solution(params: dict):
    """
    尝试为 dy/dx = g(y/x)（通过 u=y/x 转换为 du/dx = (1/x)*(g(u)-u)）
    求出解析解 y(x) 在给定 x 范围上的闭式表达式或数值评估结果。

    优先级：
    1) 若 params['g'] 为字符串表达式，则尝试用 sympy 符号积分并解析求解 u(x)；
    2) 否则返回 None（视为无显式解析解，保留数值流程不变）。

    输入 params 要求（尽可能完整）：
    - 'g': 字符串或可调用（如果是可调用，则不会尝试符号求解）
    - 'x0': 初始 x（必须非零）
    - 'y0': 初始 y
    - 可选 'xs': 要评估的 x 网格（若提供，优先使用；否则将根据 x0/x_end/n_steps 生成）
    - 可选 'x_end', 'n_steps'

    返回：
    - 若成功：字典 {'xs': xs_array, 'ys': ys_array, 'u_expr': sympy_expr_u}，其中 ys = x * u(x)
    - 若失败或无法求闭式解：返回 None
    """
    # 参数检查
    if not isinstance(params, dict):
        return None
    g = params.get("g")
    x0 = params.get("x0")
    y0 = params.get("y0")
    if x0 is None or y0 is None or g is None:
        return None

    # 仅当 g 为字符串时尝试符号求解；否则视为无解析解
    if not isinstance(g, str):
        return None

    # 保护性检查：x0 不能为 0（变换处奇点）
    if x0 == 0:
        return None

    # 准备要评估的 x 网格
    xs = params.get("xs")
    if xs is not None:
        xs_arr = np.asarray(xs)
    else:
        x_end = params.get("x_end", x0 + 1.0)
        n_steps = int(params.get("n_steps", 400))
        if n_steps <= 0:
            n_steps = 400
        # 步长取值：在不改变原逻辑的前提下，兼顾精度和效率
        xs_arr = np.linspace(x0, x_end, n_steps)

    # 避免包含 0，因右边含 ln|x|
    if np.any(xs_arr == 0):
        return None

    # 符号推导：求 ∫ du/(g(u)-u) = ln|x| + C，尝试求 u(x)
    try:
        u = sp.symbols("u")
        x = sp.symbols("x")
        # 将字符串表达式转换为 sympy 表达式
        # 允许常见的 numpy/ math 风格函数名需要用户在传入时使用 sympy 兼容表达式
        g_expr = sp.sympify(g, locals={})
        # 被积函数 1/(g(u)-u)
        integrand = 1 / (g_expr - u)
        F_u = sp.integrate(integrand, (u,))
        # 若积分返回含有 Integral 对象，认为无法求出显式原函数
        if isinstance(F_u, sp.Integral) or any(isinstance(a, sp.Integral) for a in sp.preorder_traversal(F_u)):
            return None

        # 初值 u0
        u0 = sp.Rational(0)
        try:
            u0_val = sp.Rational(y0) / sp.Rational(x0)
            u0 = u0_val
        except Exception:
            # 若不能直接转换为 Rational，则使用浮点
            u0 = sp.N(y0 / x0)

        # 常数 C 的符号表达
        # 使用 abs/符号，从数值上计算 C
        # 注意：对于 x0 > 0 与 x0 < 0 的 ln 处理使用 ln(abs(x0))
        C = F_u.subs(u, u0) - sp.log(sp.Abs(x0))

        # 构造方程 F(u) - ln|x| - C = 0，尝试求解 u（关于 x 的显式表达式）
        eq = sp.Eq(F_u - sp.log(sp.Abs(x)) - C, 0)
        sols = sp.solve(eq, u)
        if not sols:
            return None

        # 取第一个解，尝试 lambdify 为数值函数 u(x)
        u_solution = sols[0]
        # 若解中包含 u（未真正消去），认为不完全求解
        if u_solution.free_symbols and u in u_solution.free_symbols:
            return None

        # 将结果转为可数值评估的函数
        try:
            u_of_x_func = sp.lambdify(x, u_solution, modules=["numpy", "math"])
        except Exception:
            return None

        # 评估 u(x) 并计算 y = x * u(x)
        xs_numeric = np.asarray(xs_arr, dtype=float)
        # 域检查：避免 x 取负数导致 log/abs 等问题，根据解析表达式可能需要限定正域
        # 这里仅在评估时捕获异常并在失败时返回 None
        try:
            u_vals = u_of_x_func(xs_numeric)
            u_vals_arr = np.asarray(u_vals, dtype=float)
            ys_numeric = xs_numeric * u_vals_arr
        except Exception:
            return None

        return {"xs": xs_numeric, "ys": ys_numeric, "u_expr": u_solution}

    except Exception:
        # 任意异常视为无法求闭式解析解
        return None


# ---------- 可视化模块：绘制解析解与数值解对比（仅在存在解析解时调用） ----------
def _extract_xs_ys_from_result(result: Any):
    """
    从 solve_homogeneous/solve_variable_separable 的返回值中提取 (xs, ys)。
    支持 tuple/list (xs, ys), dict {'x'/'xs':..., 'y'/'ys':...}, 或对象属性。
    返回 (xs_array, ys_array) 或抛出 TypeError。
    """
    # 先处理 tuple/list
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        xs = np.asarray(result[0], dtype=float)
        ys = np.asarray(result[1], dtype=float)
        return xs, ys

    if isinstance(result, dict):
        # 尝试常见键
        xs = None
        ys = None
        # 作者给自己的提示：把xs，ys转成浮点数
        if "x" in result:
            xs = np.asarray(result["x"], dtype=float)
        elif "xs" in result:
            xs = np.asarray(result["xs"], dtype=float)
        if "y" in result:
            ys = np.asarray(result["y"], dtype=float)
        elif "ys" in result:
            ys = np.asarray(result["ys"], dtype=float)
        # 有时求解器把解析前的 u 存为 ys_sep 或 y（而我们已经在 solve_homogeneous 转换）
        if xs is not None and ys is not None:
            return xs, ys
        # 兜底：尝试第一对可转换的数组
        for k1, v1 in result.items():
            for k2, v2 in result.items():
                if k1 == k2:
                    continue
                try:
                    a1 = np.asarray(v1, dtype=float)
                    a2 = np.asarray(v2, dtype=float)
                    if a1.shape == a2.shape:
                        return a1, a2
                except Exception:
                    continue
        raise TypeError("无法从字典结果中提取 xs, ys")
    # 对象属性情况
    if hasattr(result, "x") and hasattr(result, "y"):
        xs = np.asarray(getattr(result, "x"), dtype=float)
        ys = np.asarray(getattr(result, "y"), dtype=float)
        return xs, ys
    if hasattr(result, "xs") and hasattr(result, "ys"):
        xs = np.asarray(getattr(result, "xs"), dtype=float)
        ys = np.asarray(getattr(result, "ys"), dtype=float)
        return xs, ys

    raise TypeError("无法识别的数值解结构以提取 xs, ys")


def plot_analytical_vs_numerical(numerical_results: Any, params: dict, save_path: str):
    """
    在解析解存在时，绘制解析解与数值解对比图并保存为文件（禁止 plt.show()）。

    参数说明：
    - numerical_results: solve_homogeneous 返回值（任意支持的结构）
    - params: 传递给 analytical_solution 的参数字典
    - save_path: 图片保存路径，例如 "plots/comparison_plot.png"
    """
    analytical_res = analytical_solution(params)
    if analytical_res is None:
        # 明确表示无解析解，不进行任何可视化操作（严格要求）
        return None

    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 提取数值解的 xs, ys（若无法提取则抛出）
    xs_num, ys_num = _extract_xs_ys_from_result(numerical_results)

    # 解析解提供的 xs, ys，如果解析解未使用与数值相同的 xs，则重采样解析解到数值网格
    xs_ana = analytical_res.get("xs")
    ys_ana = analytical_res.get("ys")
    if xs_ana is None or ys_ana is None:
        return None

    # 如果解析解的 xs 与数值解的 xs 形状不同，尝试在数值解 xs 上评估解析表达式（若可用）
    if xs_ana.shape != xs_num.shape:
        # 如果 analytical_res 含有 'u_expr'（sympy 表达式），尝试用它来在数值 xs 上重建解析解
        u_expr = analytical_res.get("u_expr")
        if u_expr is not None:
            try:
                x = sp.symbols("x")
                u_func = sp.lambdify(x, u_expr, modules=["numpy", "math"])
                u_vals_on_num = u_func(xs_num)
                ys_ana_on_num = xs_num * np.asarray(u_vals_on_num, dtype=float)
                xs_plot = xs_num
                ys_plot = ys_ana_on_num
            except Exception:
                # 若无法按数值网格重建，退回到解析解自带的网格
                xs_plot = xs_ana
                ys_plot = ys_ana
        else:
            xs_plot = xs_ana
            ys_plot = ys_ana
    else:
        xs_plot = xs_ana
        ys_plot = ys_ana

    # ---------- 可视化：绘制折线对比 ----------
    plt.figure()
    plt.plot(xs_num, ys_num, label="数值解", linestyle="-", linewidth=1.5)
    plt.plot(xs_plot, ys_plot, label="解析解", linestyle="--", linewidth=1.2)
    # 横纵轴标签：尽可能使用原代码中的变量物理意义。这里原问题表示 y 为函数值，x 为自变量
    plt.xlabel("自变量 x")
    plt.ylabel("函数值 y")
    plt.title("解析解与数值解对比")
    plt.legend()
    # 保存并关闭（禁止 plt.show()）
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def solve_homogeneous(
    g,
    *,
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int = None,
    # capture arbitrary kwargs to forward to solve_variable_separable
    **kwargs,
):
    '''对外调用接口'''
    """
    在区间 [x0, x_end] 上求解 dy/dx = g(y/x)，给定初值 y(x0)=y0。

    参数说明：
    - g: 可调用对象 g(u) 或以字符串形式给出的表达式（变量名为 'u'，可使用 np 与 math）。
    - x0, y0: 初值（注意 x0 不能为 0）。
    - x_end: x 的终点。
    - n_steps: 可选，转发给 solve_variable_separable 的步数参数。
    - kwargs: 其他关键字参数会直接转发给 solve_variable_separable。

    返回值：
    - 与 solve_variable_separable 相同的返回格式，但其依赖变量从 u(x) 被转换为 y(x)=x*u(x)。
    """
    # 校验 x0 非零（因为变换依赖 1/x）
    if x0 == 0:
        raise ValueError("x0 must be non-zero for the substitution u = y/x (f(x)=1/x singular at 0).")

    # 确保 g 是可调用的函数
    g_callable = _make_callable_g(g)

    # 定义转换后的可分离方程：
    # du/dx = (1/x) * (g(u) - u)
    def f_x(x):
        # 分母检查，避免除以零
        if x == 0:
            raise ValueError("分母为零，x=0 在此处不可接受")
        return 1.0 / x

    def g_sep(u):
        return g_callable(u) - u

    # 求初始 u0
    u0 = y0 / x0

    # 准备传入求解器的参数（浅拷贝）
    solver_kwargs = dict(kwargs)  # 浅拷贝
    if n_steps is not None:
        solver_kwargs["n_steps"] = n_steps

    # 调用可分离变量求解器，求解 u(x)
    result_u = solve_variable_separable(
        f_x=f_x,
        g_y=g_sep,
        x0=x0,
        y0=u0,
        x_end=x_end,
        **solver_kwargs,
    )

    # 将 u(x) 的解转换回 y(x) = x * u(x)
    result_y = _convert_solution_u_to_y(result_u)
    return result_y

def main():
    start_time = time.time()  # 程序计时开始
    parser = argparse.ArgumentParser(
        description="Solve homogeneous ODE dy/dx = g(y/x). "
                    "g can be a Python expression in variable 'u' using numpy as np and math."
    )
    parser.add_argument("--g", type=str, default="u**2",
                        help="Expression for g(u) (default: 'u**2'). Use 'np' for numpy functions.")
    parser.add_argument("--x0", type=float, default=1.0, help="Initial x (must be non-zero).")
    parser.add_argument("--y0", type=float, default=2.0, help="Initial y.")
    parser.add_argument("--x_end", type=float, default=1.9, help="End x.")
    parser.add_argument("--n_steps", type=int, default=400, help="Number of steps (forwarded).")
    parser.add_argument("--no_plot", action="store_true", help="If set, do not ask solver to show plot.")
    args = parser.parse_args()

    extra = {"n_steps": args.n_steps}

    # 可以生成对比图像，生成图像为u和x的图像，详见variable_separable_solver.variable_separable_solver文件
    if args.no_plot:
        extra["show_plot"] = False
    else:
        extra["show_plot"] = True

    # 使用 try/finally 确保无论发生什么都能记录结束时间
    try:
        sol = solve_homogeneous(
            args.g,
            x0=args.x0,
            y0=args.y0,
            x_end=args.x_end,
            **extra,
        )

        # ---------- 解析解存在性判断与可视化调用 ----------
        # 作者给自己的提示：这里是在处理解析解
        # 准备传入 analytical_solution 的参数
        params = {"g": args.g, "x0": args.x0, "y0": args.y0, "x_end": args.x_end, "n_steps": args.n_steps}
        # 尝试从数值解中提取 xs，用于解析解评估（若 possible）
        try:
            xs_num, _ = _extract_xs_ys_from_result(sol)
            params["xs"] = xs_num
        except Exception:
            # 若无法提取数值网格则不提供 xs，由 analytical_solution 自行生成
            pass

        # 仅当 analytical_solution 返回非 None 时才进行绘图
        analytical_res = analytical_solution(params)
        if analytical_res is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("plots", f"Figure_{timestamp}.png")
            try:
                plot_analytical_vs_numerical(sol, params, save_path)
            except Exception as e:
                # 可视化失败但不影响主流程
                print("解析解绘图失败：", e)

    finally:
        end_time = time.time()
        print(f"程序总运行时间：{end_time - start_time:.4f}秒")


if __name__ == "__main__":
    main()