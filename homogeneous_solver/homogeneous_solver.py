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

注意: x0 不能为 0，因为 f(x)=1/x 在 x=0 处有奇点。
"""

from typing import Any, Callable, Tuple
import numpy as np
import math
import argparse
import types
import copy

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


def _demo_example():
    """
    演示示例：求解 dy/dx = g(y/x)，其中 g(u) = u**2（即 dy/dx = (y/x)**2）。
    该函数仅作为用法示例。
    """
    g_expr = "u**2"
    x0 = 1.0
    y0 = 2.0
    x_end = 5.0

    print("Demo: solving dy/dx = g(y/x) with g(u) = {}".format(g_expr))
    print("IC: x0 = {}, y0 = {}, x_end = {}".format(x0, y0, x_end))

    sol = solve_homogeneous(
        g_expr,
        x0=x0,
        y0=y0,
        x_end=x_end,
        n_steps=400,
        show_plot=True,  # forwarded to solve_variable_separable if supported
    )

    print("Returned solution type:", type(sol))
    # 如果返回的是二元组/列表，打印前 5 个点
    if isinstance(sol, (tuple, list)) and len(sol) >= 2:
        xs, ys = sol[0], sol[1]
        print("Sample points (first 5):")
        for xi, yi in zip(xs[:5], np.asarray(ys)[:5]):
            print(f" x={xi:.6g}, y={yi:.6g}")
    elif isinstance(sol, dict):
        xs = sol.get("x") or sol.get("xs")
        ys = sol.get("y") or sol.get("ys")
        if xs is not None and ys is not None:
            print("Sample points (first 5):")
            for xi, yi in zip(xs[:5], np.asarray(ys)[:5]):
                print(f" x={xi:.6g}, y={yi:.6g}")
    else:
        print("Solution returned in an unrecognized structure; inspect manually.")


def main():
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

    sol = solve_homogeneous(
        args.g,
        x0=args.x0,
        y0=args.y0,
        x_end=args.x_end,
        **extra,
    )

if __name__ == "__main__":
    main()