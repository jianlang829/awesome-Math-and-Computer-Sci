#!/usr/bin/env python3
'''chinese'''
"""
linear_combination_solver.py

求解形如:
    dy/dx = g(a*x + b*y + c)

采用变量替换 u = a*x + b*y + c，将原方程转化为关于 u 的可分离变量方程：
    du/dx = a + b * g(u)

代码在原有基础上做了若干改进：
- analytical_solution 对字符串 g 的处理更健壮：支持 np.sin / math.sin 等风格（会尝试将 np. 或 math. 前缀去掉并把常见函数名映射为 sympy 函数）。
  在解析失败时会尽量记录/warn 有用的错误信息并返回 None（不抛出未捕获异常）。
- _convert_solution_u_to_y 被重构为更清晰且更严格的转换逻辑：优先处理标准 tuple/list (xs, ys) 和标准 dict {'x':..,'y':..} 或 {'xs':..,'ys':..}，
  不再使用过于宽泛的兜底循环以避免错误转换。
- 细化了 try/except，尽量将异常范围限制在必要的语句块内，并使用 warnings.warn 提供诊断信息。

依赖：numpy, sympy, matplotlib（可选）, variable_separable_solver 包
"""
from typing import Any, Callable, Tuple
import numpy as np
import math
import argparse
import copy
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import sympy as sp
import warnings
import re

# 导入可分离变量求解器
try:
    from variable_separable_solver.variable_separable_solver import solve_variable_separable
except Exception as e:
    raise ImportError(
        "Could not import solve_variable_separable from variable_separable_solver. "
        "Ensure that variable_separable_solver package/module is available on PYTHONPATH."
    ) from e


def _make_callable_g(g) -> Callable[[float], float]:
    """
    确保 g 是一个以 u 为参数的可调用函数。
    支持：
      - 传入可调用对象（直接返回）
      - 传入字符串表达式（变量名为 'u'），允许使用 np, math

    与 homogeneous_solver._make_callable_g 保持一致的安全/风格处理。
    """
    if callable(g):
        return g
    if isinstance(g, str):
        expr = g.strip()
        safe_globals = {"__builtins__": None, "np": np, "math": math}
        code = compile(expr, "<g_expr>", "eval")

        def g_func(u):
            local = {"u": u}
            return eval(code, safe_globals, local)

        return g_func
    raise TypeError("g must be a callable or a string expression in variable 'u'.")


def _convert_solution_u_to_y(result: Any, a: float, b: float, c: float) -> Any:
    """
    将 solve_variable_separable 返回的 u(x) 解转换为对应的 y(x)：
        y(x) = (u(x) - a*x - c) / b

    仅支持严格且常见的返回格式：
      - tuple/list: (xs, us, ...) 假定索引0为 xs, 索引1为 u 值 -> 返回相同类型，index1替换为 y 值
      - dict: 至少包含 'x' 和 'y' 键（其中 'y' 为 u 值）或 'xs' 和 'ys'（其中 'ys' 为 u 值）
    其它类型将抛出 TypeError，以避免错误的盲目转换。
    """
    if b == 0:
        raise ValueError("b == 0: no conversion from u to y is necessary/valid")

    def compute_y_from_xy(xs, us):
        xs_arr = np.asarray(xs, dtype=float)
        us_arr = np.asarray(us, dtype=float)
        # 支持广播，但期望 xs 和 us 在可广播的意义下能得到长度相同的结果
        try:
            ys_arr = (us_arr - a * xs_arr - c) / b
        except Exception as exc:
            raise ValueError(f"无法将 xs 与 us 转换为 y：{exc}") from exc
        # 返回与 us 相同的容器类型（list -> list，否则 ndarray）
        if isinstance(us, list):
            return ys_arr.tolist()
        return ys_arr

    # 处理 tuple 或 list，假定 index0 = xs, index1 = us
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        xs = result[0]
        us = result[1]
        ys = compute_y_from_xy(xs, us)
        out = list(result)
        out[1] = ys
        return tuple(out) if isinstance(result, tuple) else out

    # 处理 dict，优先识别标准键对
    if isinstance(result, dict):
        res = dict(result)  # 浅拷贝
        if "x" in res and "y" in res:
            res["y"] = compute_y_from_xy(res["x"], res["y"])
            return res
        if "xs" in res and "ys" in res:
            res["ys"] = compute_y_from_xy(res["xs"], res["ys"])
            return res
        # 若无法识别标准键对，则不给出猜测，明确抛出错误（避免误转换）
        raise TypeError(
            "Unsupported dict result format for conversion to y(x). Expected keys ('x','y') or ('xs','ys')."
        )

    # 对象类型：仅在明确含有 xs/ys 或 x/y 属性时处理
    if hasattr(result, "__dict__") or hasattr(result, "__slots__"):
        try:
            out_obj = copy.copy(result)
        except Exception:
            out_obj = result  # 如果无法拷贝则在原对象上操作（尽量避免）
        if hasattr(out_obj, "x") and hasattr(out_obj, "y"):
            xs = getattr(out_obj, "x")
            us = getattr(out_obj, "y")
            ys = compute_y_from_xy(xs, us)
            try:
                setattr(out_obj, "y", ys)
            except Exception:
                raise TypeError("Unable to set 'y' attribute on result object during conversion.")
            return out_obj
        if hasattr(out_obj, "xs") and hasattr(out_obj, "ys"):
            xs = getattr(out_obj, "xs")
            us = getattr(out_obj, "ys")
            ys = compute_y_from_xy(xs, us)
            try:
                setattr(out_obj, "ys", ys)
            except Exception:
                raise TypeError("Unable to set 'ys' attribute on result object during conversion.")
            return out_obj
        raise TypeError("Unsupported object result format for conversion to y(x). Expected attributes x/y or xs/ys.")

    raise TypeError("Unsupported result type from solver for conversion to y(x).")


# ---------- 解析解模块：增强对 g 字符串的处理与错误信息 ----------
def _prepare_sympy_expression_from_string(g_str: str):
    """
    将用户传入的 g 字符串预处理为 sympy 可识别的表达式：
    - 将 'np.' 与 'math.' 前缀移除（例如 'np.sin(u)' -> 'sin(u)')；
    - 为常见函数名提供映射，比如 sin, cos, exp, log, sqrt, Abs 等；
    - 返回 (sympy_expr, None) 或 (None, error_message) 以便上层决定如何处理。
    """
    if not isinstance(g_str, str):
        return None, "g is not a string"

    s = g_str.strip()

    # 移除常见前缀 'np.' 与 'math.'（把 np.sin(u) -> sin(u)）
    s_no_prefix = re.sub(r'\b(np|math)\.', '', s)

    # 建立 sympy locals 映射
    sym_locals = {
        # 三角
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        # 指数与对数
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        # 双曲
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        # 其他
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }

    # 尝试 sympify
    try:
        expr = sp.sympify(s_no_prefix, locals=sym_locals)
        return expr, None
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        return None, f"sympify failed: {e}"
    except Exception as e:
        # 捕获其它潜在错误并返回信息
        return None, f"unexpected error during sympify: {e}"


def analytical_solution(params: dict):
    """
    尝试对 dy/dx = g(a*x + b*y + c) 做符号求解（当 g 为字符串时尝试）。
    改进点：
    - 更智能地处理 numpy/math 风格的字符串表达式（尝试去除 np./math. 前缀并映射常见函数）
    - 更细致地捕获 sympy 的异常，并在失败时通过 warnings.warn 给出诊断信息，然后返回 None

    返回：
      - 成功时返回 dict {'xs': xs_array, 'ys': ys_array, 'u_expr' or 'y_expr': sympy_expr}
      - 失败时返回 None（并通过 warnings.warn 提供失败原因）
    """
    if not isinstance(params, dict):
        return None
    required = ("g", "a", "b", "c", "x0", "y0")
    if not all(k in params for k in required):
        return None

    g = params["g"]
    a = params["a"]
    b = params["b"]
    c = params["c"]
    x0 = params["x0"]
    y0 = params["y0"]

    # 生成 xs 网格
    xs = params.get("xs")
    if xs is not None:
        xs_arr = np.asarray(xs, dtype=float)
    else:
        x_end = params.get("x_end", x0 + 1.0)
        n_steps = int(params.get("n_steps", 400))
        if n_steps <= 0:
            n_steps = 400
        xs_arr = np.linspace(x0, x_end, n_steps)

    # 仅在 g 为字符串时尝试符号推导
    if not isinstance(g, str):
        return None

    # 预处理 g 字符串以生成 sympy 表达式
    g_expr, err = _prepare_sympy_expression_from_string(g)
    if g_expr is None:
        warnings.warn(f"analytical_solution: cannot parse g expression for symbolic work: {err}")
        return None

    # 特殊情形 b == 0：u = a*x + c，dy/dx = g(a*x + c)
    try:
        if b == 0:
            # (1) 如果 a == 0，dy/dx = g(c) 为常数
            if a == 0:
                try:
                    g_c_sym = g_expr.subs(sp.symbols('u'), c)
                    g_c_val = float(sp.N(g_c_sym))
                    ys = g_c_val * xs_arr + (y0 - g_c_val * x0)
                    return {"xs": xs_arr, "ys": ys, "y_expr": sp.simplify(g_c_sym * sp.Symbol("x") + (y0 - g_c_val * x0))}
                except Exception:
                    # 如果无法数值化 g(c)，尝试构造符号表达式并数值评估
                    try:
                        C = sp.simplify(y0 - g_expr.subs(sp.symbols('u'), c) * x0)
                        y_expr = g_expr.subs(sp.symbols('u'), c) * sp.Symbol("x") + C
                        y_func = sp.lambdify(sp.symbols("x"), y_expr, modules=["numpy", "math"])
                        ys = np.asarray(y_func(xs_arr), dtype=float)
                        return {"xs": xs_arr, "ys": ys, "y_expr": sp.simplify(y_expr)}
                    except Exception as e:
                        warnings.warn(f"analytical_solution b==0 a==0: failed to form symbolic constant solution: {e}")
                        return None

            # (2) a != 0: y = (1/a) * ∫ g(t) dt with t = a*x + c
            t = sp.symbols("t")
            try:
                G_t = sp.integrate(g_expr, (t,))
            except Exception as e:
                warnings.warn(f"analytical_solution b==0: integration failed for G(t) = ∫ g(t) dt: {e}")
                return None

            if isinstance(G_t, sp.Integral) or any(isinstance(e, sp.Integral) for e in sp.preorder_traversal(G_t)):
                warnings.warn("analytical_solution b==0: sympy returned an unevaluated Integral for ∫g(t)dt")
                return None

            try:
                G_at_x0 = sp.N(G_t.subs(t, a * x0 + c))
                C = sp.N(y0 - (G_at_x0 / a))
                y_expr = sp.simplify(G_t.subs(t, a * sp.symbols("x") + c) / a + C)
                y_func = sp.lambdify(sp.symbols("x"), y_expr, modules=["numpy", "math"])
                ys = np.asarray(y_func(xs_arr), dtype=float)
                return {"xs": xs_arr, "ys": ys, "y_expr": y_expr}
            except Exception as e:
                warnings.warn(f"analytical_solution b==0: failed to evaluate y_expr numerically: {e}")
                return None

        # b != 0: 积分 1 / (a + b*g(u)) = x + C
        u = sp.symbols("u")
        integrand = 1 / (a + b * g_expr)
        try:
            F_u = sp.integrate(integrand, (u,))
        except Exception as e:
            warnings.warn(f"analytical_solution: integration of 1/(a+b*g(u)) failed: {e}")
            return None

        if isinstance(F_u, sp.Integral) or any(isinstance(e, sp.Integral) for e in sp.preorder_traversal(F_u)):
            warnings.warn("analytical_solution: sympy returned an unevaluated Integral for ∫du/(a+b*g(u))")
            return None

        # 初值 u0 = a*x0 + b*y0 + c
        try:
            u0_val = sp.simplify(a * x0 + b * y0 + c)
        except Exception:
            u0_val = sp.N(a * x0 + b * y0 + c)

        # 常数 C = F(u0) - x0
        try:
            C = sp.simplify(F_u.subs(u, u0_val) - x0)
        except Exception:
            try:
                C = sp.N(F_u.subs(u, u0_val) - x0)
            except Exception as e:
                warnings.warn(f"analytical_solution: failed to compute integration constant C: {e}")
                return None

        # 构造方程 F(u) - x - C = 0 并求解 u(x)
        x = sp.symbols("x")
        eq = sp.Eq(F_u - x - C, 0)
        try:
            sols = sp.solve(eq, u)
        except Exception as e:
            warnings.warn(f"analytical_solution: sp.solve raised an exception: {e}")
            return None

        if not sols:
            warnings.warn("analytical_solution: sp.solve returned no solutions for u(x)")
            return None

        # 取第一个可用解。若解含有 u 或其他未消去的符号则放弃。
        u_solution = sols[0]
        if u in u_solution.free_symbols:
            warnings.warn("analytical_solution: obtained u(x) still depends on u symbol; giving up")
            return None

        # 将 u(x) lambdify 并回代为 y(x)
        try:
            u_of_x_func = sp.lambdify(x, u_solution, modules=["numpy", "math"])
        except Exception as e:
            warnings.warn(f"analytical_solution: lambdify failed for u(x): {e}")
            return None

        xs_numeric = np.asarray(xs_arr, dtype=float)
        try:
            u_vals = u_of_x_func(xs_numeric)
            u_vals_arr = np.asarray(u_vals, dtype=float)
            ys_numeric = (u_vals_arr - a * xs_numeric - c) / b
        except Exception as e:
            warnings.warn(f"analytical_solution: evaluating u(x) numerically failed: {e}")
            return None

        return {"xs": xs_numeric, "ys": ys_numeric, "u_expr": sp.simplify(u_solution)}

    except Exception as e:
        # 捕捉任意未预见异常并以 warn 通知，避免抛出到上层
        warnings.warn(f"analytical_solution: unexpected failure: {e}")
        return None


# ---------- 可视化：解析解与数值解对比 ----------
def _extract_xs_ys_from_result(result: Any):
    """
    与 homogeneous_solver._extract_xs_ys_from_result 类似，用于提取数值解 (xs, ys)。
    支持 tuple/list, dict, 对象属性。
    """
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        xs = np.asarray(result[0], dtype=float)
        ys = np.asarray(result[1], dtype=float)
        return xs, ys

    if isinstance(result, dict):
        xs = None
        ys = None
        if "x" in result:
            xs = np.asarray(result["x"], dtype=float)
        elif "xs" in result:
            xs = np.asarray(result["xs"], dtype=float)
        if "y" in result:
            ys = np.asarray(result["y"], dtype=float)
        elif "ys" in result:
            ys = np.asarray(result["ys"], dtype=float)
        if xs is not None and ys is not None:
            return xs, ys
        raise TypeError("Cannot extract xs, ys from dict result: expected keys ('x','y') or ('xs','ys')")

    if hasattr(result, "x") and hasattr(result, "y"):
        xs = np.asarray(getattr(result, "x"), dtype=float)
        ys = np.asarray(getattr(result, "y"), dtype=float)
        return xs, ys
    if hasattr(result, "xs") and hasattr(result, "ys"):
        xs = np.asarray(getattr(result, "xs"), dtype=float)
        ys = np.asarray(getattr(result, "ys"), dtype=float)
        return xs, ys

    raise TypeError("Unsupported numerical result structure to extract xs, ys")


def plot_analytical_vs_numerical(numerical_results: Any, params: dict, save_path: str):
    """
    若存在解析解（analytical_solution 返回非 None），绘制数值解与解析解比较图并保存。
    """
    analytical_res = analytical_solution(params)
    if analytical_res is None:
        return None

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    xs_num, ys_num = _extract_xs_ys_from_result(numerical_results)

    xs_ana = analytical_res.get("xs")
    ys_ana = analytical_res.get("ys")
    if xs_ana is None or ys_ana is None:
        return None

    # 若解析解网格与数值解网格不同，尝试用解析表达式在数值 xs 上重建解析值
    if xs_ana.shape != xs_num.shape:
        # 优先用 y_expr（若存在），否则用 u_expr 转换
        y_expr = analytical_res.get("y_expr")
        u_expr = analytical_res.get("u_expr")
        if y_expr is not None:
            try:
                x = sp.symbols("x")
                y_func = sp.lambdify(x, y_expr, modules=["numpy", "math"])
                ys_ana_on_num = np.asarray(y_func(xs_num), dtype=float)
                xs_plot = xs_num
                ys_plot = ys_ana_on_num
            except Exception:
                xs_plot = xs_ana
                ys_plot = ys_ana
        elif u_expr is not None:
            try:
                x = sp.symbols("x")
                u_func = sp.lambdify(x, u_expr, modules=["numpy", "math"])
                u_vals_on_num = np.asarray(u_func(xs_num), dtype=float)
                ys_ana_on_num = (u_vals_on_num - params["a"] * xs_num - params["c"]) / params["b"]
                xs_plot = xs_num
                ys_plot = ys_ana_on_num
            except Exception:
                xs_plot = xs_ana
                ys_plot = ys_ana
        else:
            xs_plot = xs_ana
            ys_plot = ys_ana
    else:
        xs_plot = xs_ana
        ys_plot = ys_ana

    plt.figure()
    plt.plot(xs_num, ys_num, label="数值解", linestyle="-", linewidth=1.5)
    plt.plot(xs_plot, ys_plot, label="解析解", linestyle="--", linewidth=1.2)
    plt.xlabel("自变量 x")
    plt.ylabel("函数值 y")
    plt.title("解析解与数值解对比")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


# ---------- 对外接口 ----------
def solve_linear_combination(
    g,
    a: float,
    b: float,
    c: float,
    *,
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int = None,
    **kwargs,
):
    """
    外部接口：在区间 [x0, x_end] 上求解初值问题 dy/dx = g(a*x + b*y + c)，y(x0)=y0。

    参数：
      - g: 可调用或以 'u' 为变量的字符串表达式（允许 np, math）
      - a, b, c: 常数
      - x0, y0, x_end: 数值
      - n_steps: 转发给 solve_variable_separable 的步数
      - kwargs: 其他关键字参数转发给 solve_variable_separable

    返回：
      - 与 solve_variable_separable 相同格式的结果，但当 b != 0 时将依赖变量从 u 转换为 y。
      - 当 b == 0 时直接返回求得的 y(x)（由可分离求解器计算）
    """
    # 参数检查
    g_callable = _make_callable_g(g)

    solver_kwargs = dict(kwargs)
    if n_steps is not None:
        solver_kwargs["n_steps"] = n_steps

    # b == 0 的特殊路径：dy/dx = g(a*x + c) -> dy/dx = h(x)
    if b == 0:
        def f_x(x):
            # 对可能的异常在调用 solve_variable_separable 时捕获
            return float(g_callable(a * x + c))

        def g_y(y):
            # dy/dx = f_x(x) * 1
            return 1.0

        result_y = solve_variable_separable(
            f_x=f_x,
            g_y=g_y,
            x0=x0,
            y0=y0,
            x_end=x_end,
            **solver_kwargs,
        )
        return result_y

    # b != 0 的常规路径：转 u = a*x + b*y + c
    def f_x_unit(x):
        return 1.0

    def g_u(u):
        # 可能抛出异常（例如用户 g 返回非数值），让调用方的求解器负责处理
        return float(a + b * g_callable(u))

    u0 = a * x0 + b * y0 + c

    result_u = solve_variable_separable(
        f_x=f_x_unit,
        g_y=g_u,
        x0=x0,
        y0=u0,
        x_end=x_end,
        **solver_kwargs,
    )

    # 将 u(x) 转换回 y(x)
    result_y = _convert_solution_u_to_y(result_u, a=a, b=b, c=c)
    return result_y


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Solve dy/dx = g(a*x + b*y + c). g can be an expression in variable 'u' using numpy as np and math."
    )
    parser.add_argument("--g", type=str, default="u**2", help="Expression for g(u). Use 'np' for numpy functions.")
    parser.add_argument("--a", type=float, default=1.0, help="Coefficient a")
    parser.add_argument("--b", type=float, default=1.0, help="Coefficient b")
    parser.add_argument("--c", type=float, default=0.0, help="Constant c")
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x")
    parser.add_argument("--y0", type=float, default=1.0, help="Initial y")
    parser.add_argument("--x_end", type=float, default=1.0, help="End x")
    parser.add_argument("--n_steps", type=int, default=400, help="Number of steps for solver")
    parser.add_argument("--no_plot", action="store_true", help="Do not save comparison plot even if analytic exists")
    args = parser.parse_args()

    extra = {"n_steps": args.n_steps}
    try:
        sol = solve_linear_combination(
            args.g,
            args.a,
            args.b,
            args.c,
            x0=args.x0,
            y0=args.y0,
            x_end=args.x_end,
            **extra,
        )

        # 尝试解析解并绘图比较
        params = {"g": args.g, "a": args.a, "b": args.b, "c": args.c, "x0": args.x0, "y0": args.y0, "x_end": args.x_end, "n_steps": args.n_steps}
        try:
            xs_num, _ = _extract_xs_ys_from_result(sol)
            params["xs"] = xs_num
        except Exception:
            pass

        analytical_res = analytical_solution(params)
        if analytical_res is not None and not args.no_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("plots", f"Figure_{timestamp}.png")
            try:
                plot_analytical_vs_numerical(sol, params, save_path)
                print("Saved comparison plot to:", save_path)
            except Exception as e:
                warnings.warn(f"Failed to plot analytical vs numerical: {e}")

    finally:
        end_time = time.time()
        print(f"Total runtime: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()