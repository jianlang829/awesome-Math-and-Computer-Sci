#!/usr/bin/env python3
'''english'''
"""
homogeneous_solver.py

Solve homogeneous ODEs of the form:
    dy/dx = g(y/x)

Using the substitution u = y/x (so y = u*x) we get:
    x * du/dx = g(u) - u
or
    du/dx = (1/x) * (g(u) - u)

This is a separable equation in the form:
    du/dx = f(x) * g_sep(u)
with
    f(x) = 1/x
    g_sep(u) = g(u) - u

This module adapts that transformation and calls solve_variable_separable
from variable_separable_solver.py with the transformed functions.
It then converts the solution for u(x) back to y(x) = x * u(x) keeping the
return format compatible with solve_variable_separable.

Public API:
- solve_homogeneous(g, *, x0, y0, x_end, n_steps=..., **kwargs)
  g can be a callable g(u) or a string expression in 'u' using numpy as np
  and math module (e.g. "u**2 + 1" or "np.sin(u)").
- main() provides a simple CLI demo.

Note: x0 must be non-zero because f(x)=1/x is singular at x=0.
"""

from typing import Any, Callable, Tuple
import numpy as np
import math
import argparse
import types
import copy

# Import the separable solver - assumes it's available in the same package or path.
try:
    from ..variable_separable_solver.variable_separable_solver import solve_variable_separable
except Exception as e:
    raise ImportError(
        "Could not import solve_variable_separable from variable_separable_solver.py. "
        "Make sure that file is present and on PYTHONPATH."
    ) from e

def _make_callable_g(g) -> Callable[[float], float]:
    """
    Ensure g is a callable g(u). If g is a string, create a safe eval callable
    that evaluates the expression in terms of 'u', with access to numpy (np)
    and math modules.

    WARNING: eval is used with builtins removed - keep in mind this is not
    perfectly sandboxed for all threat models but is practical for CLI use.
    """
    if callable(g):
        return g
    if isinstance(g, str):
        expr = g.strip()
        # Build an eval environment
        safe_globals = {"__builtins__": None, "np": np, "math": math}
        # Pre-compile expression for slightly better performance/error reporting
        code = compile(expr, "<g_expr>", "eval")

        def g_func(u):
            # allow numpy arrays as well as scalars
            local = {"u": u}
            return eval(code, safe_globals, local)

        return g_func
    raise TypeError("g must be a callable or a string expression in variable 'u'.")


def _convert_solution_u_to_y(result: Any) -> Any:
    """
    Convert the solution returned by solve_variable_separable (which solved for u)
    into the corresponding solution for y = x * u.

    This function tries to preserve the original return type and structure:
    - If result is a (xs, us) tuple/list of two sequences -> return (xs, ys).
    - If result is a dict and contains keys like 'x'/'y' or 'xs'/'ys' -> convert the dependent var.
    - If result is any other object with attributes 'x'/'y' or 'xs'/'ys', attempt to create
      a shallow copy and set the appropriate attribute.
    - Otherwise, return a dict {'x': xs, 'y': ys}.

    Note: we treat the second element (index 1) of a 2-element tuple/list as the dependent
    variable array to be converted.
    """
    # Helper to do elementwise multiplication, working for lists/tuples/ndarrays
    def times_x(xs, us):
        xs_arr = np.asarray(xs)
        us_arr = np.asarray(us)
        ys_arr = xs_arr * us_arr
        # Try to return same type as input us if it was a list
        if isinstance(us, list):
            return ys_arr.tolist()
        return ys_arr

    # Case 1: tuple or list where len>=2 and index 0 is xs, index 1 is us
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        xs = result[0]
        us = result[1]
        ys = times_x(xs, us)
        # Preserve tuple/list type
        out = list(result)
        out[1] = ys
        return tuple(out) if isinstance(result, tuple) else out

    # Case 2: dict-like
    if isinstance(result, dict):
        res = dict(result)  # shallow copy
        # try likely key pairs
        if "x" in res and "y" in res:
            # here 'y' actually contains u, so convert it
            res["y"] = times_x(res["x"], res["y"])
            return res
        if "xs" in res and "ys" in res:
            res["ys"] = times_x(res["xs"], res["ys"])
            return res
        # fallback: try to find any single dependent array key (heuristic)
        # prefer keys that are not x/xs
        xs_key = None
        for k in ("x", "xs"):
            if k in res:
                xs_key = k
                break
        if xs_key is not None:
            # find first other array-like key
            for k, v in res.items():
                if k == xs_key:
                    continue
                # convert this one
                try:
                    res[k] = times_x(res[xs_key], v)
                    return res
                except Exception:
                    pass
        # cannot confidently convert - return as-is but also return a new 'y_from_u' if possible
        return res

    # Case 3: object with attributes
    if hasattr(result, "__dict__") or hasattr(result, "__slots__"):
        # try to shallow copy the object to avoid mutating original
        try:
            out_obj = copy.copy(result)
        except Exception:
            out_obj = result  # fallback to mutating original

        # try both ('x','y') and ('xs','ys')
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
        # fallback - return object unchanged
        return out_obj

    # Unknown type - cannot transform; raise or return informative dict
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
    Solve dy/dx = g(y/x) on [x0, x_end] with initial condition y(x0) = y0.

    Parameters
    - g: callable g(u) or string expression in 'u' (uses np and math).
    - x0, y0: initial condition (x0 must be non-zero).
    - x_end: endpoint for x.
    - n_steps: optional, forwarded to solve_variable_separable (if provided).
    - kwargs: any other keyword args forwarded to solve_variable_separable.

    Returns:
    - The same return value as solve_variable_separable, but with the dependent
      variable values converted from u(x) back to y(x) = x * u(x).
    """
    # Validate x0
    if x0 == 0:
        raise ValueError("x0 must be non-zero for the substitution u = y/x (f(x)=1/x singular at 0).")

    # Ensure g is callable
    g_callable = _make_callable_g(g)

    # Define transformed separable equation:
    # du/dx = (1/x) * (g(u) - u)
    def f_x(x):
        return 1.0 / x

    def g_sep(u):
        return g_callable(u) - u

    # initial u0
    u0 = y0 / x0

    # Prepare arguments to forward preserving names expected by solver
    solver_kwargs = dict(kwargs)  # shallow copy
    if n_steps is not None:
        solver_kwargs["n_steps"] = n_steps

    # Call the variable separable solver solving for u as the dependent variable
    result_u = solve_variable_separable(
        f_x=f_x,
        g_y=g_sep,
        x0=x0,
        y0=u0,
        x_end=x_end,
        **solver_kwargs,
    )

    # Convert solution for u(x) to y(x) = x * u(x)
    result_y = _convert_solution_u_to_y(result_u)
    return result_y


def _demo_example():
    """
    Demo solving dy/dx = g(y/x) with g(u) = u**2 (i.e., dy/dx = (y/x)**2).
    This is just an example to show usage.
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
    # If it's a tuple/list, print sample
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
    parser.add_argument("--x_end", type=float, default=5.0, help="End x.")
    parser.add_argument("--n_steps", type=int, default=400, help="Number of steps (forwarded).")
    parser.add_argument("--no_plot", action="store_true", help="If set, do not ask solver to show plot.")
    args = parser.parse_args()

    extra = {"n_steps": args.n_steps}
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

    # Print brief summary
    print("Solution returned of type:", type(sol))
    if isinstance(sol, (tuple, list)) and len(sol) >= 2:
        xs, ys = sol[0], sol[1]
        print(f"Computed {len(xs)} points. First 5 (x, y):")
        for xi, yi in zip(xs[:5], np.asarray(ys)[:5]):
            print(f"  {xi:.6g}, {yi:.6g}")
    elif isinstance(sol, dict):
        xs = sol.get("x") or sol.get("xs")
        ys = sol.get("y") or sol.get("ys")
        if xs is not None and ys is not None:
            print(f"Computed {len(xs)} points. First 5 (x, y):")
            for xi, yi in zip(xs[:5], np.asarray(ys)[:5]):
                print(f"  {xi:.6g}, {yi:.6g}")
    else:
        print("Solution returned in an unrecognized structure; inspect manually.")


if __name__ == "__main__":
    main()