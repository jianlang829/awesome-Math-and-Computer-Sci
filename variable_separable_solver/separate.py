'''此程序求解微分方程 dy/dx = x/y'''

import sympy as sp

# 1. 定义符号
x = sp.Symbol('x')
# 无效构造
# y = sp.Symbol('y', function=True)  # 声明y是x的函数
y = sp.Function('y')

# 2. 构造微分方程 dy/dx = x/y
eq = sp.Eq(sp.Derivative(y(x), x), x / y(x))

# 3. 关键：求解通解（可能返回列表,通解可能包含正负两种情况,此时 dsolve 会将这些解放在列表中返回）
general_sol = sp.dsolve(eq)
print("通解的类型：", type(general_sol))
print("通解：", general_sol)

# 3.1 关键：从列表中提取第一个解（如果是列表的话）
if isinstance(general_sol, list):
    general_sol = general_sol[0]  # 取第一个解

# 4. 代入初始条件 y(0)=1 求特解
# 提取通解中的表达式（去除等号）
sol_expr = general_sol.rhs - general_sol.lhs
# 代入x=0, y=1，求解常数C1
C1 = sp.Symbol('C1')

# 加入打印语句，查看代入后的表达式
substituted_expr = sol_expr.subs({x: 0, y(x): 1})
print("代入初始条件后的表达式：", substituted_expr)
# 查看solve的结果
const = sp.solve(substituted_expr, C1)
print("solve返回的结果：", const)  # 若为空列表，则说明无解

# 遭遇无解情况()
# const = sp.solve(sol_expr.subs({x: 0, y(x): 1}), C1)[0]

# 代入常数得到特解
if const:  # 如果有解
    const = const[0]
    particular_sol = general_sol.subs(C1, const)
    print("特解：", particular_sol)
else:
    print("无法求解常数，可能初始条件矛盾或代入错误")