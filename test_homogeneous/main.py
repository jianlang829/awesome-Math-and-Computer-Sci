# 外部调用代码（例如 test.py）
from homogeneous_solver.homogeneous_solver import solve_homogeneous

# 定义 g(u) = u + sin(u)，其中 u = y/x
# 方式 1：直接用字符串表达式（推荐，简单直观）
g_expr = "u + np.sin(u)"  # 支持 numpy 函数（需用 np. 前缀）

# 初值与求解范围
x0 = 1.0    # 初始 x（不能为 0）
y0 = 0.5    # 初始 y，即 y(x0)=0.5
x_end = 5.0 # 求解到 x=5.0
n_steps = 1000  # 步数（影响精度）

# 调用求解函数
result = solve_homogeneous(
    g=g_expr,
    x0=x0,
    y0=y0,
    x_end=x_end,
    n_steps=n_steps
)

'''可选'''
# 函数返回的结果是上述字典，存储在 result 变量中
# 提取结果（通过字典的键获取对应值）
xs = result["xs"]  # x 坐标数组
ys_sep = result["ys_sep"]  # 分离方法的 y(x) 解数组
ys_rk = result["ys_rk"]    # 龙格-库塔方法的 y(x) 解数组
analytic_sol = result["analytic_solution"]  # 解析解
runtime = result["runtime_sec"]  # 运行时间

# 这里以打印分离方法的结果为例（可根据需要替换为 ys_rk 或解析解）
print("x    |  y_sep(x)")
print("-" * 20)
for x, y in zip(xs[:5], ys_sep[:5]):  # 打印前 5 个点
    print(f"{x:.2f}  |  {y:.4f}")

# 如果需要对比多种解法，也可以同时打印
print("\n对比前 5 个点的结果：")
print("x    |  分离方法  |  龙格-库塔  |  解析解")
print("-" * 40)
for x, y1, y2 in zip(xs[:5], ys_sep[:5], ys_rk[:5]):
    print(f"{x:.2f}  |  {y1:.4f}  |  {y2:.4f}")