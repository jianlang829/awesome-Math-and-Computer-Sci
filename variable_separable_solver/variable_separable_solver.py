'''将搜索区间从固定的 `[y0-100, y0+100]` 改为基于上一步结果的动态区间 `[ys[i-1]*0.5, ys[i-1]*1.5]`，并将区间定义放入循环，能显著提升算法的精确度和速度，核心原因如下：


### **1. 精度提升：减少假根，聚焦真实解邻域**
- **缩小搜索范围，降低多解干扰**：  
  原固定区间 `[y0±100]` 可能包含多个满足 `F(y)=G(x)` 的点（尤其是当 `F(y)` 非单调时，如震荡函数），导致 `brentq` 误判假根。而动态区间以上一步的解 `ys[i-1]` 为基准，仅在其附近的小范围（0.5~1.5倍）搜索，大幅减少了假根存在的可能性。
  
- **适配解的局部连续性**：  
  微分方程的解通常具有局部连续性（相邻 `x` 对应的 `y` 变化平滑），尤其是当 `x` 采样间隔较小时，`ys[i]` 与 `ys[i-1]` 差异不会太大。动态区间贴合这种连续性，确保搜索范围集中在真实解的邻域，提高求根准确性。

- **避免“累积误差爆炸”**：  
  若初始固定区间找到的解偏离真实值，后续步骤会基于错误值继续计算，误差会累积放大。动态区间通过“上一步正确解约束下一步范围”，形成反馈校正机制，抑制误差累积。


### **2. 速度提升：减少无效计算，优化搜索效率**
- **减少采样和求根的计算量**：  
  原固定区间范围大（如±100），采样点数 `N_samples=400` 需覆盖整个宽区间，导致大量无效采样（远离真实解的区域）。动态区间范围小（仅上一步解的0.5~1.5倍），相同采样点数下，对真实解邻域的采样密度更高，且无需在无关区域浪费计算资源。
  
- **降低 `brentq` 迭代次数**：  
  `brentq` 求根效率与初始区间大小正相关：区间越小，迭代收敛越快。动态区间从一开始就聚焦于真实解附近，大幅减少 `brentq` 所需的迭代步数，尤其在解变化平缓的区域效果显著。

- **减少扩展区间的重试次数**：  
  原固定区间若未找到解，需多次扩展（如扩大2倍，最多3次），每次扩展都要重新采样和搜索，耗时巨大。动态区间因贴合解的趋势，极少需要扩展，甚至无需扩展即可找到有效区间，节省了扩展逻辑的计算成本。


### **3. 鲁棒性增强：适配解的动态变化趋势**
- **适应解的增长/衰减特性**：  
  例如，对于指数增长的解（如 `dy/dx = x*y`），`ys[i]` 会随 `x` 快速增大，固定区间 `[y0±100]` 很快会失效（解超出范围），而动态区间 `[ys[i-1]*0.5, ys[i-1]*1.5]` 会随解同步扩大，始终覆盖真实解。
  
- **自动调整区间尺度**：  
  当解较小时（如初始阶段 `y≈y0`），区间范围较小（如 `y0*0.5` 到 `y0*1.5`），避免过度搜索；当解增大后，区间会按比例自动扩展，无需人工调整尺度参数，适配不同量级的解。


### **总结**
动态区间通过**“局部连续性约束”**和**“自适应尺度调整”**，既保证了搜索范围始终聚焦于真实解邻域（提升精度），又减少了无效计算和迭代次数（提升速度），同时对解的各种变化趋势（增长、衰减、平缓）具有更强的适应性，是数值求根问题中优化效率和精度的经典策略。'''

'''variable_separable_solver.py
求解形如 dy/dx = f(x) * g(y) 的通用变量分离型一阶常微分方程(数值方法，积分 + 求根，非解析方法)
严格流程:分离变量(1/g(y) dy = f(x) dx)→ 两端数值积分(scipy.integrate.quad)
→ 反求 y(x)(用根求解 brentq)。
并使用 solve_ivp (RK45) 作为基准解比较，绘图并计算 RMSE。
方法在数学原理上就是变量分离法，只是因为没有解析解（或没用到解析解），才用 “数值积分 + 数值求根” 来落地 —— 这是变量分离法在实际计算中的常见做法（尤其是复杂方程）
为什么不用解析解，在实际中很多复杂方程没有解析解，只能用数值解
依赖:numpy, matplotlib, scipy
'''

'''安装:pip install --upgrade numpy matplotlib scipy
作者:jianlang829,lang306'''

'''分离变量数值解 与 RK45 数值解 的 RMSE = 96.7663600935
分离变量数值解 与 解析解 的 最大绝对误差 = 105.3865498333
分离变量数值解 与 解析解 的 平均绝对误差 = 96.3461285046

改进1：提高 quad 的积分精度参数
在 G_of_x 和 F_of_y 函数中，给 quad 增加 epsabs（绝对误差上限）和 epsrel（相对误差上限）参数，强制提高积分精度：
分离变量数值解 与 RK45 数值解 的 RMSE = 96.7670558829
分离变量数值解 与 解析解 的 最大绝对误差 = 105.5874697237
分离变量数值解 与 解析解 的 平均绝对误差 = 96.3467960682
测试：无效果，故改回

改进2：brentq 求根的准确性直接决定最终的 y 值，需重点调整搜索策略。
扩大并动态调整求根搜索区间,解是指数增长（如 dy/dx = x*y），可进一步扩大到 [y0-1e6, y0+1e6]，避免漏解
RuntimeError: 在为 x=0.01 求解 y 时，无法在 [-999999.0, 1000001.0](及扩展区间)中找到使 F(y)-G(x) 有符号变化的区间，无法用 brentq 求根。 可能原因:g(y) 在区间内有奇点，或解超出搜索区间范围。请扩大搜索区间或检查 g(y)
猜想是区域过大遇到了奇点
？？？
修改为1000.0
分离变量数值解 与 RK45 数值解 的 RMSE = 939.2071247549
分离变量数值解 与 解析解 的 最大绝对误差 = 976.3138692226
分离变量数值解 与 解析解 的 平均绝对误差 = 935.9572792547
无效果，改回
？？？
改进3：增加采样点数，400改为1000提高区间搜索精度
采样点太少可能错过符号变化区间，导致求根错误，增加采样点数：
分离变量数值解 与 RK45 数值解 的 RMSE = 99.9102103026
分离变量数值解 与 解析解 的 最大绝对误差 = 105.5900773468
分离变量数值解 与 解析解 的 平均绝对误差 = 99.6366285153
无效果，改回
经过5次精度调优，无效果，说明核心问题不在积分精度本身，而是变量分离法的 “求根环节” 或 “迭代逻辑” 存在根本性偏差—— 积分精度调优只是 “微调”，没法解决核心逻辑漏洞。
'''
import time
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
def f_x(x):
    # 这里修改 f(x)
    return x

def g_y(y):
    # 这里修改 g(y)
    return y

# 如果要使用示例2(dy/dx = sin(x)*(1+y^2))，请把上面的 f_x/g_y 替换为:
# def f_x(x):
#     return np.sin(x)
# def g_y(y):
#     return 1 + y**2
# 或者将 EXAMPLE = 2，然后在下方参数配置里替换为示例2(已给出说明)

# 初始条件与求解参数(可修改)
x0 = 0.0         # 初始 x(或求解区间起点)
y0 = 1.0         # 初始 y(必须满足 g_y(y0) != 0)
x_end = 2.0      # 求解区间终点
n_steps = 200  
'''改进5：减小迭代步长（降低累积误差）
步长越大，每一步的截断误差和积分误差累积越严重，需增加迭代步数 n_steps：
增加迭代步数
结果：IntegrationWarning: The maximum number of subdivisions (200) has been achieved.
  If increasing the limit yields no improvement it is advised to analyze
  the integrand in order to determine the difficulties.  If the position of a
  local difficulty can be determined (singularity, discontinuity) one will
  probably gain from splitting up the interval and calling the integrator
  on the subranges.  Perhaps a special-purpose integrator should be used.
  积分警告：已达到最大细分数（200）。
分离变量数值解 与 RK45 数值解 的 RMSE = 96.9266208953
分离变量数值解 与 解析解 的 最大绝对误差 = 105.3865498333
分离变量数值解 与 解析解 的 平均绝对误差 = 96.7218949196
  如果增加限制没有改善，建议分析被积函数以确定困难所在。如果可以确定局部困难的位置（奇点、不连续处），可能通过拆分区间并在子区间上调用积分器会有所收获。或许应该使用专用积分器。'''
# n_steps = 5000    # 迭代步数(步长 h 自动计算为 (x_end-x0)/n_steps )
'''步长和迭代步数的关系
假设你要从 x0(起点)走到 x_end(终点)，这段路程就是 “求解区间”。
迭代步数(n_steps)：你走的 “总步数”(比如 200 步)。
步长(h)：你每一步迈出去的 “距离”(比如每步走 0.05 米)。
两者的关系就是：步长 h = 总路程(x_end - x0)÷ 迭代步数(n_steps)
为什么这俩参数影响误差？
这里是积分，分得越细（即步长短步数多），越贴合真实，误差越小'''
# -------------------------------------------------------------------------

# ----------------------- 合法性检查 -----------------------
if g_y(y0) == 0:
    print("错误:初始值 y0 导致 g(y0) == 0，会在分离变量时导致除以零。请更换 y0 或修改 g_y(y)。")
    sys.exit(1)

# 设置 x 网格
xs = np.linspace(x0, x_end, n_steps + 1)
'''np.linspace()：NumPy 库的一个函数，作用是 “在两个数之间，均匀地拉出一串数”（比如在 1 和 11 之间，均匀拉 201 个数）。
x0：起点（比如你之前的初始 x 值 1）。
x_end：终点（比如你要求解到的 x 值 11）。
n_steps + 1：要生成的 “点数”（比如迭代步数 n_steps=200，就生成 200+1=201 个点）。
xs：最终得到的 “x 值列表”（一个包含 201 个均匀分布 x 的数组）。'''

# ----------------------- 核心求解函数模块 -----------------------
# 计算 G(x) = \int_{x0}^{x} f(s) ds
def G_of_x(x):
    '''你要解的方程是 dy/dx = f(x)g(y)，分离变量后是 1/g(y)dy = f(x)dx。两边积分后会得到：∫(y0到y) 1/g(t)dt = ∫(x0到x) f(s)ds + C（积分常数）。
而这个 G_of_x(x) 函数，就是专门计算右边的 ∫(x0到x) f(s)ds —— 简单说，它的作用是 “给一个 x，就返回 f (x) 从起点 x0 到这个 x 的积分结果”。'''
    
    '''此函数在计算右边的积分'''
    
    '''def G_of_x(x):：定义一个函数，输入是任意 x 值，输出是积分结果。
    quad(...)：scipy 库的数值积分函数，专门用来计算定积分（因为有些 f (x) 的积分没有解析解，只能用数值方法逼近）。
    lambda s: f_x(s)：积分的被积函数，就是你定义的 f_x（比如示例 1 里的 f_x=x，这里就相当于 “对 s 求积分，被积函数是 s”）。
    x0 和 x：积分的上下限 —— 从 x0（初始 x 值）积分到输入的 x。
    limit=200：积分的迭代次数上限，设得高一点能提高积分精度（避免积分本身出错）。
    val, err = quad(...)：quad 返回两个值，val 是积分的结果（我们要的），err 是积分的误差估计（可以忽略，重点用 val）。
    return val：函数最终返回积分结果，也就是 G(x) = ∫(x0到x) f(s)ds。'''
    
    '''满足变量分离的积分要求：分离变量后右边的积分是 “随 x 变化的”，每个 x 都对应一个积分值，G_of_x(x) 就是高效计算这个值的工具。'''
    
    '''适配任意 f (x)：不管 f (x) 能不能用公式算出解析积分（比如 f (x)=e^(-x²) 就没有初等解析解），quad 都能通过数值方法算出近似值，让代码能通用。'''
    
    '''为后续求 y 铺路：分离变量后的左边积分是关于 y 的，知道了右边的积分结果（G (x)），才能通过根求解（brentq）反推出对应的 y 值。'''
    val, err = quad(lambda s: f_x(s), x0, x, limit=200) # quad 返回 (value, err)
    '''改进1：提高精度：无效'''
    # val, err = quad(lambda s: f_x(s), x0, x, limit=500, epsabs=1e-12, epsrel=1e-10)
    return val

# 计算 F(y) = \int_{y0}^{y} 1/g(t) dt
def F_of_y(y):
    '''此函数在计算左边的积分'''
    # integrand = 1 / g_y(t)
    def integrand(t):
        '''1. 从g(t)得到1.0/g(t),2. 处理g(t)==0,即某点发散无积分，的异常情况'''
        '''其实核心逻辑很简单：当积分的被积函数（这里是 1/g (t)）(g(t)==0)在积分区间内某点变成无穷大时，这个积分就会 “发散”—— 也就是积分结果会趋近于无穷大，根本算不出一个有限的数值。'''
        gt = g_y(t)
        if gt == 0:
            # 为避免除零抛出，返回大型数值(quad 在遇到真正奇异点可能报错)
            # 但更稳妥的做法是让 quad 发现并报错，由上层捕获
            raise ZeroDivisionError(f"g(y) 在 t={t} 处为零，积分发散或不可直接计算。")
        return 1.0 / gt
    # quad 支持上下限反向(会返回负值)
    # val, err = quad(integrand, y0, y, limit=200)
    '''改进1：提高精度'''
    '''在 G_of_x 和 F_of_y 函数中，给 quad 增加 epsabs（绝对误差上限）和 epsrel（相对误差上限）参数，强制提高积分精度：'''
    '''参数含义：epsabs=1e-12 表示积分的绝对误差不超过 1e-12；limit=500 增加积分区间的细分次数（默认 50，复杂函数需更大）。'''
    '''尤其对震荡剧烈或接近奇点的 f(x)/g(y)，能显著降低积分误差'''
    '''无效'''
    val, err = quad(integrand, y0, y, limit=200)
    return val

# H(y; rhs) = F(y) - rhs，这里 rhs = G(x) 为常数
def H_of_y(y, rhs):
    '''这个函数是变量分离法求解中的 “方程桥梁”，核心作用是将积分后的等式转化为 “求根问题”，具体拆解如下：
一、核心含义
变量分离后，两边积分的等式是：1/g(y)dy = f(x)dx,左边的积分就是 F_of_y(y)（你之前定义的函数）。
右边的积分就是 G_of_x(x)（记为 rhs，即 “right-hand side，等式右边的值”）。
因此，等式可改写为：F_of_y(y) - rhs = 0 —— 这就是 H_of_y(y, rhs) 的定义：H_of_y(y, rhs) = F(y) - rhs
二、通俗理解
想象你要解一个方程：A = B。为了找到满足等式的未知数，通常会转化为 A - B = 0，然后找 “使左边等于 0 的未知数”。
这里的 H_of_y(y, rhs) 就是干这个的：
输入一个 y 和已知的 rhs（右边积分的结果），计算 F(y) - rhs。
当这个结果等于 0 时，对应的 y 就是我们要找的解（因为此时 F(y) = rhs，满足积分等式）。
三、核心用途
为 “反求 y 值” 铺路。变量分离后，我们已知右边的 rhs = G(x)（对于某个 x，这个值是确定的常数），但左边的 F(y) 是关于 y 的函数。要找到对应的 y，就需要解 F(y) = rhs —— 而这等价于解 H_of_y(y, rhs) = 0。
后续会用 scipy.integrate.brentq 等求根函数，找到使 H_of_y(y, rhs) = 0 的 y 值，这个 y 就是 “对应 x 的解”。
简单说：H_of_y 把 “求解积分等式” 转化为 “求解方程零点”，是连接积分结果和最终 y 值的关键工具。'''
    return F_of_y(y) - rhs

# 在给定 x 网格上用“分离变量数值法”求 y(x)
def solve_by_separation(xs, y0):
    '''求解数值解，即ys（与xs一一对应）'''
    ys = np.zeros_like(xs)
    ys[0] = y0
    # 预计算 G(x) 对每个 x
    Gs = np.array([G_of_x(x) for x in xs])

    # 根求解区间(来自要求):以 [y0 - 100, y0 + 100] 为初始搜索区间
    '''改进2：brentq 求根的准确性直接决定最终的 y 值，需重点调整搜索策略。
扩大并动态调整求根搜索区间,解是指数增长（如 dy/dx = x*y），可进一步扩大到 [y0-1e6, y0+1e6]，避免漏解'''
    '''确认是假根问题，将 solve_by_separation 中的搜索区间改为 “以上一步 y 为起点”：
    后续采样、求根都用 search_low 和 search_high，替代原来的 global_search_low/high'''
    # global_search_low = y0 - 100.0
    # global_search_high = y0 + 100.0

    # 为每个 x(从索引1开始)求出 y，使得 F(y) = G(x)
    for i in range(1, len(xs)):
        rhs = Gs[i]  # 这是 F(y) 的目标值

        # 特殊情况:如果 rhs == 0(即 x == x0)，则 y = y0
        if abs(rhs) < 1e-15:
            ys[i] = y0
            continue
        '''放入循环里实现动态区间
        适配示例一的搜索范围'''
        # 已知y随x递增，区间可设为“上一步y - 10（容错）到 上一步y + 100”
        # search_low = ys[i-1] - 10.0 # 上一步的y值（当前y一定比它大）
        # search_high = ys[i-1] + 100.0  # 每次向上扩展100（可根据y增长速度调整）

        # # 按比例缩放，更贴合指数增长
        # search_low = ys[i-1] * 0.9  # 向下留10%容错（避免微小波动）
        # search_high = ys[i-1] * 2.0  # 向上扩1倍（足够覆盖指数增长的步长增量）

        # 动态区间：以y0为基准，初始步留更小的扩展范围，后续步按比例扩
        search_low = ys[i-1] * 0.5  # 向下留1%容错（比之前的0.9更贴近y0）
        search_high = ys[i-1] * 1.5  # 向上留1%容错（初始步y变化小，不用扩2倍）

        # # 已知y随x递减，区间可设为“上一步y - 100 到 上一步y + 10（容错）”
        # search_low = ys[i-1] - 100.0  # 主要向下扩展
        # search_high = ys[i-1] + 10.0  # 留10的容错

        # 对未知单调性的通用情况：如果不确定解的趋势，可用你的修改，但缩小扩展幅度（比如 ±50）,100，减少假根概率
        # search_low = ys[i-1] - 50.0  # 左右各扩50（幅度可根据实际情况调整）
        # search_high = ys[i-1] + 50.0

        # search_low = ys[i-1] - 100.0  # 左右各扩50（幅度可根据实际情况调整）
        # search_high = ys[i-1] + 100.0

        '''二、潜在问题：区间过宽可能再次引入假根
区间太宽（比如 ±100）可能包含多个满足 F(y)-rhs=0 的点（假根），尤其是当 F(y) 非单调时（比如 g(y) 是周期函数，导致 1/g(y) 的积分震荡）。
例如：若 F(y) 是正弦函数（震荡的），宽区间内可能有多个点满足 F(y)=rhs，brentq 可能找到第一个符号变化的点（假根），而非真实解。'''

        '''这样修改后，代码既能兼容更多类型的方程，又能有效避免假根，精度会更稳定。
        建议根据方程的单调性进一步优化：已知趋势时，往趋势方向多扩、反方向少扩（留容错）；未知时，适当缩小扩展幅度（如 ±50），平衡兼容性和抗假根能力。'''
        # 为了稳定求根，先在 global 区间内做采样，寻找符号变化的子区间
        '''改进3：增加采样点数，400改为1000提高区间搜索精度
采样点太少可能错过符号变化区间，导致求根错误，增加采样点数：
无效果，改回'''
        N_samples = 400  # 采样点数(可调整，越多越稳健但慢)
        # N_samples = 200  # 从50提到100，提高初始步的采样密度
        sample_points = np.linspace(search_low, search_high, N_samples)

        '''???400 > 1000'''
        # sample_points = np.linspace(global_search_low, global_search_high, N_samples)
        H_vals = []
        finite_flags = []
        # 临时打印前10个和后10个H_vals，验证符号
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
        for j in range(N_samples - 1):
            if not (finite_flags[j] and finite_flags[j + 1]):
                continue
            v1 = H_vals[j]
            v2 = H_vals[j + 1]
            # if v1 == 0.0:
            #     y_root = sample_points[j]
            #     bracket_found = True
            #     break
            # if v2 == 0.0:
            #     y_root = sample_points[j + 1]
            #     bracket_found = True
            #     break

            # 新增：如果其中一个点的H(y)接近0，直接视为根
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
                    '''改进4：增加采样点数，提高区间搜索精度
采样点太少可能错过符号变化区间，导致求根错误，增加采样点数：
RuntimeError: 在为 x=0.01 求解 y 时，无法在 [-99.0, 101.0](及扩展区间)中找到使 F(y)-G(x) 有符号变化的区间，无法用 brentq 求根。 可能原因:g(y) 在区间内有奇点，或解超出搜索区间范围。请扩大搜索区间或检查 g(y)。
范围太小，无效，改回'''
                    y_root = brentq(lambda yy: H_of_y(yy, rhs), a, b, maxiter=200)
                    # y_root = brentq(lambda yy: H_of_y(yy, rhs), a, b, maxiter=500, rtol=1e-12, atol=1e-15)
                    bracket_found = True
                    break
                except Exception as e:
                    # 如果 brentq 失败，继续尝试下一个符号变化区间
                    # print(f"brentq 在区间 ({a}, {b}) 失败:{e}")
                    continue

        if not bracket_found:
            # 如果没有在初始全局区间找到符号变化，尝试扩展搜索区间(线性扩展两倍)
            expand_factor = 2.0
            # 在扩展区间的循环中，把 expand_factor 从2.0改成1.2
            # expand_factor = 1.2  # 每次扩展20%，更精准
            tries = 0
            max_tries = 3
            # max_tries = 5  # 最多试5次，足够覆盖初始步的小变化
            expanded_low = search_low
            expanded_high = search_high
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
                            '''改进4：增加采样点数，提高区间搜索精度
采样点太少可能错过符号变化区间，导致求根错误，增加采样点数：
raise RuntimeError(f"在为 x={xs[i]:.6g} 求解 y 时，无法在 [{global_search_low}, {global_search_high}](及扩展区间)中找到使 F(y)-G(x) 有符号变化的区间，无法用 brentq 求根。"
RuntimeError: 在为 x=0.01 求解 y 时，无法在 [-99.0, 101.0](及扩展区间)中找到使 F(y)-G(x) 有符号变化的区间，无法用 brentq 求根。 可能原因:g(y) 在区间内有奇点，或解超出搜索区间范围。请扩大搜索区间或检查 g(y)。
范围太小，无效，改回'''
                            y_root = brentq(lambda yy: H_of_y(yy, rhs), a, b, maxiter=200)
                            # y_root = brentq(lambda yy: H_of_y(yy, rhs), a, b, maxiter=500, rtol=1e-12, atol=1e-15)
                            bracket_found = True
                            break
                        except Exception:
                            continue
                tries += 1

        if not bracket_found:
            raise RuntimeError(f"在为 x={xs[i]:.6g} 求解 y 时，无法在 [{search_low}, {search_high}](及扩展区间)中找到使 F(y)-G(x) 有符号变化的区间，无法用 brentq 求根。"
                               " 可能原因:g(y) 在区间内有奇点，或解超出搜索区间范围。请扩大搜索区间或检查 g(y)。")

        ys[i] = y_root

    # print(ys)
    return ys

# ----------------------- 基准验证模块(RK45) -----------------------
def solve_by_rk45(x0, y0, x_end, xs):
    def rhs(x, y):
        return f_x(x) * g_y(y)
    sol = solve_ivp(rhs, (x0, x_end), [y0], method='RK45', t_eval=xs)
    if not sol.success:
        raise RuntimeError("solve_ivp (RK45) 未能成功求解: " + str(sol.message))
    return sol.y[0]

# ----------------------- 结果可视化与误差计算模块 -----------------------
'''这个模块是程序的 “结果展示与精度评估工具”，核心作用是计算两种数值解的误差，并绘制对比图，让你直观看到变量分离法的求解效果'''
def compute_rmse(a, b):
    '''compute_rmse(a, b)：计算均方根误差
作用：衡量两组数据（a 和 b）的整体偏差，数值越小精度越高。
计算逻辑：先算每组数据的差值→平方（放大大幅偏差）→求平均值→开根号，最终得到 RMSE 值。
这里用于对比 “变量分离解（ys_sep）” 和 “RK45 解（ys_rk）” 的整体偏差。'''
    '''此处函数作用：定义rmse的计算公式，即先算每组数据的差值→平方（放大大幅偏差）→求平均值→开根号，最终得到 RMSE 值'''
    return np.sqrt(np.mean((a - b) ** 2))

def visualize_and_report(xs, ys_sep, ys_rk, example=1, analytic_solution=None):
    # 计算并打印 RMSE(保留10位小数)
    '''第一步：计算并打印误差
打印 “变量分离解 vs RK45 解” 的 RMSE（保留 10 位小数）。
如果是示例 1（有解析解），额外计算 “变量分离解 vs 解析解” 的最大绝对误差（最离谱的偏差）和平均绝对误差（整体平均偏差），并打印。'''
    rmse_val = compute_rmse(ys_sep, ys_rk)
    print(f"分离变量数值解 与 RK45 数值解 的 RMSE = {rmse_val:.10f}")

    # 绘图
    '''第二步：绘制对比图
创建一张 10×6 英寸的图，在同一张图上画三条曲线（如果有解析解的话）：
变量分离解：实线，标签 “分离变量数值解”。
RK45 解：虚线，标签 “RK45 数值解”。
解析解（仅示例 1）：点线，标签 “解析解”。
添加坐标轴标签（x、y）、标题、图例、网格线，让图清晰易读。
保存图片为 “Figure_1.png”（存在程序运行目录下）。'''
    '''图中内容解读：
横坐标（x）：你设置的求解区间（从 x0 到 x_end）。
纵坐标（y）：对应 x 的 y 值。
三条曲线（示例 1）：
解析解（点线）：理论上的 “真实解”，是完美标准。
RK45 解（虚线）：高精度的 “基准解”，接近真实解。
变量分离解（实线）：你方法的求解结果。
判断标准：如果三条曲线几乎重合，说明你的变量分离法精度很高；如果实线和另外两条线偏离很大，说明误差大，需要优化（比如调整步长、积分精度）。
示例 2（无解析解）：图中只有两条曲线（分离变量解 + RK45 解），核心看两者是否吻合。'''
    '''特别注意：示例一中，rk45与解析解曲线高度重合'''
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys_sep, label="分离变量数值解", lw=2)
    plt.plot(xs, ys_rk, '--', label="RK45 数值解", lw=2) #--r相比--将线条颜色改为红色
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
    plt.savefig("Figure_1.png")

# ----------------------- 主程序(执行流程) -----------------------
def main():
    '''main 函数的作用是按顺序执行求解流程，简单说就是：
决定用哪个示例方程（示例 1 或示例 2）；
调用变量分离法求解，得到 ys_sep；
调用 RK45 方法求解，得到 ys_rk；
（如果是示例 1）生成解析解；
调用可视化函数，输出对比图和误差报告。
它就像一个 “指挥官”，按步骤调动各个工具（求解函数、绘图函数），最终完成从 “输入参数” 到 “输出结果” 的全过程。'''
    # 如果需要按示例2切换 f_x/g_y，可在此处替换函数定义(演示说明)
    '''这里是 “方程切换开关”：如果 EXAMPLE 变量设为 2，就会用示例 2 的 f_x 和 g_y（需要手动取消注释）；默认用示例 1 的方程（dy/dx = x*y）。'''
    # if EXAMPLE == 2:
        # 示例2:dy/dx = sin(x) * (1 + y^2)
        # 说明:要切换到示例2，请解除下面两行的注释(或者直接在上方参数区域替换 f_x 和 g_y)
        # global f_x, g_y
        # f_x = lambda x: np.sin(x); g_y = lambda y: 1 + y**2
    start_time = time.time()
    print("开始用分离变量法计算(数值积分 + 根求解)...")
    ys_sep = solve_by_separation(xs, y0)
    print("分离变量法计算完成。")

    print("开始用 solve_ivp (RK45) 作为基准解...")
    ys_rk = solve_by_rk45(x0, y0, x_end, xs)
    print("RK45 计算完成。")

    '''按顺序调用之前定义的两个核心求解函数，得到各自的 y 值列表。'''

    '''这里的解析解是 “手动输入” 的：因为示例 1 的方程（dy/dx = x*y）有明确的解析解公式（y = y0*e^(x²/2 - x0²/2)），所以直接在代码中写出这个公式，定义为 analytic_sol 函数。
作用是生成 “理论真实解”，用于和数值解（变量分离法、RK45）对比，验证数值方法的精度。'''
    # 若为示例1，构造解析解函数
    analytic_solution = None
    if EXAMPLE == 1:
        # 解析解:dy/dx = x*y -> ln(y) - ln(y0) = (x^2 - x0^2)/2
        def analytic_sol(x_arr):
            return y0 * np.exp((x_arr ** 2 - x0 ** 2) / 2.0)
        analytic_solution = analytic_sol

    # print('-'*50)

    # '''排查 “假根” 问题'''
    # # 1. 选择一个中间点（比如第50个点，确保索引不超过数组长度）
    # check_idx = 50  # 可修改为其他索引（如100、200）
    # if check_idx >= len(xs):
    #     check_idx = len(xs) // 2  # 若50超出范围，取中间位置
    # x_check = xs[check_idx]
    # print(f"\n=== 开始排查 x = {x_check:.6f} 处的误差 ===")

    # # 2. 计算 rhs = G(x)（右边积分结果）
    # rhs = G_of_x(x_check)  # 直接调用G_of_x计算当前x的积分
    # print(f"步骤1：G(x) = {rhs:.10f}（右边积分结果）")

    # # 3. 用解析解计算真实y值（示例1专属）
    # if EXAMPLE == 1:
    #     y_true = y0 * np.exp((x_check**2 - x0**2) / 2.0)
    #     print(f"步骤2：解析解 y_true = {y_true:.10f}（真实值）")

    #     # 4. 计算 F(y_true)（左边积分结果）
    #     f_true = F_of_y(y_true)
    #     print(f"步骤3：F(y_true) = {f_true:.10f}（左边积分结果）")

    #     # 5. 对比 F(y_true) 和 rhs（理论上应相等）
    #     integral_error = abs(f_true - rhs)
    #     print(f"步骤4：F(y_true) 与 G(x) 的误差 = {integral_error:.1e}")
    #     if integral_error < 1e-5:
    #         print("→ 结论：积分计算正确（误差在合理范围）")
    #     else:
    #         print("→ 警告：积分计算错误（误差过大，检查F_of_y或G_of_x）")

    #     # 6. 对比程序计算的ys_sep与真实y_true
    #     y_sep = ys_sep[check_idx]
    #     y_error = abs(y_sep - y_true)
    #     print(f"\n步骤5：程序计算的y_sep = {y_sep:.10f}")
    #     print(f"步骤6：y_sep 与 y_true 的误差 = {y_error:.1e}")
    #     if y_error > 1e-3:
    #         print("→ 结论：brentq找了假根（y值严重偏离真实解）")
    #     else:
    #         print("→ 结论：y值正确（求根环节无问题）")
    '''排查完毕，确认是加根问题，加入了动态区间尝试解决
    分离变量数值解 与 RK45 数值解 的 RMSE = 597.8759315082
分离变量数值解 与 解析解 的 最大绝对误差 = 1076.2258498631
分离变量数值解 与 解析解 的 平均绝对误差 = 526.9810394205
测试，无效'''

    print('-'*50)

    '''调用可视化函数，把 xs、ys_sep、ys_rk 以及解析解（如果有）传入，生成对比曲线和误差数据（RMSE、最大误差等）。'''
    # 可视化并输出误差
    visualize_and_report(xs, ys_sep, ys_rk, example=EXAMPLE, analytic_solution=analytic_solution)
    end_time = time.time()
    print(end_time - start_time)
if __name__ == "__main__":
    main()