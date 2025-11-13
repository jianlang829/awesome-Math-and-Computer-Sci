#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正并稳健地用 sympy 求解常微分方程 dy/dx = x / y，
并用初始条件 y(0)=1 筛选出满足条件的特解。

解析结果为 y^2 = x^2 + C -> y = ±sqrt(x^2 + C)
"""

import sympy as sp

def main():
    x = sp.symbols('x')
    y = sp.Function('y')  # 保留为可被调用的函数类：y(x)

    # 构造微分方程 dy/dx = x / y
    eq = sp.Eq(sp.Derivative(y(x), x), x / y(x))

    # 求通解（dsolve 可能返回 Eq(...) 或 [Eq(...), ...]）
    general_sol = sp.dsolve(eq)
    print("通解（dsolve 原始返回）：", general_sol)

    # 规范为列表，便于统一处理
    sols = general_sol if isinstance(general_sol, list) else [general_sol]

    # 初始条件
    x0 = 0
    y0 = 1

    C = sp.symbols('C1')  # 预期 sympy 使用的常数名通常为 C1
    particular_solutions = []

    for sol in sols:
        '''这段代码的核心功能是：**从微分方程的通解列表中，筛选并求解出满足指定初始条件 \( y(x_0) = y_0 \) 的特解**，尤其适用于通解包含多个分支（如正负号分支）或积分常数的场景。


### 实现逻辑拆解：

#### 1. 遍历通解列表（`for sol in sols`）
- `sols` 是微分方程的通解列表（如 `[Eq(y(x), sqrt(C1 + x²)), Eq(y(x), -sqrt(C1 + x²))]`）。
- 每次循环处理一个通解分支（如先处理正号分支，再处理负号分支）。


#### 2. 提取通解表达式并分析常数（以单个通解 `sol` 为例）
- `rhs = sol.rhs`：提取通解右侧的表达式（如 `sqrt(C1 + x²)` 或 `-sqrt(C1 + x²)`）。
- `consts = list(rhs.free_symbols - {x})`：  
  找出表达式中除了自变量 `x` 之外的自由符号（通常是积分常数，如 `C1`），作为需要求解的常数候选。


#### 3. 处理无常数的特殊情况
- 若 `consts` 为空（通解中没有积分常数，非常规情况）：  
  直接验证该表达式是否满足初始条件（代入 `x=x0` 后是否等于 `y0`）。若满足，则加入特解列表；否则跳过。


#### 4. 求解积分常数（针对有常数的常规情况）
- 取第一个常数 `const = consts[0]`（通常通解中只有一个积分常数，如 `C1`）。
- 构建方程并求解：  
  `c_candidates = sp.solve(sp.Eq(rhs.subs(x, x0), y0), const)`  
  即把 `x=x0` 代入通解表达式，令其等于 `y0`，求解常数 `const` 的可能值（返回候选解列表）。


#### 5. 验证候选常数，筛选有效特解
- 对每个候选常数 `c_val`：  
  - 代入通解表达式，得到 `rhs.subs(const, c_val)`（替换常数后的具体解）。  
  - 验证：代入 `x=x0` 后是否真的等于 `y0`（用 `sp.simplify` 处理符号计算，避免形式误差）。  
  - 若验证通过，说明该候选值对应的解是满足初始条件的特解，加入结果列表；否则跳过（例如负号分支中 `y0` 为正时的矛盾解）。


### 总结
这段代码通过**遍历通解分支→提取常数→求解常数→验证解的有效性**四步，精准筛选出符合初始条件的特解，尤其能处理通解包含正负分支、符号计算可能产生假解的场景（如负号分支代入正初始值时的矛盾），确保特解的正确性。'''
        rhs = sol.rhs  # 形如 sqrt(C1 + x**2) 或 -sqrt(C1 + x**2)
        # 找出除 x 以外的自由符号，作为候选常数
        consts = list(rhs.free_symbols - {x})

        if not consts:
            # 如果没有常数（非常规情况），测试是否直接满足初始条件
            if sp.simplify(rhs.subs(x, x0) - y0) == 0:
                particular_solutions.append(sp.Eq(y(x), rhs))
            else:
                # 不满足初始条件
                pass
            continue

        # 目前只考虑第一个常数（通常就是 C1）
        const = consts[0]

        # 解方程 rhs(x0) = y0 求常数值（可能返回空列表）
        c_candidates = sp.solve(sp.Eq(rhs.subs(x, x0), y0), const)

        # 逐个候选值验证（避免像 -sqrt(C1)=1 这类形式上的假解）
        for c_val in c_candidates:
            # 验证代入后在 x0 处是否等于 y0
            test_val = sp.simplify(rhs.subs(const, c_val).subs(x, x0))
            # 用 simplify 比较，兼容符号表达式
            if sp.simplify(test_val - y0) == 0:
                particular_solutions.append(sp.Eq(y(x), rhs.subs(const, c_val)))
            else:
                # 如果不相等，说明该候选在符号上满足方程但不满足指定分支的初始值（例如负号冲突）
                pass

    if particular_solutions:
        print("满足初始条件 y(0)=1 的特解有：")
        for ps in particular_solutions:
            print("  ", sp.simplify(ps))
    else:
        print("没有找到满足初始条件的特解（可能初始条件矛盾或需要考虑分段/非常规解）")

    # 备用：直接给出解析的显式通解形式，便于理解
    # print("\n解析通解（手工推导）： y = ±sqrt(x**2 + C)")
    # print("由初始条件 y(0)=1 可得到 C = 1，且因为 y(0)=1 为正，取正号分支：")
    # print("特解： y = +sqrt(x**2 + 1)")

if __name__ == "__main__":
    main()