def lagrange3(p1, p2, p3):
    """输入三个点 (x1,y1)(x2,y2)(x3,y3)，返回二次函数系数 a,b,c 使 f(x)=ax²+bx+c"""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    # 基函数分母
    d1 = (x1 - x2) * (x1 - x3)
    d2 = (x2 - x1) * (x2 - x3)
    d3 = (x3 - x1) * (x3 - x2)
    # 合并同类项得系数
    a = y1 / d1 + y2 / d2 + y3 / d3
    b = -y1 * (x2 + x3) / d1 - y2 * (x1 + x3) / d2 - y3 * (x1 + x2) / d3
    c = y1 * x2 * x3 / d1 + y2 * x1 * x3 / d2 + y3 * x1 * x2 / d3
    return a, b, c


# 例：题目给的 (1,4)(2,9)(3,16)
a, b, c = lagrange3((1, 4), (2, 9), (3, 16))
print(f"f(x) = {a:.1f}x² + {b:.1f}x + {c:.1f}")
# 验证
print(f"f(1)={a*1**2+b*1+c}, f(2)={a*2**2+b*2+c}, f(3)={a*3**2+b*3+c}")
