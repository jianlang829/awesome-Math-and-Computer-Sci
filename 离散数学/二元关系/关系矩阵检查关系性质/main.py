#!/usr/bin/env python3
"""
极简关系性质检测器
用法:
    python reflex_sym_trans.py 0 1 0  1 1 0  0 0 1
    # 把 n×n 矩阵按行摊平传进来即可，元素用空格隔开
"""
import sys, itertools


def detect(A):
    n = len(A)
    reflex = all(A[i][i] == 1 for i in range(n))
    sym = all(A[i][j] == A[j][i] for i, j in itertools.combinations(range(n), 2))
    # 传递性：R∘R ⊆ R  即 A² 的每个 1 位在 A 中也必须是 1
    A2 = [
        [any(A[i][k] and A[k][j] for k in range(n)) for j in range(n)] for i in range(n)
    ]
    trans = all(not A2[i][j] or A[i][j] for i in range(n) for j in range(n))
    return reflex, sym, trans


if __name__ == "__main__":
    flat = list(map(int, sys.argv[1:]))
    n = int(len(flat) ** 0.5)
    if n * n != len(flat):
        sys.exit("元素个数必须是完全平方数")
    A = [flat[i * n : (i + 1) * n] for i in range(n)]
    refl, sym, trans = detect(A)
    print("关系矩阵:")
    for row in A:
        print(*row)
    print("\n自反(reflexive) :", refl)
    print("对称(symmetric) :", sym)
    print("传递(transitive):", trans)
    if refl and sym and trans:
        print("=> 这是一个等价关系！")
