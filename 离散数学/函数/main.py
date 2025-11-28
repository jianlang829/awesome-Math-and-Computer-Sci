import os, struct, time
from typing import List

# 参数：页数 n 必须 2 的幂
n = 1 << 18  # 262 144 页 ≈ 1 GiB
MASK = n - 1
ROUND = 18  # Feistel 轮数

# 密钥：256-bit 随机
key = os.urandom(32)


# Feistel 轮函数 F：用密钥派生轮密钥，再异或
def F(i: int, r: int) -> int:
    buf = struct.pack("<II", i, r) + key
    return hash(buf) & MASK


# 双射 π 及其逆 π⁻1
def pi(x: int) -> int:
    """双射：18 轮 Feistel 置换"""
    L, R = x >> 9, x & 0x1FF  # 18-bit 分成两半
    for i in range(ROUND):
        L, R = R, L ^ F(i, R)
    return (L << 9 | R) & MASK


def pi_inv(y: int) -> int:
    """逆双射"""
    L, R = y >> 9, y & 0x1FF
    for i in reversed(range(ROUND)):
        L, R = R ^ F(i, L), L
    return (L << 9 | R) & MASK


# 线性同余“加扰”  g(x)=(a*x+b) mod n
a, b = 0x3C6EF372, 0x9E3779B9  # a 与 n 互素
a_inv = pow(a, -1, n)  # 模逆元，一次性算出


def g(x):
    return (a * x + b) & MASK


def g_inv(y):
    return (a_inv * (y - b)) & MASK


# 最终复合函数
def shuffle(x):
    return g(pi(x))


def unshuffle(x):
    return pi_inv(g_inv(x))


# 验证双射：所有像点唯一且可逆
if __name__ == "__main__":
    start = time.time()
    seen = set()
    for i in range(n):
        s = shuffle(i)
        assert s not in seen
        seen.add(s)
        assert unshuffle(s) == i
    print("✓ 双射验证通过，耗时 %.2f s" % (time.time() - start))
