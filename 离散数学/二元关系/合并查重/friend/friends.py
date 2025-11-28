#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
R = pd.read_csv(DATA_DIR / "R.csv")
S = pd.read_csv(DATA_DIR / "S.csv")

# 转成 (user, friend) 元组的集合，去掉可能重复行
R_set = set(map(tuple, R.itertuples(index=False, name=None)))
S_set = set(map(tuple, S.itertuples(index=False, name=None)))

# 关系运算
common = R_set & S_set  # 交 ∩
all_friends = R_set | S_set  # 并 ∪
# 补：全集先构造，再减
max_user = 5  # 与数据里最大 id 一致即可
universe = set(
    (u, f) for u in range(1, max_user + 1) for f in range(1, max_user + 1) if u != f
)
non_friends = universe - R_set  # 补 ¬R


# 结果转回 DataFrame，方便看
def to_df(s):
    return pd.DataFrame(sorted(s), columns=["user", "friend"])


if __name__ == "__main__":
    print("共同好友 (R ∩ S):")
    print(to_df(common), "\n")

    print("所有好友 (R ∪ S):")
    print(to_df(all_friends), "\n")

    print("R 的非好友 (¬R):")
    print(to_df(non_friends))
