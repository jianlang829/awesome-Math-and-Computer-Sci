parent = {}  # 绳子结


def find(x):  # 找鸭王
    if parent.setdefault(x, x) != x:
        parent[x] = find(parent[x])
    return parent[x]


def union(a, b):  # 打结
    parent[find(b)] = find(a)


# 1. 读“a,b是同一人”
for line in open("equiv.csv"):  # 格式：账号A,账号B
    a, b = line.strip().split(",")
    union(a, b)

# 2. 分堆输出
import collections

buckets = collections.defaultdict(list)
for x in parent:
    buckets[find(x)].append(x)
with open("user_bucket.csv", "w") as f:
    for root, members in buckets.items():
        f.write(f"{root}\t{','.join(members)}\n")
