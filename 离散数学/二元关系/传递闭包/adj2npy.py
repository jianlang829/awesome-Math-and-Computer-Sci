import numpy as np, json, csv

ip2id = json.load(open("ip2id.json"))
n = len(ip2id)
A = np.zeros((n, n), dtype=bool)
with open("input.csv") as f:
    for a, b in csv.reader(f):
        i, j = ip2id[a], ip2id[b]
        A[i, j] = 1
np.save("A.npy", A)
