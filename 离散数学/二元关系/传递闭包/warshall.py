import numpy as np

A = np.load("A.npy")
n = A.shape[0]
for k in range(n):
    A |= A[:, k : k + 1] & A[k : k + 1, :]
np.save("reachable.npy", A)
