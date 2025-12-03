import numpy as np

A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])
x = np.linalg.solve(A, b)
print(x)  # 输出: [2. 1.]
