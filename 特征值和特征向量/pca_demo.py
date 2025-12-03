import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(0)
X = np.random.randn(100, 2) @ [[3, 1], [1, 2]]  # 椭圆数据

pca = PCA(n_components=1)
X1 = pca.fit_transform(X)
X_rec = pca.inverse_transform(X1)

plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.5, label="origin")
plt.scatter(X_rec[:, 0], X_rec[:, 1], s=20, alpha=0.5, label="1D PCA")
plt.legend()
plt.savefig(
    "pca_demo.png", dpi=75, bbox_inches="tight", pad_inches=0.1, facecolor="black"
)
plt.close()

print(pca.components_, pca.explained_variance_)
