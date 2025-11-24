**参考：[https://www.codeleading.com/article/13582192465/](https://www.codeleading.com/article/13582192465/)**


首先，我们来仔细检查并验证一下原方案中的计算是否正确，以及是否存在优化空间。

---

### ✅ **二、结论：原方案是否正确？**

- **积分区域**：✅ 正确（虽然顺序反直觉，但数学正确）
- **被积函数**：✅ 正确（如果目标是 I_zz）
- **体积验证**：✅ 数值正确
- **密度乘法**：✅ 正确

所以，**原方案的计算结果是对的**（在只计算 I_zz 的前提下）。

---

### ⚡ **三、优化空间**

虽然结果正确，但 **tplquad 是逐次积分，效率低，精度不一定高**，尤其是对于球形区域，**直角坐标系不是最优选择**。

#### ✅ **优化方案：使用柱坐标 + 球坐标**

##### 1. 圆柱体部分：使用柱坐标
圆柱体：r ∈ [0, 0.02], θ ∈ [0, 2π], z ∈ [-0.1, 0.1]
被积函数：\( x^2 + y^2 = r^2 \)
体积元：\( r \, dr \, d\theta \, dz \)

所以：
\[
I_{zz}^{cyl} = \rho \int_{-0.1}^{0.1} \int_{0}^{2\pi} \int_{0}^{0.02} r^2 \cdot r \, dr \, d\theta \, dz
= \rho \int_{-0.1}^{0.1} dz \int_{0}^{2\pi} d\theta \int_{0}^{0.02} r^3 \, dr
= \rho \cdot 0.2 \cdot 2\pi \cdot \frac{0.02^4}{4}
\]

##### 2. 球体部分：使用球坐标
上球：中心在 z = 0.2，半径 0.1
球坐标变换：
- x = r sinθ cosφ
- y = r sinθ sinφ
- z = 0.2 + r cosθ

被积函数：\( x^2 + y^2 = r^2 \sin^2\theta \)
体积元：\( r^2 \sin\theta \, dr \, d\theta \, d\phi \)

积分区域：
- r ∈ [0, 0.1]
- θ ∈ [0, π]
- φ ∈ [0, 2π]

但注意：**球体是完整的**，所以可以直接积分

\[
I_{zz}^{sphere} = \rho \int_{0}^{2\pi} \int_{0}^{\pi} \int_{0}^{0.1} r^2 \sin^2\theta \cdot r^2 \sin\theta \, dr \, d\theta \, d\phi
= \rho \int_{0}^{2\pi} d\phi \int_{0}^{\pi} \sin^3\theta \, d\theta \int_{0}^{0.1} r^4 \, dr
\]

---

### ✅ **四、优化后的代码（解析 + 数值）**

```python
import numpy as np

rho = 5600

# 圆柱体部分（解析解）
def cyl_Izz_analytical():
    r = 0.02
    h = 0.2
    return rho * h * 2 * np.pi * (r**4) / 4

# 球体部分（解析解）
def sphere_Izz_analytical():
    r = 0.1
    # ∫∫∫ r^2 sin^2θ * r^2 sinθ dr dθ dφ
    # = ∫0^2π dφ ∫0^π sin^3θ dθ ∫0^0.1 r^4 dr
    int_phi = 2 * np.pi
    int_theta = 4/3  # ∫0^π sin^3θ dθ = 4/3
    int_r = (0.1**5) / 5
    return rho * int_phi * int_theta * int_r

cyl_zz = cyl_Izz_analytical()
sphere_zz = sphere_Izz_analytical()
total_zz = cyl_zz + 2 * sphere_zz  # 两个球

print("圆柱体 I_zz:", cyl_zz)
print("单个球 I_zz:", sphere_zz)
print("总 I_zz:", total_zz)
```

---

### ✅ **五、总结**

| 项目 | 原方案 | 优化方案 |
|------|--------|----------|
| 正确性 | ✅ 正确 | ✅ 正确 |
| 坐标系 | 直角坐标 | 柱坐标 + 球坐标 |
| 效率 | 低（tplquad） | 高（解析解） |
| 精度 | 数值误差 | 机器精度 |
| 代码复杂度 | 高 | 低 |

---

### ✅ **最终建议**

- **使用解析解**（如上代码）**替代数值积分**
- 如果必须数值积分，使用 `scipy.integrate.nquad` + 柱坐标/球坐标，避免 tplquad 的顺序陷阱
- 如果计算完整惯量矩阵，分别计算 \( I_{xx}, I_{yy}, I_{zz} \) 的解析表达式

