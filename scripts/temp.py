import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

# 设置 von Mises 分布的参数
kappa = 1/0.17**2  # 集中度参数，类似于高斯分布的 1/σ^2
mean = 0   # 均值

# 生成角度数据
initial_input = np.linspace(0, 2*np.pi, 361)  # 从 0 到 2π，包含 361 个点

# 生成 von Mises 分布的数据
y = vonmises.pdf(initial_input, kappa, loc=mean)

# 标准化 y
y /= np.sum(y)

# 绘制 von Mises 分布曲线
plt.plot(initial_input, y, label='Von Mises Distribution')
plt.title('Normalized Von Mises Distribution')
plt.xlabel('Angle (radians)')
plt.ylabel('Normalized Probability Density')
plt.grid(True)
plt.legend()
plt.show()