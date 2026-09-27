import matplotlib.pyplot as plt
from scipy.special import iv, kv
import math as m
import numpy as np
from TWT_CORE_SIMP import calc_Rn_sqr_values
# 把你实际仿真时传入的真实参数填在这里：
beta_space = 15232.136205940269  # 你的实际值
r_beam = 0.00001       # 你的实际值
Fill_Rate = 1    # 你的实际值
Space_cut = 100 # 故意取大一点看看

p_SWS=0

# a = r_beam * Fill_Rate
n_values = np.arange(1, Space_cut)
# beta_n_values = beta_space * n_values

# Iv0 = iv(0, beta_n_values * a)
# Iv1 = iv(1, beta_n_values * r_beam)
# Kv0 = kv(0, beta_n_values * a)
# Kv1 = kv(1, beta_n_values * r_beam)

# Bes = Iv1 * Kv0 + Iv0 * Kv1
Rn_sqr = calc_Rn_sqr_values(beta_space, Space_cut, p_SWS, r_beam, Fill_Rate)
print(Rn_sqr)

# 画图看收敛性
plt.figure(figsize=(10, 4))
plt.plot(n_values, Rn_sqr, 'b-o', markersize=3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel(' n')
plt.ylabel('Rn^2')
plt.title('Rn-n')
plt.grid(True)
plt.show()
