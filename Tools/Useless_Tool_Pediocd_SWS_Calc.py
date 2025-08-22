
import math as ma

# 输入区域
f=180*1e9
U=23000
# 计算区域
w=2*ma.pi*f;yita=1.76*1e11;erb=8.85e-12
Ve=ma.sqrt(2*yita*U);me=9.11*1e-31

beta_e=w/Ve
p=2.0*ma.pi/beta_e


print('The Perdioc of SWS is %.3f mm' %(p*1e3))