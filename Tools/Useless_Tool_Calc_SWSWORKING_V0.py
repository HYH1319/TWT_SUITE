import math as m
import Vpc_calc as V

U=23000;I=0.090;
f0=352*1e9*6.28;
yita=1.76e11;erb=8.85e-12;pi=3.1415926;
Ve=m.sqrt(2*yita*U);
t=0.026e-3;
w=1*t;
S=w*t*pi/4;
Vec=V.Vpc_calc(U);

fe=m.sqrt(yita*I/(S*Ve*erb));
V_0=Vec/(1+fe/f0)
print('The Working point Velocity of Vp is %.4f c' %(V_0)) 


for U in range(8000,24000,100):
 Vpc=V.Vpc_calc(U)
 print(f"The Vec of Voltage {U} is {Vpc}")