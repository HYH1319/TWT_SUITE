import math as m
import matplotlib.pyplot as plt


def Vpc_calc(U):
    gamma=U/5.11e5;
    Vp_c=1*m.pow(1-1/m.pow(1+gamma,2),0.5);

    return Vp_c


if __name__ == "__main__":
    for U in range(22000,23000,100):
        Vp_c=Vpc_calc(U)

    print('The phase of Velocity is %.6f c,which U is %.1f V' %(Vp_c,U))