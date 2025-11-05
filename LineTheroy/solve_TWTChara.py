from sympy import symbols, expand, sqrt, I, Eq, simplify, fraction, radsimp
import numpy as np

def solveTWT(Q_val, C_val, b_val, d_val):
    """ 数值求解四次方程根
    参数：
        Q_val, C_val, b_val, d_val : 数值参数（支持复数）
    返回：
        numpy数组 : 按numpy.roots排序的四个根
    """
    # 符号定义（每次调用独立创建避免污染）
    delta, Q, C, b, d = symbols('delta Q C b d', complex=True)
    
    # 构造表达式（完全展开并化简）
    sqrt_4QC = sqrt(4*Q*C)
    sqrt_4QC3 = sqrt(4*Q*C**3)
    
    # 前两个因子（强制通分展开）
    numerator1 = (delta*(1 - sqrt_4QC3) + I*sqrt_4QC)
    numerator2 = (delta*(1 + sqrt_4QC3) - I*sqrt_4QC)
    left_part1 = numerator1 * numerator2
    
    # 后两个因子（直接展开）
    left_part2 = (delta + I*b + d) * (delta - I*b - d - 2*I/C)
    
    # 左边整体表达式
    left_total = expand(left_part1 * left_part2)
    
    # 右边表达式（处理分母）
    right_numerator = (delta - I/C)**2 * 2*C*(1 + C*b - I*C*d)
    right_denominator = 1 - 4*Q*C**3
    equation = Eq(left_total * right_denominator, right_numerator)
    
    # 获取多项式标准形式
    poly = expand(equation.lhs - equation.rhs)
    poly = poly.collect(delta)
    
    # 提取系数并代入数值
    coeffs_expr = poly.as_poly(delta).all_coeffs()
    subs_dict = {Q: Q_val, C: C_val, b: b_val, d: d_val}
    coeffs = [complex(coeff.subs(subs_dict)) for coeff in coeffs_expr]
    
    return np.roots(coeffs)

# 验证示例
if __name__ == "__main__":
    # 测试参数（确保分母不为零）
    roots = solveTWT(0, 0.1, 0, 0)
    sorted_roots = sorted(roots, key=lambda x: x.real, reverse=True)
    print(sorted_roots)
    