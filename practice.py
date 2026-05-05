import numpy as np

N = 100 #경사하강법 반복횟수
alpha = 0.1 #학습률
alt = np.array([1, 2, 3]) #해발고도 리스트
grad = np.array([4, 5, 6]) #경사 리스트(각 x, dy/dx 기울기o)
v = np.array([7, 8, 9]) #속도 리스트, 얘네들 싹다 손으로 작성...?
polynomialC = np.random.normal(0, 1, 6)
polynomialV = [alt**2, alt*grad, grad**2, alt, grad, np.ones(len(alt))]

def polynomial(a, g):
    return np.dot(polynomialC, [a*a, a*g, g*g, a, g, 1])

for _ in range(N): 
    deviation = np.array([polynomial(alt[i], grad[i]) - v[i] for i in range(len(alt))])
    DelCL = np.array([np.dot(deviation, polynomialV[i]) for i in range(6)])
    polynomialC -= alpha * DelCL

print(polynomialC)


