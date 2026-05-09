import numpy as np

N = 10000 #경사하강법 반복횟수
alpha = 0.00001 #학습률
alt = np.array([1, 2, 3]) #해발고도 리스트
grad = np.array([3, 7, 5]) #경사 리스트(각 x, dy/dx 기울기o)
fatigue = np.array([1, 4, 9]) #피로도 리스트
v = np.array([3, 38, 15]) #속도 리스트, 얘네들 싹다 손으로 작성...?

# 정규화 [0,1] 사이
altnn = np.array([np.min(alt), np.max(alt)-np.min(alt)])
gradnn = np.array([np.min(grad), np.max(grad)-np.min(grad)])
fatiguenn = np.array([np.min(fatigue), np.max(fatigue)-np.min(fatigue)])
vnn = np.array([np.min(v), np.max(v)-np.min(v)])

altN = (alt - altnn[0]) / altnn[1]
gradN = (grad - gradnn[0]) / gradnn[1]
fatigueN = (fatigue - fatiguenn[0]) / fatiguenn[1]
vN = (v - vnn[0]) / vnn[1]

#지수함수 회귀
exponentialC = np.random.normal(0, 1, 16)

def exponential(a, g, f):
    a, g, f = (a - altnn[0])/altnn[1], (g - gradnn[0])/gradnn[1], (f - fatiguenn[0])/fatiguenn[1]
    return (np.dot(exponentialC[0:4],[a, g, f, 1])*np.exp(exponentialC[4]*a)\
    + np.dot(exponentialC[5:9],[a, g, f, 1])*np.exp(exponentialC[9]*g)\
    + np.dot(exponentialC[10:14],[a, g, f, 1])*np.exp(exponentialC[14]*f)\
    + exponentialC[15])


for _ in range(N):
    a = altN
    g = gradN
    f = fatigueN
    ea = np.exp(exponentialC[4]*a)
    eg = np.exp(exponentialC[9]*g)
    ef = np.exp(exponentialC[14]*f)
    DelCP = [
        a*ea,
        g*ea,
        f*ea,
        ea,
        a*(exponentialC[0]*a + exponentialC[1]*g + exponentialC[2]*f + exponentialC[3])*ea,
        a*eg,
        g*eg,
        f*eg,
        eg,
        g*(exponentialC[5]*a + exponentialC[6]*g + exponentialC[7]*f + exponentialC[8])*eg,
        a*ef,
        g*ef,
        f*ef,
        ef,
        f*(exponentialC[10]*a + exponentialC[11]*g + exponentialC[12]*f + exponentialC[13])*ef,
        np.ones_like(alt)
    ]
    deviation = np.array([exponential(a[i], g[i], f[i]) - vN[i] for i in range(len(a))])
    DelCL = np.array([np.dot(deviation, DelCP[i]) for i in range(16)])
    exponentialC -= alpha * DelCL

print(exponential(2, 7, 4)*vnn[1]+vnn[0])