import numpy as np


def polynomial_kernel(ages, T, s, alpha):
    ind = (ages > alpha) * (ages < alpha + s)
    ages = ages[ind]
    n = len(ages)
    
    a = (60*(2*alpha-2*T+s))/((s**5)*(s**2+5*s*(alpha-T)+5*(alpha**2-2*T*alpha+T**2)))
    
    eps = np.finfo(float).eps
    lowAlphaBound = T - s / 2 - 10 * eps
    upperAlphaBound = T - s / 2 + 10 * eps
    if alpha > lowAlphaBound and alpha < upperAlphaBound:
        b = 30/s**5
    else:
        b = a*(5*T**2+2*T*alpha+T*s-7*alpha**2-7*alpha*s-2*s**2) / (4*alpha-4*T+2*s)

    c = a*(-3 * alpha ** 2 - 3 * alpha * s - s ** 2) + b * (-2 * alpha - s)
    d = -a*alpha**3-b*alpha**2-c*alpha
    
    w = (ages - alpha) * (ages - (alpha + s)) * (a*ages**3 + b*ages**2 + c*ages + d)
    w = w / sum(w)
    
    bias=abs(sum(w*ages)-T)

    return w, ind, bias, n
