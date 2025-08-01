
import numpy as np

def D(x):
    grad = np.concatenate( (x[1:] - x[:-1] , [0]))/2
    return grad
def Dt(x):
    div = -np.concatenate(( [x[0]], x[1:-1] - x[:-2] , [-x[-2]]))/2
    return div


def mse(I, ref):
    return np.sum((I-ref)**2)/I.size


def regression(listDistance,lam):
    dt = 0.01
    N = 1000000
    x = np.random.randn(len(listDistance))

    eps = 1 
    conv = 10e-8 #critere de convergence
    i=0

    while eps > conv and i < N:
        temp = x
        x = x - dt *2*(x - listDistance + lam*Dt(D(x)))
        eps = np.linalg.norm(temp - x)
  
        i +=1
    return x 

