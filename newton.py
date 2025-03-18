import numpy as np
from scipy.differentiate import derivative

def newton(func, x0, fprime=None, tol=1.48e-08, maxiter=50):
    x = x0
    
    for _ in range(maxiter):
        fx = func(x)
        if np.all(np.abs(fx) < tol):
            return x
        
        dfx = fprime(x) if fprime else derivative(func,x).df
        #dfx[dfx == 0] = np.nan  
        
        x_new = x - fx / dfx
        if np.all(np.abs(x_new - x) < tol):
            return x_new
        
        x = x_new 
    
    return x
