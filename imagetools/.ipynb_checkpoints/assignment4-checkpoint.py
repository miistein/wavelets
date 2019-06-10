""" Assignment 4

COMPLETE THIS FILE

Your name here:

"""

from .assignment3 import *

def bilateral_naive(y, sig, s1=2, s2=2, h=1):
    import math
    
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1

    # define kernel function
    def phi(alpha):        
        variance = sig**2
        num = max(alpha - 2 * h * variance,0)
        denom = 2*math.sqrt(2)*h* variance / math.sqrt(c)
        return math.exp(-num/denom)
    
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            tempx = tempz = 0
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    dist2 = ((y[i + k, j + l] - y[i, j])**2).mean()
                    # complete
                    tempz += phi(dist2)
                    tempx += phi(dist2) * y[i + k, j + l]
            x[i,j] = tempx
            Z[i,j] = tempz
                    
    Z[Z == 0] = 1
    x = x / Z
    
    return x

def bilateral(y, sig, s1=10, s2=10, h=1,boundary='periodical'):
    import math
    
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1

    # define kernel function
    def phi(alpha):        
        variance = sig**2
        num = alpha
        num[(alpha - 2 * h * variance) < 0] = 0
        denom = 2*math.sqrt(2)*h* variance / math.sqrt(c)
        return np.exp(-num/denom)
    
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            shifted_y = shift(y, -k, -l, boundary)
            dist2 = np.square(shifted_y - y)
            if y.ndim==3:
                dist2 = dist2.mean(axis=y.ndim-1)
                x += np.repeat(phi(dist2)[:, :, np.newaxis], 3, axis=2) * shifted_y
                Z += phi(dist2)[:, :, np.newaxis]
            else:
                x += phi(dist2) * shifted_y
                Z += phi(dist2)
                    
    Z[Z == 0] = 1
    x = x / Z
    
    return x

def nlmeans_naive(y, sig, s1=2, s2=2, p1=1, p2=1, h=1):
    import math
    
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    p = (2*p1 + 1)*(2*p2+1)

    # define kernel function
    def phi(alpha):        
        variance = sig**2
        num = max(alpha - 2 * h * variance,0)
        denom = 2*math.sqrt(2)*h* variance / math.sqrt(c*p)
        return math.exp(-num/denom)
    
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))

    for i in range(s1, n1-s1-p1):
        for j in range(s2, n2-s2-p2):
            tempx = tempz = 0
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    dist2 = 0
                    for u in range(-p1, p1 + 1):
                        for v in range(-p2, p2 + 1):
                            # complete
                            dist2 += ((y[i + k + u, j + l + v] - y[i + u, j + v])**2).mean()
                                
                    tempx += phi(dist2/(c*p)) * y[i + k, j + l]
                    tempz += phi(dist2/(c*p))
            x[i,j] = tempx
            Z[i,j] = tempz
            
    Z[Z == 0] = 1
    x = x / Z
    
    return x

def nlmeans(y, sig, s1=7, s2=7, p1=None, p2=None, h=1, boundary='periodical'):
    p1 = (1 if y.ndim == 3 else 2) if p1 is None else p1
    p2 = (1 if y.ndim == 3 else 2) if p2 is None else p2
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    p = (2*p1 + 1) * (2*p2+1)
    
    # define kernel function
    def phi(alpha):        
        variance = sig**2
        num = np.maximum(alpha - 2 * h * variance,0)
        denom = 2*math.sqrt(2)*h* variance / math.sqrt(c*p)
        return np.exp(-num/denom)
    
    x = np.zeros(y.shape)
    Z = np.zeros(y.shape)
    
    nu = kernel('box',tau=max(p1,p2))
    
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            shifted_y = shift(y, -k, -l, boundary)   
            dist2 = convolve(np.square(shifted_y-y),nu,boundary)
            
            if y.ndim==3:
                dist2 = dist2.mean(axis=y.ndim-1)
                x += np.repeat(phi(dist2)[:, :, np.newaxis], 3, axis=2) * shifted_y
                Z += phi(dist2)[:, :, np.newaxis]
            else:
                x += phi(dist2) * shifted_y
                Z += phi(dist2)

    Z[Z == 0] = 1
    x = x / Z

    return x