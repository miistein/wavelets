""" Assignment 2

COMPLETE THIS FILE

Your name here:

"""





import math 
from .provided import *

def shift(x, k, l, boundary):
    n1,n2 = x.shape[:2]
    xshifted = np.zeros(x.shape)
    if boundary == 'periodical':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
    if boundary == 'extension':
        irange = np.concatenate((np.zeros(max(-k,0)),np.arange(max(k,0), n1-max(-k,0)),
                                 (n1 - 1)* np.ones(max(k,0))))
        jrange = np.concatenate((np.zeros(max(-l,0)),
                                 np.arange(max(l,0), n2-max(-l,0)),
                                 (n2 - 1)* np.ones(max(l,0))))
                                
        xshifted = x[irange,:][:,jrange]
        return xshifted
    elif boundary == 'zero-padding':
        if k < 0 :
            xshifted = np.roll(x,-k,axis = 0)
            xshifted[0:-k,:,:] = 0
        elif k>=0:
            xshifted = np.roll(x,-k, axis = 0)
            xshifted[n1-k:n1,:,:] = 0
        if l>=0:
            xshifted = np.roll(xshifted, -l, axis = 1)
            xshifted[:,n2-l:n2,:] = 0
        elif l<0:
            xshifted = np.roll(xshifted, -l, axis = 1)
            xshifted[:,0:-l,:] = 0
            
      
    
    return xshifted 
        
        
def kernel(name, tau, epsilon):
    if name == 'gaussian':
        s = -1 * 2*tau**2 * np.log(epsilon) 
        s = int(math.floor(math.sqrt(s)))
        axis = np.arange(2*s+1)
        xv, yv = np.meshgrid(axis, axis, sparse=False, indexing='ij')
        k = np.exp(-1.0*((xv-s)**2+(yv-s)**2)/2*(tau**2))
        norm = np.sum(k)
        k = np.divide(k,norm)
    elif name == 'exponential':
        s = (-1 * tau * np.log(epsilon))**2
        s = int(math.floor(math.sqrt(s)))
        axis = np.arange(2*s+1)  
        xv, yv = np.meshgrid(axis, axis, sparse=False, indexing='ij')
        k = np.exp(-1*(1.0*((xv-s)**2+(yv-s)**2))/2*(tau**2))
        norm = np.sum(k)
        k = np.divide(k,norm)   
    elif name == 'box':
        s = math.floor(tau)
        axis = 2*s +1 
        k = np.ones((axis,axis))
        norm = np.sum(k)
        k = np.divide(k, norm)
    return k 
        
        
        
def convolve_naive(x, nu) :
    n1, n2 = x.shape[:2]
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    xconv = np.zeros(x.shape)
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            for k in range(-s1, s1+1):
                for l in range(-s2, s2+1):
                    xconv[i,j,:] += x[i+k, j+l]*nu[k+s1, l+s2]
    return xconv


def convolve(x, nu, boundary):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv += nu[k+s1,l+s2]*shift(x,-k,-l,boundary )
    return xconv
            
            
            
            
            
            