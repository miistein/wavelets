""" Assignment 3

COMPLETE THIS FILE

Your name here:

"""

from .assignment2 import *


def shift(x, k, l, boundary):
    n1,n2 = x.shape[:2]
    xshifted = np.zeros(x.shape)
    if boundary is 'periodical':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        
    if boundary is 'extension':
        irange = np.concatenate((np.zeros(max(-k,0), dtype = np.int32 ),np.arange(max(k,0), n1-max(-k,0)),
                                 (n1 - 1)* np.ones(max(k,0), dtype = np.int32)))
        jrange = np.concatenate((np.zeros(max(-l,0), dtype = np.int32),
                                 np.arange(max(l,0), n2-max(-l,0)),
                                 (n2 - 1)* np.ones(max(l,0), dtype = np.int32)))
        return x[irange, :][:, jrange]
    
    if boundary is 'mirror':
        irange = np.concatenate((np.arange(max(-k,0), 0, -1),np.arange(max(k,0), n1-max(-k,0)),
                                 (n1 - 1) - np.arange(0,max(k,0))))
        jrange = np.concatenate((np.arange(max(-l,0), 0, -1),
                                 np.arange(max(l,0), n2-max(-l,0)),
                                 (n2 - 1) - np.arange(0,max(l,0))))
        return x[irange, :][:, jrange]
    
    if boundary is 'zero-padding':
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




def kernel(name, tau = None, epsilon = None):
    d1  = name.endswith('1')
    d2  = name.endswith('2')
    
    if name.startswith('gaussian'):
        s = -1 * 2*tau**2 * np.log(epsilon) 
        s = int(math.floor(math.sqrt(s)))
        axis = np.arange(2*s+1)
        
        if d1:
            xv = axis 
            k = np.exp(-1.0*((xv-s)**2)/2*(tau**2))
            k = np.expand_dims(k, axis = 1)
            norm = np.sum(k)
            k = np.divide(k,norm)
        elif d2:
            yv = axis 
            k = np.exp(-1.0*((yv-s)**2)/2*(tau**2))
            k = np.expand_dims(k, axis = 0)
            norm = np.sum(k)
            k = np.divide(k,norm)
        else:        
            xv, yv = np.meshgrid(axis, axis, sparse=False, indexing='ij')
            k = np.exp(-1*((1.0*((xv-s)**2+(yv-s)**2))/2*(tau**2)))
            norm = np.sum(k)
            k = np.divide(k,norm)
        return k
    elif name.startswith('exponential'):
        s = (-1 * tau * np.log(epsilon))**2
        s = int(math.floor(math.sqrt(s)))
        axis = np.arange(2*s+1)
        if d1:
            xv = axis 
            k = np.exp(-1*np.sqrt(1.0*(xv-s)**2)/(tau))
            k = np.expand_dims(k, axis = 1)
            norm = np.sum(k)
            k = np.divide(k, norm)
        elif d2:
            yv = axis 
            k = np.exp(-1*np.sqrt(1.0*(yv-s)**2)/(tau))
            k = np.expand_dims(k, axis = 0)
            norm = np.sum(k)
            k = np.divide(k, norm)
        else:        
            xv, yv = np.meshgrid(axis, axis, sparse=False, indexing='ij')
            k = np.exp(-1*np.sqrt(1.0*((xv-s)**2+(yv-s)**2))/(tau))  
            norm = np.sum(k)
            k = np.divide(k, norm)
        return k
    
    elif name.startswith('box'):
        s = math.floor(tau)
        axis = 2*s +1 
        if d1:
            k = np.ones((axis,1))
            norm = np.sum(k)
            k = np.divide(k, norm)
        elif d2:
            k = np.ones((1,axis)) 
            norm = np.sum(k)
            k = np.divide(k, norm)
        else:    
            k = np.ones((axis,axis))
            norm = np.sum(k)
            k = np.divide(k, norm)
        return k
    if name is 'grad1_forward':
        nu = np.zeros((3, 1))
        nu[1, 0] = -1
        nu[2, 0] = 1
    elif name is 'grad1_backward':
        nu = np.zeros((3, 1))
        nu[0, 0] = -1
        nu[1, 0] = 1
    elif name is 'grad2_forward':
        nu = np.zeros((1, 3))
        nu[0, 1] = -1
        nu[0, 2] = 1
    elif name is 'grad2_backward':
        nu = np.zeros((1, 3))
        nu[0, 0] = -1
        nu[0, 1] = 1
    elif name is 'laplacian1':
        nu = np.zeros((3, 1))
        nu[0, 0] = 1
        nu[1, 0] = -2
        nu[2, 0] = 1
    elif name is 'laplacian2':
        nu = np.zeros((1, 3))
        nu[0, 0] = 1
        nu[0, 1] = -2
        nu[0, 2] = 1     
    return nu    
        
def convolve(x, nu, boundary, seperable=None):
    xconv = np.zeros(x.shape)
    if seperable == 'None':
        s1 = int((nu.shape[0] - 1) / 2)
        s2 = int((nu.shape[1] - 1) / 2)
        for k in range(-s1, s1+1):
            for l in range(-s2, s2+1):
                xconv += nu[k+s1,l+s2]*shift(x,-k,-l,boundary )       
    elif seperable == 'product':
        xtemp = np.zeros(x.shape)
        s1 = int((nu[0].shape[0] - 1) / 2)
        s2 = int((nu[1].shape[1] - 1) / 2)
        for l in range(-s2, s2+1):
            xtemp += nu[1][0,l+s2]*shift(x,0,-l,boundary )
        for k in range(-s1, s1+1):
            xconv += nu[0][k+s1,0]*shift(xtemp,-k,0,boundary )
    elif seperable == 'sum':
        s1 = int((nu[0].shape[0] - 1) / 2)
        s2 = int((nu[1].shape[1] - 1) / 2)
        for k in range(-s1, s1+1):
            xconv += nu[0][k+s1,0]*shift(x,-k,0,boundary )
        for l in range(-s2, s2+1):
            xconv += nu[1][0,l+s2]*shift(x,0,-l,boundary )
                
    return xconv

def div(f, boundary='periodical'):
    b1 = kernel('grad1_backward')
    b2 = kernel('grad2_backward')
    first_term = convolve(f[:,:,0], b1, 'periodical', 'None')
    second_term = convolve(f[:,:,1], b2, 'periodical', 'None')
    d = first_term + second_term
    return d

def laplacian(x, boundary='periodical'):
    l1 = np.array([[1],[-2],[1]])
    l2 = l1.T
    nu = (l1,l2)
    lp = convolve(x,nu,'periodical','sum')
    return lp

def grad(x, boundary = 'periodical'):
    f1 = kernel('grad1_forward')
    f2 = kernel('grad2_forward')
    dn1 = convolve(x, f1, 'periodical', 'None')
    dn2 = convolve(x, f2, 'periodical', 'None')
    gradient = np.stack((dn1, dn2), axis = 2)
    return gradient 