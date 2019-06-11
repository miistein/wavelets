""" Assignment 5

COMPLETE THIS FILE

Your name here: A12346613

"""

from .assignment4 import *
import numpy.fft as npf
import itertools

def convolvefft(x, lbd):  
    result = np.zeros(x.shape)
    kernel = np.zeros(x.shape)
    # if lambda kernel is 2D, extend to 3D
    if lbd.ndim==2 and x.ndim==3:
        kernel = np.broadcast_to(lbd[...,None],lbd.shape + (3,))
    elif lbd.ndim==2 and x.ndim==2:
        kernel = lbd
        
    result = np.real(npf.ifft2(npf.fft2(x,axes=(0,1)) * kernel,axes=(0,1)))
    
    return result

def kernel(name, tau=1, eps=1e-3):
    def isvalid(i,j,f):
        Z = 0
        for ix in range(-i,i+1):
            for iy in range(-j,j+1):
                if f(ix,iy) <= eps:
                    return False,-1

                Z += f(ix,iy)

        return True,Z
    
    def find_parameters(f):
        start_idx = 100
            
        coords = []
        x = tuple(range(start_idx+1))
        for i,j in itertools.product(x,x):
            valid,Z = isvalid(i,j,f)
            if valid:
                coords.append((i,j,Z,f(i,j)))
        
        s1,_,Z,_ = sorted(coords,key=lambda x:x[0])[-1]
        _,s2,_,_ = sorted(coords,key=lambda x:x[1])[-1]
        
        return s1,s2,Z
    
    def get_f():
        if name.startswith('gaussian'):
            def f(i,j): 
                denom = 2*(tau**2)
                num = math.hypot(i,j)**2
                return math.exp(-num/denom)  
        elif name.startswith('exponential'):
            def f(i,j): 
                denom = tau
                num = math.hypot(i,j)
                return math.exp(-num/denom)      
        elif name.startswith('box'):
            def f(i,j):
                return 1 if max(abs(i),abs(j)) <= tau else 0
        else:
            raise KeyWordError

        return f
    
    def get_nu():
        dfilter = ('grad' in name) or ('laplacian' in name)

        if dfilter:
            def define_kernel(*seq,dim=1):
                # helper function to define kernel. 
                assert(len(seq)==3)
                nu = np.zeros((3,dim))
                for i,_ in enumerate(seq):
                    nu[i,dim-1] = seq[i]

                return nu
                
            if "grad1_forward" in name: 
                nu = define_kernel(0,-1,1,dim=1)
            elif "grad1_backward" in name:
                nu = define_kernel(-1,1,0,dim=1)
            elif name is "grad2_forward":
                # this is wrong
                nu = define_kernel(0,-1,1,dim=1).T
            elif name is "grad2_backward":
                # this is wrong
                nu = define_kernel(0,-1,1,dim=1).T
            elif name is "laplacian1":
                nu = define_kernel(1,-2,1)
            elif name is "laplacian2": 
                # the only place where I use a 2D array
                nu = np.zeros((3,3))
                nu[0,1] = 1
                nu[1,1] = -4
                nu[2,1] = 1
                nu[1,0] = 1
                nu[1,2] = 1
            else:
                raise NotImplementedError
                
            return nu          
        elif name=='motion':
            return np.load('assets/motionblur.npy')
        else:
            f = get_f()
            if name.endswith('1'):
                direction = 1
            elif name.endswith('2'):
                direction = 2
            else:
                direction = None

            if name.startswith('gaussian'):
                def f(i,j):
                    denom = 2*(tau**2)
                    num = math.hypot(i,j)**2
                    return math.exp(-num/denom)
            elif name.startswith('exponential'):
                def f(i,j): 
                    denom = tau
                    num = math.hypot(i,j)
                    return math.exp(-num/denom)
            elif name.startswith('box'):
                def f(i,j):
                    return 1 if max(abs(i),abs(j)) <= tau else 0
            else:
                raise Exception

            # find s1,s2
            s1,s2,Z = find_parameters(f)
            
            if (name=='gaussian') and (tau==1) and (eps==1e-3):
                assert(s1==3 and s2==3)

            if (name=='exponential') and (tau==3) and (eps==1e-3):
                assert(s1==20 and s2==20)
        
            if direction is not None:
                if direction==1:
                    nu = np.zeros((s1*2+1,1))
                    s2 = 0
                    # recompute Z
                    _,Z = isvalid(s1,s2,f)
                else:
                    nu = np.zeros((1,s2*2+1))
                    s1 = 0
                    # recompute Z
                    _,Z = isvalid(s1,s2,f)
            else:
                nu = np.zeros((s1*2+1,s2*2+1))
                
            for ix,iy in np.ndindex(nu.shape):
                nu[ix,iy] = f(ix-s1,iy-s2) / Z
                
            return nu

    nu = get_nu()
    
    return nu

def kernel2fft(nu, n1, n2, separable=None):
    import itertools
    if separable is None:      
        s1 = (nu.shape[0] - 1) // 2
        s2 = (nu.shape[1] - 1) // 2

        assert(n1>=s1 and n2>=s2)

        tmp = np.zeros((n1, n2))
        tmp[:s1+1, :s2+1] = nu[s1:2*s1+1, s2:2*s2+1]
        #yellow
        tmp[n1-s2:, :s2+1] = nu[:s1, s2:2*s2+1]
        # green
        tmp[:s1+1, n2-s2:] = nu[s1:2*s1+1, :s2]
        # red
        tmp[n1-s1:, n2-s2:] = nu[:s1, :s2]
    else:
        def helper_sep(nu1,nu2,idx1,idx2):   
            temp = np.zeros((idx1[1]-idx1[0],idx2[1]-idx2[0]))
            
            if separable=='product':
                temp = np.matmul(nu1[slice(*idx1)],nu2[:,slice(*idx2)])
            elif separable=='sum':
                indeces = itertools.product(range(idx1[0],idx1[1]), \
                                            range(idx2[0],idx2[1]))
                for (id1,id2) in indeces:
                    i,j = id1-idx1[0],id2-idx2[0]
                    if (id2==0):
                        temp[i,j] += nu1[id1]
                    if (id1==0):
                        temp[i,j] += nu2[:,id2]
                    
            return temp
        
        nu1, nu2 = nu
        s1 = (nu1.size - 1) // 2
        s2 = (nu2.size - 1) // 2
        assert(n1>=s1 and n2>=s2)

        tmp = np.zeros((n1, n2))
        tmp[:s1+1, :s2+1] = helper_sep(nu1,nu2,(s1,2*s1+1), (s2,2*s2+1))
        #yellow
        tmp[n1-s2:, :s2+1] = helper_sep(nu1,nu2,(0,s1), (s2,2*s2+1))
        # green
        tmp[:s1+1, n2-s2:] = helper_sep(nu1,nu2,(s1,2*s1+1), (0,s2))
        # red
        tmp[n1-s1:, n2-s2:] = helper_sep(nu1,nu2,(0,s1), (0,s2))
    
    lbd = npf.fft2(tmp,axes=(0,1))

    return lbd      

def kernel2fft(nu, n1, n2, separable=None):
    if separable is None:
        tmp = np.zeros((n1, n2))
        s1 = int((nu.shape[0] - 1) / 2)
        s2 = int((nu.shape[1] - 1) / 2)
        tmp[:s1+1, :s2+1] = nu[s1:2*s1+1, s2:2*s2+1]
        tmp[:s1+1, n2-s2:n2] = nu[s1:2*s1+1,:s2]
        tmp[n1-s1:n1, :s2+1] = nu[:s1, s2: 2*s2+1]
        tmp[n1-s1:n1, n2-s2:n2] = nu[0:s1, 0:s2]
        lbd = nf.fft2(tmp, axes=(0, 1))
    else:
        tmp = np.zeros((n1, n2))
        s1 = int((nu[0].shape[0] - 1) / 2)
        s2 = int((nu[1].shape[1] - 1) / 2)
        nu = nu[0]*nu[1]
        tmp[:s1+1, :s2+1] = nu[s1:2*s1+1, s2:2*s2+1]
        tmp[:s1+1, n2-s2:n2] = nu[s1:2*s1+1,:s2]
        tmp[n1-s1:n1, :s2+1] = nu[:s1, s2: 2*s2+1]
        tmp[n1-s1:n1, n2-s2:n2] = nu[0:s1, 0:s2]
        lbd = nf.fft2(tmp, axes=(0, 1))
        #tmp1 = np.zeros((n1, n2))
        #tmp2 = np.zeros((n1, n2))
        #tmp1[:s1+1,0] = nu[0][s1:2*s1+1,0]
        #tmp1[n1-s1:n1,0] = nu[0][:s1,0]
        #tmp2[0,n2-s2:n2] = nu[1][0,:s2]
        #tmp2[0,:s2+1] = nu[1][0,s2:2*s2+1]
        #tmp = tmp1+tmp2
        #lbd = nf.fft2(tmp, axes=(0,1))
        
    return lbd