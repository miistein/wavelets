""" Assignment 6

COMPLETE THIS FILE

Your name here: Justin Law
PID: A12346613

"""

from .assignment5 import *
            
def deconvolve_naive(y, lbd, return_transfer=False):
    hhat = np.conjugate(lbd) / np.power(np.abs(lbd),2)
    yhat = npf.fftn(y)
    if y.ndim==3:
        hhat3d = np.repeat(hhat[:, :, np.newaxis], 3, axis=2)
        xdec = np.real(npf.ifftn(yhat*hhat3d))
    else:
        xdec = np.real(npf.ifftn(yhat*hhat))
        
    if return_transfer: 
        return xdec, hhat
    else:
        return xdec

def deconvolve_wiener(x, lbd, sig, mpsd, return_transfer=False):
    n1,n2 = x.shape[:2]
    hhat = np.conjugate(lbd)
    lhs = np.power(np.abs(lbd),2) 
    rhs = n1*n2*(sig**2)*np.reciprocal(mpsd)
    hhat /= (lhs + rhs)
    yhat = npf.fftn(x)
    
    if x.ndim==3:
        hhat3d = np.repeat(hhat[:, :, np.newaxis], 3, axis=2)
        xdec = np.real(npf.ifftn(yhat*hhat3d))
    else:
        xdec = np.real(npf.ifftn(yhat*hhat))

    if return_transfer: 
        return xdec, hhat
    else:
        return xdec
    
def average_power_spectral_density(x):
    '''
        x is a list of K images (color or grayscale)
    '''
    
    x_hat = [np.abs(npf.fft2(element,axes=(0,1)))**2 for element in x]
    
    # sum over the third dimension if rgb image
    x_hat = [np.sum(element,axis=2) / 3 for element in x_hat if element.ndim==3]
    
    # take the average as described in Q1
    average_S = sum(x_hat) / len(x)

    return average_S

def mean_power_spectrum_density(apsd):
    n1,n2 = apsd.shape
    u,v = fftgrid(n1,n2)

    suv = np.log(apsd/(n1*n2))
    omegas = np.sqrt((u/n1)**2 + (v/n2)**2)
    
    # preprocessing
    omegas[omegas==0] = 1 # set to 1 so that not included in sum
    suv[0,0] = 0 # set to 0 so not included in sum
    
    # calculate t(u,v) and s(u,v)*t(u,v)
    tuv = np.log(omegas)
    suv_tuv = suv*tuv
    
    det = (n1 * n2 * np.sum(tuv**2)) - (np.sum(tuv)**2)
    
    alpha = (n1 * n2 * np.sum(suv_tuv)) - (np.sum(tuv) * np.sum(suv))
    alpha /= det
    
    beta = (np.sum(tuv**2) * np.sum(suv)) - (np.sum(tuv) * np.sum(suv_tuv))
    beta /= det                           
   
    mpsd = n1*n2* np.exp(beta) * (omegas**alpha)
    mpsd[0,0] = np.inf
    return mpsd, alpha, beta

def plot_log(u,lines,*psds):
    n1,n2 = psds[0].shape
    _,v = fftgrid(n1,n2)
    
    
    fig,ax = plt.subplots(1, 1)
    
    for i,psd in enumerate(psds):
        ax.set_yscale('log')
        xs = v[u,:]
        ys = psd[u,:]
        
        xs, ys = zip(*sorted(zip(xs, ys),key=lambda x:x[0]))
        ax.plot(xs,ys,label=lines[i])
        
    return ax