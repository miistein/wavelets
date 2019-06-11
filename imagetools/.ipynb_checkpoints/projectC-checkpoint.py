""" Project C

COMPLETE THIS FILE

Your names here:
Justin Law

"""

from .assignment6 import *

class Identity(LinearOperator):
    def __init__(self, shape):
        ishape = oshape = shape
        LinearOperator.__init__(self, ishape, oshape)
        
    def __call__(self, x):
        return x
    
    def adjoint(self, x):
        # note that the adjoint of the identity matrix is the identity matrix itself.
        # it is easy to see this by the definition of a linear operator, noting the commutative property of the inner product    
        return self.__call__(x)

    def gram(self, x):
        # apply the gram matrix to x, which is equivalent to just returning x
        return self.adjoint(self.__call__(x))
    
    def gram_resolvent(self, x, tau):
        # do the inversion by the conjugate gradient
        return cg(lambda z: z + tau * self.gram(z), x)
    
class Convolution(LinearOperator):
    def __init__(self, shape, nu, separable=None):
        ishape = oshape = shape
        LinearOperator.__init__(self, ishape, oshape)

        self._separable = separable
        n1, n2 = ishape[:2]
        self._nu_fft = kernel2fft(nu, n1, n2, separable=self._separable)
        # mu = F^{-1}(conjugate(F(nu))), see assignment 5 for explanation
        self._mu_fft = np.conjugate(self._nu_fft)
        
    def __call__(self, x):
        return convolvefft(x,self._nu_fft)
    
    def adjoint(self, x):
        return convolvefft(x, self._mu_fft)
    
    def gram(self, x):
        # self.adjoint(self.__call__(x),x), reduced to convolvefft(x, nu_fft*mu_fft) by properties of fft
        return convolvefft(x, nu_fft*mu_fft)
    
    def gram_resolvent(self, x, tau):
        # Assuming that boundary is always periodical
        # Just like the Gram LinearOperator, can do the inversion in the Fourier domain when periodical
        # reduced to:
        res_nu = 1 / (1 + tau * self._nu_fft * self._mu_fft)
        return convolvefft(x, res_nu)
    
class RandomMasking(LinearOperator):
    def __init__(self, shape, p):
        ishape = oshape = shape
        LinearOperator.__init__(self, ishape, oshape)
        
        self._apply_random_mask = np.random.choice(np.array([0,1]), size=oshape, p=[p,1-p])
        
    def __call__(self, x):
        # apply pixel by pixel the random mask
        return self._apply_random_mask * x
    
    def adjoint(self, x):
        # easy to see this is self-adjoint by the definition of a linear operator, noting the commutative property of the inner product 
        return self.__call__(x)

    def gram(self, x):
        # can also be reduced to self._apply_random_mask * x because 0*0=0, 1*0 =0, 0*1=0,1*1=1
        return self.adjoint(self.__call__(x))
    
    def gram_resolvent(self, x, tau):
        return cg(lambda z: z + tau * self.gram(z), x)
        
################
# Part 2

def flip (x):
    return np.transpose(x,axes= (1,0,2))

def dwt1d(x, h, g): # 1d and 1scale
    coarse = convolve(x, g)
    detail = convolve(x, h)
    z = np.concatenate((coarse[::2, :], detail[::2, :]), axis=0)
    return z

def dwt(x, J, h, g):
    if J == 0:
        return x
    n1, n2 = x.shape[:2]
    m1, m2 = (int(n1 / 2), int(n2 / 2))
    z = dwt1d(x, h, g)
    z = flip(dwt1d(flip(z), h, g))
    z[:m1, :m2] = dwt(z[:m1, :m2], J - 1, h, g)
    return z

def idwt(z, J, h, g): # 2d and multi-scale
    if J == 0:
        return z
    n1, n2 = z.shape[:2]
    m1, m2 = (int(n1 / 2), int(n2 / 2))
    x = z.copy()
    x[:m1, :m2] = idwt(x[:m1, :m2], J - 1, h, g)
    x = flip(idwt1d(flip(x), h, g))
    x = idwt1d(x, h, g)
    return x

def idwt1d(z, h, g): # 1d and 1scale
    n1 = z.shape[0]
    m1 = int(n1 / 2)
    coarse, detail = np.zeros(z.shape), np.zeros(z.shape)
    coarse[::2, :], detail[::2, :] = z[:m1, :], z[m1:, :]
    x = convolve(coarse, g[::-1]) + convolve(detail, h[::-1])
    return x
    
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

def convolve(x, nu, boundary='periodical', seperable='None'):
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


def dwt_power(n1, n2, J, ndim=3):

    if J == 0:
        return np.ones((n1, n2, *[1] * (ndim - 2)))
    m1, m2 = int(n1/2), int(n2/2)
    c = 2 * dwt_power(m1, m2, J - 1, ndim=ndim)
    de = np.ones((m1, m2, *[1] * (ndim - 2)))
    p = np.concatenate((np.concatenate((c, de), axis=0),
                        np.concatenate((de, de), axis=0)), axis=1)
    return p

class DWT:
    def __init__(self, shape, J, name):
        self.name = name
        self.J = J
        self.shape=shape
        h,g = wavelet(self.name)
     
    def __call__(self, x):
        h,g = wavelet(self.name)
        z_dwt = dwt(x,self.J,h,g)
        return z_dwt
    
    def invert(self,x):
        h,g = wavelet(self.name)
        return idwt(x,self.J,h,g)
    
    def power(self):
        n1,n2= self.shape[:2]
        J= self.J
        return dwt_power(n1, n2, self.J, ndim=len(self.shape))

    def adjoint(self, x):
        return self.invert(x)

    def gram(self, x):
        return self.adjoint(self.__call__(x))

    def gram_resolvent(self, x, tau):
        #return cg(lambda z: z + tau * self.gram(z), x)
        # reduced to:
        return x/(1+tau)
   
    def wavelet(name, d=2):
    
        if name in ('haar', 'db1'):
            h = np.array([-1, 1])
        if name is 'db2':
            h = np.array([1, np.sqrt(3), -(3 + 2 * np.sqrt(3)), 2 + np.sqrt(3)])
        if name is 'db4':
            h = np.array(
            [-0.230377813308855230, +0.714846570552541500, -0.630880767929590400,
             -0.027983769416983850, +0.187034811718881140, +0.030841381835986965,
             -0.032883011666982945, -0.010597401784997278])
        if name is 'db8':
            h = np.array(
            [-0.0544158422, +0.3128715909, -0.6756307363, +0.5853546837,
             +0.0158291053, -0.2840155430, -0.0004724846, +0.1287474266,
             +0.0173693010, -0.0440882539, -0.0139810279, +0.0087460940,
             +0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])
        if name is 'sym4':
            h = np.array(
            [-0.03222310060404270, -0.012603967262037833, +0.09921954357684722,
             +0.29785779560527736, -0.803738751805916100, +0.49761866763201545,
             +0.02963552764599851, -0.075765714789273330])
        if name is 'coif4':
            h = np.array(
            [-0.00089231366858231460, -0.00162949201260173260, +0.00734616632764209350,
             +0.01606894396477634800, -0.02668230015605307200, -0.08126669968087875000,
             +0.05607731331675481000, +0.41530840703043026000, -0.78223893092049900000,
             +0.43438605649146850000, +0.06662747426342504000, -0.09622044203398798000,
             -0.03933442712333749000, +0.02508226184486409700, +0.01521173152794625900,
             -0.00565828668661072000, -0.00375143615727845700, +0.00126656192929894450,
             +0.00058902075624433830, -0.00025997455248771324, -6.2339034461007130e-05,
             +3.1229875865345646e-05, +3.2596802368833675e-06, -1.7849850030882614e-06])
        h = h / np.sqrt(np.sum(h**2))
        g = (-1)**(1 + np.arange(h.shape[0])) * h[::-1]
        h = np.concatenate((h, np.array([0.])))
        g = np.concatenate((g, np.array([0.])))
        h = h.reshape(-1, *[1] * (d - 1))
        g = g.reshape(-1, *[1] * (d - 1))
        return h, g
    
    def dtw_crop(x, J):
        n1, n2 = x.shape[:2]
        r1 = np.mod(n1, 2**J)
        r2 = np.mod(n2, 2**J)
        if r1 > 0:
            x = x[int(r1/2):-(r1-int(r1/2)), :]
        if r2 > 0:
            x = x[:, int(r2/2):-(r2-int(r2/2))]
        return x

def softthresh(z, t):
    new_z = (z - np.sign(z-t) * t) * np.maximum(np.abs(z)-t,0).astype(bool)
    
    return new_z

def softthresh_denoise(y, sig, W, alpha):
    p=W.power()
    lamda=np.zeros(p.shape)
    tau=np.zeros(p.shape)
    lamda = alpha*p
    tau= (np.sqrt(2)*(sig**2))/lamda
    
    #z=dtw_crop(y,J)
    z = W(y)
    z_denoise = softthresh(z,tau)
    # adjoint is inverse transform for both DWT and UDWT
    denoise = W.adjoint(z_denoise)
    return denoise

################################################ 
####### part 3

def wavelet(name, d=2):
    if name in ('haar', 'db1'):
        h = np.array([-1, 1])
    if name is 'db2':
        h = np.array([1, np.sqrt(3), -(3 + 2 * np.sqrt(3)), 2 + np.sqrt(3)])
    if name is 'db4':
        h = np.array(
            [-0.230377813308855230, +0.714846570552541500, -0.630880767929590400,
            -0.027983769416983850, +0.187034811718881140, +0.030841381835986965,
            -0.032883011666982945, -0.010597401784997278])
    if name is 'db8':
        h = np.array(
            [-0.0544158422, +0.3128715909, -0.6756307363, +0.5853546837,
            +0.0158291053, -0.2840155430, -0.0004724846, +0.1287474266,
            +0.0173693010, -0.0440882539, -0.0139810279, +0.0087460940,
            +0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])
    if name is 'sym4':
        h = np.array(
            [-0.03222310060404270, -0.012603967262037833, +0.09921954357684722,
            +0.29785779560527736, -0.803738751805916100, +0.49761866763201545,
            +0.02963552764599851, -0.075765714789273330])
    if name is 'coif4':
        h = np.array(
            [-0.00089231366858231460, -0.00162949201260173260, +0.00734616632764209350,
            +0.01606894396477634800, -0.02668230015605307200, -0.08126669968087875000,
            +0.05607731331675481000, +0.41530840703043026000, -0.78223893092049900000,
            +0.43438605649146850000, +0.06662747426342504000, -0.09622044203398798000,
            -0.03933442712333749000, +0.02508226184486409700, +0.01521173152794625900,
            -0.00565828668661072000, -0.00375143615727845700, +0.00126656192929894450,
            +0.00058902075624433830, -0.00025997455248771324, -6.2339034461007130e-05,
            +3.1229875865345646e-05, +3.2596802368833675e-06, -1.7849850030882614e-06])
    h = h / np.sqrt(np.sum(h**2))
    g = (-1)**(1 + np.arange(h.shape[0])) * h[::-1]
    h = np.concatenate((h, np.array([0.])))
    g = np.concatenate((g, np.array([0.])))
    h = h.reshape(-1, *[1] * (d - 1))
    g = g.reshape(-1, *[1] * (d - 1))
    return h, g

def udwt_power(J, ndim=3):
    p = ((4)**J, )
    for j in range(1, J+1):
        p += ((4)**(J - j), ) * 3
    p = np.array(p, dtype=np.float32).reshape(1, 1, -1, *[1] * (ndim - 2))
    p[:, :, 1:] *= 2
    return p

def fb_apply(x, fb):    
    x = nf.fft2(x, axes=(0, 1))
    z = fb * x[:, :, np.newaxis]
    z = np.real(nf.ifft2(z, axes=(0, 1)))
    return z

class UDWT(LinearOperator):
    def __init__(self, shape, J, name = 'db2' , using_fb = True):
        self.__ishape = shape
        self.__using_fb = using_fb
        self.__J = J
        h, g = wavelet(name)
        self.__h = h
        self.__g = g
        LinearOperator.__init__(self, shape, shape)
        
    def iudwt(self, z):
        h = self.__h
        g = self.__g
        J = self.__J
        if J == 0:
            try:
                return z[:,:,0]
            except IndexError:
                return z
        h2 = np.zeros((2*h.shape[0]-1,1))
        g2 = np.zeros((2*g.shape[0]-1,1))
        h2[::2] = h
        g2[::2] = g
        self.__h = h2
        self.__g = g2
        self.__J = J - 1
        coarse = self.iudwt(z[:, :, :-3])
        tmpg = convolve(coarse, g[::-1]) + convolve(z[:, :, -3], h[::-1])
        tmph = convolve(z[:, :, -2], g[::-1]) + convolve(z[:, :, -1], h[::-1])
        x = (np.swapaxes(convolve(np.swapaxes(tmpg,0,1), g[::-1]),0,1) + np.swapaxes(convolve(np.swapaxes(tmph,0,1),h[::-1]),0,1))/2
        self.__h = h
        self.__g = g
        self.__J = J
        return x


    def __call__(self, x):
        h = self.__h
        g = self.__g
        J = self.__J
        if self.__using_fb == True :
            filter_bank = self.udwt_create_fb()
            return fb_apply(x, filter_bank)
        if J == 0:
            return x[:, :, np.newaxis]
        tmph = np.swapaxes(convolve(np.swapaxes(x,0,1), h),0,1) / 2
        tmpg = np.swapaxes(convolve(np.swapaxes(x,0,1), g),0,1) / 2
        detail = np.stack((convolve(tmpg, h),
        convolve(tmph, g),
        convolve(tmph, h)), axis=2)
        coarse = convolve(tmpg, g)
        h2 = np.zeros((2*h.shape[0]-1,1))
        g2 = np.zeros((2*g.shape[0]-1,1))
        h2[::2] = h
        g2[::2] = g
        self.__h = h2
        self.__g = g2
        self.__J = J - 1
        z = np.concatenate((self.__call__(coarse), detail), axis=2)
        self.__h = h
        self.__g = g
        self.__J = J
        return z
        
    def udwt_create_fb(self, ndim=3):
        h = self.__h
        g = self.__g
        J = self.__J
        ishape = self.__ishape
        n1, n2 = ishape[:2]
        if J == 0:
            return np.ones((n1, n2, 1, *[1] * (ndim - 2)))
        h2 = np.zeros((2*self.__h.shape[0]-1,1))
        g2 = np.zeros((2*self.__g.shape[0]-1,1))
        h2[::2] = h
        g2[::2] = g
        self.__h = h2
        self.__g = g2
        self.__J = self.__J - 1
        fbrec = self.udwt_create_fb(ndim = ndim)
        gf1 = kernel2fft(g,n1,1)
        hf1 = kernel2fft(h,n1,1)
        gf2 = kernel2fft(g.T,1,n2)
        hf2 = kernel2fft(h.T,1,n2)
        fb = np.zeros((n1, n2, 4), dtype=np.complex128)
        fb[:, :, 0] = np.outer(gf1, gf2) / 2
        fb[:, :, 1] = np.outer(gf1, hf2) / 2
        fb[:, :, 2] = np.outer(hf1, gf2) / 2
        fb[:, :, 3] = np.outer(hf1, hf2) / 2
        fb = fb.reshape(n1, n2, 4, *[1] * (ndim - 2))
        fb = np.concatenate((fb[:, :, 0:1] * fbrec, fb[:, :, -3:]),
        axis=2)
        self.__h = h
        self.__g = g
        self.__J = J
        return fb
        
    def invert(self, x):
        return np.linalg.pinv(x)
        
    def power(self):
        return udwt_power(self.__J)

    def adjoint(self, x):
        return self.iudwt(x)

    def gram(self, x):
        return self.adjoint(self.__call__(x))

    def gram_resolvent(self, x, tau):
        #return cg(lambda z: z + tau * self.gram(z), x)
        # reduced to:
        #return cg(lambda z: z + tau * self.gram(z), x)
        return x/(1+tau)
    
# helper function
# use psnr from assignment 4 for noise dB calculation
def psnr(x, x0):
    R = (255 if np.any(x>1) else 1)
    den = ((x.flatten() - x0.flatten())**2).mean()
    snr = 10*math.log(R**2/den, 10)
    return snr