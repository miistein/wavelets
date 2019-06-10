""" Project C

COMPLETE THIS FILE

Your names here:

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
        return self.__call__(self.adjoint(x))
    
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
        return self.__call__(self.adjoint(x))
    
    def gram_resolvent(self, x, tau):
        return cg(lambda z: z + tau * self.gram(z), x)
 