import numpy as np
from scipy.fftpack import dct,idct


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

# Creates the DCT basis matrices. 
# (1) memory consumption.
# G is a 3-D tensor that contains all the 2-D basis matrices for a given 8x8 block.
# The i entry in G corresponds to the transformation for the (i//8,i%8) frequency.
DCT = lambda u,v: np.array([
    [np.cos((2*x+1)*u*np.pi * 1/16)*np.cos((2*y+1)*v*np.pi * 1/16) for x in range(8)]
    for y in range(8)]
)
G = np.array([DCT(k//8,k%8) for k in range(64)])
ALPHA = lambda u: 1/np.sqrt(2) if u == 0 else 1
MASKS= np.array([ALPHA(i//8)*ALPHA(i%8) for i in range(64)])

def vectorized_2D_DCT(X):
  block = np.einsum('ijk,jk->i',G,X)
  block = 0.25 * (block * MASKS)
  return block.reshape(8,8).T


inverse_DCT =  lambda x,y: np.array([
    [ALPHA(u)*ALPHA(v)* np.cos((2*x+1)*u*np.pi * 1/16) *np.cos((2*y+1)*v*np.pi * 1/16) for u in range(8)] 
                          for v in range(8)]
)
G_inverse = np.array([inverse_DCT(k//8,k%8) for k in range(64)])

def vectorized_inverse_DCT(X):
  block = np.einsum('ijk,jk->i',G_inverse,X)
  block = 0.25 * block
  return block.reshape(8,8).T