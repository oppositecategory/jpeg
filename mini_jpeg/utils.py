import numpy as np 

# Quantization matrix
Q = np.array([[16,11,10,16,24,40,51,61],
              [12,12,14,19,26,58,60,55],
              [14,13,16,24,40,57,69,56],
              [14,17,22,29,51,87,80,62],
              [18,22,37,56,68,109,103,77],
              [24,35,55,64,81,104,113,92],
              [49,64,78,87,103,121,120,101],
              [72,92,95,98,112,100,103,99],
              ])

# Matrix of zig-zag inidices for frequency blocks. 
zigzag_order = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 50, 43, 36, 29, 22,
    15, 23, 30, 37, 44, 51, 52, 45,
    38, 31, 39, 46, 53, 54, 47, 55,
    56, 57, 58, 59, 60, 61, 62, 63
]

quantize_block = lambda X: np.round(X/Q)

def zigzag(block : np.ndarray):
  return np.array([block[index] for index in zigzag_order]).astype(int)

def run_length_encoding(block : np.ndarray):
  zero_freqs = 0
  SYMBOL = (15,0)
  #EOB = (0,0)
  codes = []
  indices = []
  for i,freq in enumerate(block.flatten()[1:]):
    if freq != 0:
      run_length = zero_freqs
      size = min(15, int(np.ceil(np.log2(np.abs(freq)+1))))
      codes.append((run_length,size))
      zero_freqs = 0
      indices.append(i+1)
    elif zero_freqs == 16:
      codes.append(SYMBOL)
      indices.append(i+1)
      zero_freqs = 0
    else:
      zero_freqs+=1

  # Make sure to remove redundant SYMBOLs.
  for code in codes[::-1]:
    if code != SYMBOL:
      break
    codes.pop()
  #codes.append(EOB)
  return codes, indices

def pad_and_block(img : np.ndarray):
  """ Function splits our image into chunks of 8x8 sub-matrices.
      If image's size isn't integer multiples of 8 we pad with the edges values.
  """
  M,N = img.shape
  A = np.zeros((M + (M % 8),N + (N % 8)))
  A[:M,:N] = img.copy()
  A[:M, N:] =  np.broadcast_to(A[:M,N-1,None],(M,N%8))
  A[M:, ::] = np.broadcast_to(A[M:,::],(M%8,N))
  blocks = A.reshape(-1,8,8)
  return blocks

def crop_and_block(img : np.ndarray):
  """ Function splits our image into chunks of 8x8 sub-matrices.
      If image's size isn't integer multiples of 8 we crop it.
  """
  M,N = img.shape
  M1, N1 = M - (M%8), N - (N%8)
  A = img[:M1, :N1]
  blocks = A.reshape(-1,8,8)
  return blocks

def convert_to_binary(x,num_bits=15):
  # Note that x can have at most 15 bits, hence bounded by 2^14-1 to it's -2^14.
  # We make sure x is in the range and write it as a signed number.
  x = max(min(x, 16383), -16484)
  if x >= 0: return format(x,f'0{num_bits}b')
  return format(x & (2**num_bits - 1),'b')
