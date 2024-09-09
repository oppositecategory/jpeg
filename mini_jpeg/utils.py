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
indices = np.arange(64).reshape(8,8)
order = [[] for i in range(8+8-1)]

for i in range(8):
   for j in range(8):
      s = i+j
      if s % 2 == 0:
         order[s].insert(0,indices[i,j])
      else:
         order[s].append(indices[i,j])
zigzag_order = np.concatenate(order)

quantize_block = lambda X: np.round(X/Q)

def revert_zigzag(X):
  block = np.zeros((8,8))
  for i,index in enumerate(zigzag_order):
    row, col = divmod(index, 8)
    block[row,col] = X[i]
  return block 

def decode_AC_block(compressed, huffman_data):
  sorted_symbols, lens_arr = huffman_data
  code = 0
  codes = [0]
  codes_len = []
  for i,amount in enumerate(lens_arr):
    for _ in range(amount):
      codes_len.append(i+1)

  for i in range(1,len(codes_len)):
    code = (code+1) << (codes_len[i] - codes_len[i-1])
    codes.append(code)

  huffman_table = {code:symbol for symbol,code in zip(sorted_symbols,codes)} 
  decoded_block = []
  for encoded_symbol,freq in compressed:
    symbol = huffman_table[encoded_symbol]
    for _ in range(symbol[0]):
      decoded_block.append(0)
    decoded_block.append(freq)
  for i in range(63-len(decoded_block)):
    decoded_block.append(0)
  return decoded_block


def zigzag(block : np.ndarray):
  return np.array([block[index] for index in zigzag_order]).astype(np.int16)

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
  return blocks, (M1,N1)

def convert_to_binary(x,num_bits=15):
  # Note that x can have at most 15 bits, hence bounded by 2^14-1 to it's -2^14.
  # We make sure x is in the range and write it as a signed number.
  x = max(min(x, 16383), -16484)
  if x >= 0: return format(x,f'0{num_bits}b')
  return format(x & (2**num_bits - 1),'b')

def format_table(table):
  for q,v in table.items():
    v = '0'*(4 - ((len(v) % 4))) + v
    table[q] = v
  return table
