import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

from huffman_encoding import *
from cosine_transform import * 
from utils import *

def compress_block(block : np.ndarray):
  """
    Compression pipeline for a 8x8 block matrix.
    Functions applies 2-D DCT transformation and then quantizes the frequencies 
    matrix. In the next step the function flattens out the block in zig-zag order
    to accumulate non-zeros together and then apply RLE before encoding the 
    block with Huffman Encoding.
    Returns:
        - DC_frequency: the leftmost frequency in the 8x8 DCT matrix. It's needed for different encoding.
        - compressed_block: the 8x8 matrix returned in compressed format after quantization and huffman encoding
        - huffman_table: the huffman table used to encode the top byte in the binary representation. It's needed to be saved for later decoding.
  """
  block = vectorized_2D_DCT(block)

  block = quantize_block(block)
  block = zigzag(block.flatten())
  DC_frequency = block[0]

  RLE_block = run_length_encoding(block)
  encoder = HuffmanEncoder(RLE_block)
  huffman_table = encoder.get_codes() 
  print(huffman_table)

  #compressed_block = np.array([huffman_table[symbol] + convert_to_binary(freq,symbol[1]) for symbol,freq in zip(RLE_block, block)])
  #return DC_frequency,compressed_block, huffman_table

# For testing
B = -1*np.array([
    [76,73,67,62,58,67,64,55],
    [65,69,73,38,19,43,59,56],
    [66,69,60,15,-16,24,62,55],
    [65,70,57,6,-26,22,58,59],
    [61,67,60,24,2,40,60,58],
    [49,63,68,58,51,60,70,53],
    [43,57,64,69,73,67,63,45],
    [41,49,59,60,63,52,50,34]
])

def process_image(path):
   obj = Image.open(path).convert('L')
   img = np.array(obj)
   img = img - 128
   blocks = block_splitting(img)
   compressed_ACs = []
   DC_frequencies = []
   huffman_tables = []
   for i,block in enumerate(blocks):
      print(f"Proccesing block {i}/{len(blocks)}")
      compress_block(block)
      #DC_freq, reduced_block, htable = compress_block(block)
      break
      #DC_frequencies.append(DC_freq)
       #huffman_tables.append(htable)
    


      


if __name__ == "__main__":
    process_image('cat.jpg')
    #output = compress_block(B)
    #print(output)