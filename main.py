import numpy as np 
import matplotlib.pyplot as plt 

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed   
import os

from mini_jpeg.huffman_encoding import *
from mini_jpeg.cosine_transform import * 
from mini_jpeg.utils import *
from mini_jpeg.codec import *


NUM_THREADS = os.cpu_count()
            
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

  RLE_block, indices = run_length_encoding(block)
  encoder = HuffmanEncoder(RLE_block)
  huffman_table = encoder.get_codes() 

  compressed_block = np.array([huffman_table[symbol] + convert_to_binary(block[i],symbol[1]) for symbol,i in zip(RLE_block, indices)])
  return DC_frequency,compressed_block, huffman_table


def process_image(path : str):
   obj = Image.open(path).convert('L')
   img = np.array(obj)
   img = img - 128
   blocks = block_splitting(img)
   AC_coeffs = []
   DC_coeffs = []
   huffman_tables = []

   i = 0
   with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
      future_results = {executor.submit(compress_block, block): block for block in blocks}

      for future in as_completed(future_results):
         try:
            DC_freq, reduced_block, htable = future.result()
            print(f"Processed {i}/{len(blocks)} blocks")
            AC_coeffs.append(reduced_block)
            DC_coeffs.append(DC_freq)
            huffman_tables.append(htable)
            i+=1
         except Exception as e:
            pass

   # Observe the variation in DC coefficients is usually small,
   # hence if we keep only the differences we can use less bits.
   DC_differences = np.diff(DC_coeffs,n=1,prepend=[0])
   DC_differences = np.array([convert_to_binary(x,8) for x in DC_differences])

   DC_differences = DC_differences.astype(np.int8)

   # for block in AC_coeffs:
   #    x = [ np.ceil(np.log2(int(e,2)+1))//8 for e in block]
   #    print(x)

   filename = path.split('.')[0] + '_compressed'
   dist = [len(block) for block in AC_coeffs]
   ratio = (round(sum(dist) / (len(blocks)*64),2))*100
   print(f"Compression ratio: {ratio}%")
   print(len(huffman_tables))
   encode_JPEG_format(filename,DC_differences,AC_coeffs,huffman_tables)
   plt.hist(dist,bins=30)
   plt.title("Distribution of number of codes among blocks");
   plt.show();


   


if __name__ == "__main__":
    #process_image('test\cat_raw.bmp')
    decode_JPEG_format('test\cat_raw_compressed')