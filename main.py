import numpy as np 
from tqdm import tqdm


from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed   
import os
import traceback

from mini_jpeg.huffman_encoding import *
from mini_jpeg.cosine_transform import * 
from mini_jpeg.utils import *
from mini_jpeg.codec import *


NUM_THREADS = os.cpu_count()
            
def compress_block(block : np.ndarray):
  """
    Compression pipeline for a 8x8 block matrix.
    Pipeline:
    2D Discrete Cosine Transform ---> Quantization ---> Zigzag
    ---> RLE Encoding ---> Canonical Huffman Algorithm

    Returns:
        - DC_frequency: the leftmost frequency in the 8x8 DCT matrix. It's needed for different encoding.
        - compressed_block: the 8x8 matrix returned in compressed format after quantization and huffman encoding
        - huffman_table: the huffman table used to encode the top byte in the binary representation. It's needed to be saved for later decoding.
  """
  block = vectorized_2D_DCT(block)
  block = quantize_block(block).astype(np.int8)
  block = zigzag(block.flatten())
  DC_frequency = block[0]
  
  RLE_block, indices = run_length_encoding(block)

  encoder = HuffmanEncoder(RLE_block)
  sorted_symbols,lens_arr = encoder.get_canonical_codes() 
  
  code = 0
  codes_len = []
  for i,amount in enumerate(lens_arr):
     for _ in range(amount):
        codes_len.append(i+1)

  canonical_codes = [0]
  for i in range(1,len(codes_len)):
   code = (code+1) << (codes_len[i] - codes_len[i-1])
   canonical_codes.append(code)

  table = {
      symbol:code for symbol,code in zip(sorted_symbols,canonical_codes)
  }
  compressed_block = np.array([
    [table[symbol],block[i]] for symbol,i in zip(RLE_block, indices)]
  )
  
  huffman_table = [sorted_symbols,lens_arr]
  return DC_frequency,compressed_block, huffman_table

def test_pipeline_block(test,img_shape):
   DC_freq, compressed_block, huffman = compress_block(test,True)
   decoded_block = decode_AC_block(compressed_block, huffman)
   DC_differences = [DC_freq]
   AC_coeffs = [compressed_block]
   huffman_tables = [huffman]

   encode_JPEG_format('test\shimon',DC_differences,AC_coeffs,huffman_tables,img_shape)
   decode_JPEG_format('test\shimon')


def process_image(path : str):
   obj = Image.open(path).convert('L')
   img = np.array(obj)
   img = img - 128
   blocks, img_shape = crop_and_block(img)

   num_blocks = blocks.shape[0]
   AC_coeffs = [None]*num_blocks
   DC_coeffs = [None]*num_blocks
   huffman_tables = [None]*num_blocks

   i = 0
   with tqdm(total=len(blocks)) as pbar:
      with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
         future_results = {executor.submit(compress_block, blocks[i]): i for i in range(blocks.shape[i])}
         for future in as_completed(future_results):
            try:
               DC_freq, reduced_block, htable = future.result()
               index = future_results[future]
               #print(f"Processed {i}/{len(blocks)} blocks")
               AC_coeffs[index] = reduced_block
               DC_coeffs[index] = DC_freq
               huffman_tables[index] = htable
               i+=1
               pbar.update(1)
            except Exception as e:
               print(traceback.format_exc())

   #Observe the variation in DC coefficients is usually small,
   #hence if we keep only the differences we can use less bits.
   DC_differences = np.diff(DC_coeffs,n=1,prepend=[0])
   DC_differences = DC_differences.astype(np.int8)

   filename = path.split('.')[0] + '_compressed'
   dist = [len(block) for block in AC_coeffs]
   
   ratio = (round(sum(dist) / (len(blocks)*64),2))*100
   print(f"Compression ratio: {ratio}%")

   encode_JPEG_format(filename,DC_differences,AC_coeffs,huffman_tables, img_shape)


NAME='cat'

"""
Optimization ideas:
- Codec can be improved by handling specific AC frequency byte sizes to reduce memory overhead.
- Refactor the codec saving format to reduce redundant symbols.
- Add Numba for processing the DCT.
- Add multi-processing; allocate subsets of blocks for each core and process the subset of blocks by multi-threading.
"""
if __name__ == "__main__":
    process_image(f'test\{NAME}_raw.bmp')
    decode_JPEG_format(f'test\{NAME}_raw_compressed')