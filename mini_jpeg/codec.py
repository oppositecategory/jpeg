import numpy as np
import matplotlib.pyplot as plt 

import struct
from PIL import Image

from .utils import decode_AC_block, revert_zigzag,Q
from .cosine_transform import vectorized_inverse_DCT
FLAGS = {
    'SOI': 65496, # Start Of Image
    'SOB': 65475, # Start Of Block
    'SOLA': 65339, # Start Of Length Array,
    'SOAC': 65477, # Start Of AC
    'SODC': 65481 # Start Of DC
}

def encode_JPEG_format(filename, encoded_DC, encoded_AC, huffman_tables, img_shape):
    with open(filename, 'wb') as file:
        # Write SOI marker
        file.write(b'\xFF\xD8')
        shape_bytes = struct.pack('>HH', *img_shape)
        file.write(shape_bytes)
        # Huffman tables segment
        file.write(b'\xFF\xC4')
        file.write(len(huffman_tables).to_bytes(2, 'big')) 
        #print("Written table:", huffman_tables[-1])
        #print("Written:", encoded_AC[0])
        for data in huffman_tables:
            table,len_arr = data[0],data[1]

            file.write(b'\xFF\xC3')
            for symbol in table:
                key_bytes = struct.pack('>BB',*symbol) 
                file.write(key_bytes)

            file.write(b"\xFF\x3B")
            for x in len_arr:
                file.write(x.to_bytes(1,'big'))

            
        # AC frequencies segment
        file.write(b'\xFF\xC5')
        #print(encoded_AC[0])
        for i,block in enumerate(encoded_AC):
            start_of_block = b'\xFF\xC3' 
            file.write(start_of_block)
            #if i == 0:
            #    decoded_block = decode_AC_block(block, huffman_tables[0])
            #    print("W:",decoded_block)
            #    print("Length W:", len(decoded_block))
            for code,freq in block:
                code,freq = int(code),int(freq)
                code_bytes = code.to_bytes(1,
                                         byteorder='big')
                freq_bytes = freq.to_bytes(2,
                                           byteorder='big',
                                           signed=True)

                file.write(code_bytes)
                file.write(freq_bytes)
         
        # DC frequencies segment
        file.write(b'\xFF\xC9')  
        for diff in encoded_DC:
            packed_diff = struct.pack('>b',diff)
            file.write(packed_diff)
         
        # Write EOI marker
        file.write(b'\xFF\xD9')


def decode_JPEG_format(filename):
    with open(filename,'rb') as file:
        raw_binary = file.read()

    m,n = struct.unpack_from('>HH', raw_binary, offset=2)
    blocks_num = struct.unpack_from('>H',raw_binary, offset=8)[0]
    bytes_read, current, tables_num = 12,0,0
    huffman_tables = []
    while tables_num < blocks_num:
        table, len_arr = [],[]
        # Reading the ordered RLE encoding indices.
        while current != 65339:
            num_zeros, size = struct.unpack_from('>BB',
                                        raw_binary,
                                        offset=bytes_read)
            RLE = (num_zeros, size)
            table.append(RLE)
            bytes_read += 2
            current = struct.unpack_from('>H', 
                                         raw_binary,
                                         offset=bytes_read)[0]       
        bytes_read+=2
        current = struct.unpack_from('>H', 
                                     raw_binary,
                                     offset=bytes_read)[0]
        # Length array contains in the n position amount of
        # codes of n-bits.
        while current not in FLAGS.values():
            bucket = struct.unpack_from('>B',
                                        raw_binary,
                                        offset=bytes_read)[0]
            len_arr.append(bucket)

            bytes_read+=1
            current = struct.unpack_from('>H', 
                                     raw_binary,
                                     offset=bytes_read)[0]
        huffman_tables.append(
            [table,len_arr]
        )
        tables_num+=1
        bytes_read+=2
        current = struct.unpack_from('>H', 
                                     raw_binary,
                                     offset=bytes_read)[0]
              
    bytes_read+=2
    current = struct.unpack_from('>H',
                                 raw_binary,
                                 offset=bytes_read)[0]
    decoded_AC = []
    for i in range(blocks_num):
        compressed_block = []
        while current not in FLAGS.values():
            encoded_symbol,freq = struct.unpack_from('>Bh',
                                          raw_binary,
                                          offset=bytes_read)
            compressed_block.append([encoded_symbol,freq])

            bytes_read+=3
            current = struct.unpack_from('>H', 
                                         raw_binary,
                                         offset=bytes_read)[0]
        decoded_block = decode_AC_block(compressed_block,
                                        huffman_tables[i])
        bytes_read+=2
        current = struct.unpack_from('>H', 
                                     raw_binary,
                                     offset=bytes_read)[0]
        decoded_AC.append(decoded_block)
    
    current = struct.unpack_from('>H', 
                                 raw_binary,
                                 offset=bytes_read-2)[0]
    encoded_DC = []
    for _ in range(blocks_num):
        freq = struct.unpack_from('b',
                                  raw_binary,
                                  offset=bytes_read)[0]
        encoded_DC.append(freq)
        bytes_read+=1
    decoded_DC = np.array(encoded_DC).cumsum()

    blocks = []
    DCT_blocks = []
    for i,block in enumerate(decoded_AC):
        flattened_block = [decoded_DC[i]] + block
        block = revert_zigzag(flattened_block)
        block = block*Q
        DCT_blocks.append(block)
        block = vectorized_inverse_DCT(block).astype(np.int8)
        blocks.append(block)
    
    block_var = np.var(blocks,axis=(1,2))
    rimg = np.array(blocks).reshape((m,n))
    rimg = rimg + 128

    obj = Image.open('test\cat_raw.bmp').convert('L')
    oimg = np.array(obj)
    oimg = oimg[:m,:n] # We cropped before compression

    mse = np.mean((oimg - rimg)**2)
    max_I = oimg.max()
    psnr = 20 * np.log10(max_I) - 10 * np.log10(mse)


    # Paper of interest:
    # https://www.cse.iitb.ac.in/~ajitvr/CS754_Spring2017/dct_laplacian.pdf

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(top=0.8)

    axes[0,0].imshow(oimg,cmap=plt.get_cmap('gray'))
    axes[0,0].set_title("Original image")
    axes[0,1].imshow(rimg,cmap=plt.get_cmap('gray'))
    axes[0,1].set_title("Compressed image")
    axes[1,0].hist(decoded_DC,bins=200)
    axes[1,0].set_title("Histogram of DC frequency")
    axes[1,1].hist(block_var, bins=200)
    axes[1,1].set_title("Histogram of blocks variance")
    fig.tight_layout()
    fig.suptitle(f"PSNR: {psnr}",y=1)
    plt.show()

    
    
    


    



    

    
    

 
   