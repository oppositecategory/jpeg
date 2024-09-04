import struct


FLAGS = {
    'SOI': 65496, # Start Of Image
    'SOB': 65475, # Start Of Block
    'SOAC': 65477, # Start Of AC
    'SODC': 65481 # Start Of DC
}

def encode_JPEG_format(filename, encoded_DC, encoded_AC, huffman_tables):
    bytes = []
    with open(filename, 'wb') as file:
        # Write SOI marker
        file.write(b'\xFF\xD8')
                
        # Huffman tables segment
        file.write(b'\xFF\xC4')
        file.write(len(huffman_tables).to_bytes(2, 'big')) 
        for table in huffman_tables:
            file.write(b'\xFF\xC3')
            for key,value in table.items():
                key_bytes = struct.pack('BB',*key) 
                value_bytes = int(value,2).to_bytes((len(value)+7)//8,byteorder='big')
                bytes.append((len(value)+7)//8)

                file.write(key_bytes)
                file.write(b"\x3A")
                file.write(value_bytes)
        
        # AC frequencies segment
        file.write(b'\xFF\xC5')
        for i,block in enumerate(encoded_AC):
            start_of_block = b'\xFF\xC3' 
            file.write(start_of_block)
            for frequency in block:
                freq_bytes = int(frequency,2).to_bytes((len(frequency)+7)//8,byteorder='big')
                file.write(freq_bytes)
         
        # DC frequencies segment
        file.write(b'\xFF\xC9')  
        for diff in encoded_DC:
            packed_diff = struct.pack('b',diff)
            file.write(packed_diff)
        
        # Write EOI marker
        file.write(b'\xFF\xD9')


def decode_JPEG_format(filename):
    with open(filename,'rb') as file:
        raw_binary = file.read()

    blocks_num = struct.unpack_from('>H',raw_binary, offset=4)[0]
    bytes_read, current, tables_num = 8,0,0
    huffman_tables = []
    while tables_num < blocks_num:
        table = {}
        while current not in FLAGS.values():
            num_zeros, size, _, code = struct.unpack_from('<BBBB',
                                        raw_binary,
                                        offset=bytes_read)
            RLE = (num_zeros, size)
            table[RLE] = code
            bytes_read +=4
            current = struct.unpack_from('>H', 
                                    raw_binary,
                                        offset=bytes_read)[0]
        
        tables_num+=1
        bytes_read+=2
        huffman_tables.append(table)
        current = struct.unpack_from('>H', 
                                     raw_binary,
                                     offset=bytes_read+2)[0]
    
    current = struct.unpack_from('>H',
                                 raw_binary,
                                 offset=bytes_read)
    encoded_AC = []
    for _ in range(blocks_num):
        block = []
        while current not in FLAGS.values():
            AC_freq = struct.unpack_from('>B', 
                                         raw_binary,
                                         offset=bytes_read)[0]
            block.append(AC_freq)

            bytes_read+=1
            current = struct.unpack_from('>H', 
                                         raw_binary,
                                         offset=bytes_read)[0]
            
        bytes_read+=2
        current = struct.unpack_from('>H', 
                                     raw_binary,
                                     offset=bytes_read)[0]
        encoded_AC.append(block)

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
    
    block = encoded_AC[0]
    print(block)   



    

    
    

 
   