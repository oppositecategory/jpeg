# Mini JPEG

This repo contains implementation of image compression algorithm based on JPEG pipeline in Python for educational purporses. 

A vectorized implementation of 2-D DCT is done using a batch multiplication of each 8x8 block matrix with a 3-D tensor containing the DCT basis matrices for each spatial frequency.
Then I use a quantization matrix to reduce resolution for higher-frequencies values and encode them using Run-Length format for later compression using Huffman Encoding.


Still in work.
