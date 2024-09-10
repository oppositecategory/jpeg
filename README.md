# Mini JPEG

This repo contains implementation of image compression algorithm based on JPEG pipeline in Python for educational purporses. 

A vectorized implementation of 2D DCT is done using a batch Hadamard multiplication of an 8x8 block matrix with a 3D tensor containing the DCT basis matrices for each spatial frequency.
The compression works currently but still to be optimized.

**To do:**
- Codec can be improved by handling variable-length bytes for AC frequencies instead of hardcoded short to reduce overhead.
- Refactor the codec saving format to reduce redundant symbols.
- Add Numba for processing the DCT (optional).
- Add multi-processing; allocate subsets of blocks for each core and process the subset of blocks by multi-threading. For large images the compression is slow.
- Re-write the compression pipeline in C++ for better performance.
