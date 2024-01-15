from numba import cuda
import numpy as np

@cuda.jit
def simple_kernel(result_array):
    pos = cuda.grid(1)
    if pos < result_array.size:
        result_array[pos] = pos

n = 10  # Size of the array
result_array_gpu = cuda.device_array(n, dtype=np.int32)

# Adjust these values based on your GPU and problem size
threads_per_block = 2048  # This is a typical starting point
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

simple_kernel[blocks_per_grid, threads_per_block](result_array_gpu)

result_array = result_array_gpu.copy_to_host()
print(result_array)
