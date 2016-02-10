__author__ = 'lucas'
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

import numpy as np


def h_square_difference(h_array1, h_array2):
    h_array1 = np.array(h_array1).astype(np.float32)
    h_array2 = np.array(h_array2).astype(np.float32)
    array_size = h_array1.size
    d_array1 = cuda.mem_alloc(h_array1.size * h_array1.dtype.itemsize)
    d_array2 = cuda.mem_alloc(h_array2.size * h_array2.dtype.itemsize)
    cuda.memcpy_htod(d_array1, h_array1)
    cuda.memcpy_htod(d_array2, h_array2)
    #start = time.time()
    d_output = d_square_difference(d_array1, d_array2, np.int32(array_size))
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty_like(h_array1)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output


def d_square_difference(d_array1, d_array2, array_size):

    mod = SourceModule("""
        __global__ void squareDifference(float *input1, float *input2, const int size)
        {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if(idx < size)
          {
            float temp = input1[idx] - input2[idx];
            input1[idx] = temp * temp;
          }
        }
        """)

    func = mod.get_function("squareDifference")
    block_length = 1024
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(float(array_size) / block_length))
    grid_size = (grid_length, 1, 1)
    func(d_array1, d_array2, np.int32(array_size), block=block_size, grid=grid_size)
    d_output = d_array1
    return d_output

def h_copy0(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    d_input = cuda.mem_alloc(h_input.size * h_input.dtype.itemsize)
    d_output = cuda.mem_alloc(h_input.size * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    #start = time.time()
    d_output = d_copy0(d_output, d_input, block_length, array_size)
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty_like(h_input)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

def d_copy0(d_output, d_input, block_length, array_size):
    mod = SourceModule("""
        __global__ void copy(float *output, float *input)
        {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          output[idx] = input[idx];
        }
        """)
    func = mod.get_function("copy")
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(float(array_size) / block_length))
    grid_size = (grid_length, 1, 1)
    func(d_output, d_input, block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length)
    return d_output


def d_one(d_output, d_input, block_length, array_size):
    mod = SourceModule("""
        __global__ void copy(float *output, float *input)
        {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          output[idx] = 1.0;
        }
        """)
    func = mod.get_function("copy")
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(float(array_size) / block_length))
    grid_size = (grid_length, 1, 1)
    func(d_output, d_input, block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length)
    return d_output


def h_one(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    grid_length = int(np.ceil(float(array_size) / block_length))
    d_input = cuda.mem_alloc(array_size * h_input.dtype.itemsize)
    d_output = cuda.mem_alloc(grid_length * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    #start = time.time()
    d_one(d_output, d_input, block_length, array_size)
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty(grid_length).astype(np.float32)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

def h_reduce0(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    grid_length = int(np.ceil(float(array_size) / block_length))
    d_input = cuda.mem_alloc(array_size * h_input.dtype.itemsize)
    d_output = cuda.mem_alloc(grid_length * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    #start = time.time()
    d_reduce0(d_output, d_input, block_length, array_size)
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty(grid_length).astype(np.float32)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

def d_reduce0(d_output, d_input, block_length, array_size):
    mod = SourceModule("""
        __global__ void reduce(float *output, float *input, int size)
        {
          int local_idx = threadIdx.x;
          if (local_idx == 0){
            float temp = 0;
            int global_idx;
            for(int i = 0; i < blockDim.x; i++){
              global_idx = blockIdx.x*blockDim.x + i;
              if (global_idx >= size) break;
              temp += input[global_idx];
            }
            output[blockIdx.x] = temp;
          }
        }
        """)
    func = mod.get_function("reduce")
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(float(array_size) / block_length))
    grid_size = (grid_length, 1, 1)
    start = time.time()
    func(d_output, d_input, np.int32(array_size), block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length)
    end = time.time()
    print('reduce0: {0}'.format(end - start))
    return d_output

def h_reduce1(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    grid_length = int(np.ceil(float(array_size) / block_length))
    d_input = cuda.mem_alloc(array_size * h_input.dtype.itemsize)
    d_output = cuda.mem_alloc(grid_length * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    #start = time.time()
    d_reduce1(d_output, d_input, block_length, array_size)
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty(grid_length).astype(np.float32)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

def d_reduce1(d_output, d_input, block_length, array_size):
    mod = SourceModule("""
        __global__ void reduce1(float *output, float *input, int size)
        {

          unsigned int idx_x = threadIdx.x;
          unsigned int globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

          for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
            if (idx_x < s && globalIdx_x + s < size){
              input[globalIdx_x] += input[globalIdx_x + s];
            }
            __syncthreads();
          }

          if(idx_x == 0) {
            output[blockIdx.x] = input[globalIdx_x];
          }
        }
        """)
    func = mod.get_function("reduce1")
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(float(array_size) / block_length))
    grid_size = (grid_length, 1, 1)
    start = time.time()
    func(d_output, d_input, np.int32(array_size), block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length)
    end = time.time()
    print('reduce1: {0}'.format(end - start))
    return d_output

def h_reduce2(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    grid_length = int(np.ceil(float(array_size) / block_length))
    d_input = cuda.mem_alloc(array_size * h_input.dtype.itemsize)
    d_output = cuda.mem_alloc(grid_length * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    #start = time.time()
    d_reduce2(d_output, d_input, block_length, array_size)
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty(grid_length).astype(np.float32)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

def d_reduce2(d_output, d_input, block_length, array_size):
    mod = SourceModule("""
        __global__ void reduce2(float *output, float *input, int size)
        {
          extern __shared__ float shData[];
          unsigned int idx_x = threadIdx.x;
          unsigned int globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
          shData[idx_x] = input[globalIdx_x];
          __syncthreads();

          for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
            if (idx_x < s && globalIdx_x + s < size){
              shData[idx_x] += shData[idx_x + s];
            }
            __syncthreads();
          }

          if(idx_x == 0) {
            output[blockIdx.x] = shData[0];
          }
        }
        """)
    func = mod.get_function("reduce2")
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(float(array_size) / block_length))
    grid_size = (grid_length, 1, 1)
    start = time.time()
    func(d_output, d_input, np.int32(array_size), block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length)
    end = time.time()
    print('reduce2: {0}'.format(end - start))
    return d_output

def h_reduce3(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    grid_length = int(np.ceil(float(array_size) / block_length)) / 2
    d_input = cuda.mem_alloc(array_size * h_input.dtype.itemsize)
    d_output = cuda.mem_alloc(grid_length * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    #start = time.time()
    elapsed_time = d_reduce3(d_output, d_input, block_length, array_size)
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty(grid_length).astype(np.float32)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output, elapsed_time

def d_reduce3(d_output, d_input, block_length, array_size):
    mod = SourceModule("""
        __global__ void reduce3(float *output, float *input, int size)
        {
          extern __shared__ float shData[];
          unsigned int idx_x = threadIdx.x;
          unsigned int globalIdx_x = blockIdx.x * blockDim.x * 2+ threadIdx.x;
          shData[idx_x] = input[globalIdx_x];
          if(globalIdx_x + blockDim.x < size){
            shData[idx_x] += input[globalIdx_x + blockDim.x];
          }
          __syncthreads();

          for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
            if (idx_x < s && globalIdx_x + s < size){
              shData[idx_x] += shData[idx_x + s];
              __syncthreads();
            }
          }

          if(idx_x == 0) {
            output[blockIdx.x] = shData[0];
          }
        }
        """)
    func = mod.get_function("reduce3")
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(float(array_size) / block_length)) / 2
    grid_size = (grid_length, 1, 1)
    start = time.time()
    func(d_output, d_input, np.int32(array_size), block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length)
    end = time.time()
    elapsed_time = end - start
    print('reduce3: {0}'.format(end - start))
    return elapsed_time

def h_reduce(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    grid_length = int(np.ceil(int(np.ceil(float(array_size) / block_length)) / 2.0))
    d_input = cuda.mem_alloc(array_size * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    #start = time.time()
    elapsed_time = d_reduce(d_input, block_length, array_size)
    #end = time.time()
    #print('tiempo kernel sin paso de memoria : {0}'.format(end - start))
    h_output = np.empty(grid_length).astype(np.float32)
    cuda.memcpy_dtoh(h_output, d_input)
    return h_output, elapsed_time

def d_reduce(d_input, block_length, array_size):
    mod = SourceModule("""
        __global__ void reduce(float *input, int size)
        {
          extern __shared__ float shData[];
          unsigned int idx_x = threadIdx.x;
          unsigned int globalIdx_x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
          shData[idx_x] = input[globalIdx_x];
          if(globalIdx_x + blockDim.x < size){
            shData[idx_x] += input[globalIdx_x + blockDim.x];
          }
          __syncthreads();

          for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
            if (idx_x < s && globalIdx_x + s < size){
              shData[idx_x] += shData[idx_x + s];
              __syncthreads();
            }
          }

          if(idx_x == 0) {
            input[blockIdx.x] = shData[0];
          }
        }
        """)
    func = mod.get_function("reduce")
    block_size = (block_length, 1, 1)
    grid_length = int(np.ceil(int(np.ceil(float(array_size) / block_length)) / 2.0))
    print 'grid_length',
    print grid_length
    grid_size = (grid_length, 1, 1)
    start = time.time()
    func(d_input, np.int32(array_size), block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length)
    end = time.time()
    elapsed_time = end - start
    print('reduce: {0}'.format(end - start))
    return elapsed_time

def h_reduce_axis_x(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    num_rows, num_cols = h_input.shape
    grid_length_x = int(np.ceil(int(np.ceil(float(num_cols) / block_length)) / 2.0))
    d_input = cuda.mem_alloc(num_rows * num_cols * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    d_reduce_axis_x(d_input, block_length, num_rows, num_cols)
    h_output = np.empty(grid_length_x * num_rows).astype(np.float32).reshape((num_rows, grid_length_x))
    cuda.memcpy_dtoh(h_output, d_input)
    return h_output


def d_reduce_axis_x(d_input, block_length_x, num_rows, num_cols):
    mod = SourceModule("""
        __global__ void reduceAxisX(float *input, const int gridLengthX, const int numCols)
        {
          extern __shared__ float shData[];
          unsigned int idx_x = threadIdx.x;
          unsigned int globalIdx_x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
          unsigned int globalIdx = blockIdx.y * numCols + globalIdx_x;
          shData[idx_x] = input[globalIdx];
          if(globalIdx_x + blockDim.x < numCols){
            shData[idx_x] += input[globalIdx + blockDim.x];
          }
          __syncthreads();

          for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
            if (idx_x < s && globalIdx_x + s < numCols){
              shData[idx_x] += shData[idx_x + s];
              __syncthreads();
            }
          }

          if(idx_x == 0) {
            input[blockDim.y * gridLengthX + blockIdx.x] = shData[0];
          }
        }
        """)
    func = mod.get_function("reduceAxisX")
    block_size = (block_length_x, 1, 1)
    grid_length_x = int(np.ceil(int(np.ceil(float(num_cols) / block_length_x)) / 2.0))
    grid_size = (grid_length_x, num_rows, 1)
    start = time.time()
    func(d_input, np.int32(grid_length_x), np.int32(num_cols), block=block_size, grid=grid_size, shared= np.float32().itemsize*block_length_x)
    end = time.time()
    elapsed_time = end - start
    print('reduce: {0}'.format(end - start))
    return elapsed_time

def h_reduce_all(h_input):
    block_length = 32
    h_input = np.array(h_input).astype(np.float32)
    array_size = h_input.size
    d_input = cuda.mem_alloc(array_size * h_input.dtype.itemsize)
    cuda.memcpy_htod(d_input, h_input)
    while array_size > 1:
        grid_length = np.ceil(int(np.ceil(float(array_size) / block_length)) / 2.0)
        print 'grid length',
        print grid_length
        d_reduce(d_input, block_length, array_size)
        array_size = grid_length
    h_output = np.empty(grid_length).astype(np.float32)
    cuda.memcpy_dtoh(h_output, d_input)
    return h_output[0]

def main():
    arr = np.arange(9).reshape((3,3)).astype(np.float32)
    print arr

    s = h_reduce_axis_x(arr)
    print s

if __name__ == "__main__":
    main()