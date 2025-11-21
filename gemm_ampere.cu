#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(128) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 128; ++i) {
    C_local[i] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((int)blockIdx.y) * 1048576) + (i_1 * 262144)) + ((((int)threadIdx.x) >> 2) * 8192)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), B+((((i_2 * 65536) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_3 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), A+(((((((int)blockIdx.y) * 1048576) + (i_3 * 262144)) + ((((int)threadIdx.x) >> 2) * 8192)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+(((((i_4 * 65536) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 254; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((k + 2) % 3) * 8192) + (i_5 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((((int)blockIdx.y) * 1048576) + (i_5 * 262144)) + ((((int)threadIdx.x) >> 2) * 8192)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((k + 2) % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), B+((((((k * 262144) + (i_6 * 65536)) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 524288));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    tl::gemm_ss<128, 128, 32, 2, 2, 0, 0>((&(((half_t*)buf_dyn_shmem)[((k % 3) * 4096)])), (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 4096) + 12288)])), (&(C_local[0])));
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<128, 128, 32, 2, 2, 0, 0>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[20480])), (&(C_local[0])));
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<128, 128, 32, 2, 2, 0, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[12288])), (&(C_local[0])));
  #pragma unroll
  for (int i_7 = 0; i_7 < 64; ++i_7) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_7 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 1048576) + (((i_7 & 7) >> 1) * 262144)) + (((((int)threadIdx.x) & 63) >> 5) * 131072)) + ((i_7 & 1) * 65536)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + (((int)blockIdx.x) * 128)) + ((i_7 >> 3) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
}