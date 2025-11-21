#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(256) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  __shared__ uint64_t _mbarrier[6];
  if (((int)threadIdx.x) == 0) {
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 128);
    tl::mbarrier_init(_mbarrier[5], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 256; ++k) {
      tl::mbarrier_wait(_mbarrier[((k % 3) + 3)], (((k % 6) / 3) ^ 1));
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((k % 3) * 4096) + (i * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) - 1024)) = *(uint4*)(A + ((((((((int)blockIdx.y) * 1048576) + (i * 262144)) + ((((int)threadIdx.x) >> 2) * 8192)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) - 262144));
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[(k % 3)]);
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((k % 3) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 11776)) = *(uint4*)(B + ((((((k * 262144) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 65536));
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[(k % 3)]);
      tl::mbarrier_arrive(_mbarrier[(k % 3)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_2 = 0; i_2 < 128; ++i_2) {
      C_local[i_2] = 0.000000e+00f;
    }
    for (int k_1 = 0; k_1 < 256; ++k_1) {
      tl::mbarrier_wait(_mbarrier[(k_1 % 3)], ((k_1 % 6) / 3));
      tl::gemm_ss<128, 128, 32, 2, 2, 0, 0>((&(((half_t*)buf_dyn_shmem)[((k_1 % 3) * 4096)])), (&(((half_t*)buf_dyn_shmem)[(((k_1 % 3) * 4096) + 12288)])), (&(C_local[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 % 3) + 3)]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 64; ++i_3) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_3 * 2));
      ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
      ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
      *(uint1*)(C + (((((((((((int)blockIdx.y) * 1048576) + (((i_3 & 7) >> 1) * 262144)) + (((((int)threadIdx.x) & 63) >> 5) * 131072)) + ((i_3 & 1) * 65536)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + (((int)blockIdx.x) * 128)) + ((i_3 >> 3) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
    }
  }
}
