 # from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((8192, 8192), "float16"), B: T.Buffer((8192, 8192), "float16"), C: T.Buffer((8192, 8192), "float16")):
        T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "llvm", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        # with T.block("root"):
        bx = T.launch_thread("blockIdx.x", 64)
        by = T.launch_thread("blockIdx.y", 64)
        v = T.launch_thread("threadIdx.x", 256)
        with T.block(""):
            T.reads(A[by * 128:by * 128 + 128, 0:8192], B[0:8192, bx * 128:bx * 128 + 128])
            T.writes(C[by * 128:by * 128 + 128, bx * 128:bx * 128 + 128])
            A_shared = T.alloc_buffer((3, 1, 16, 256), "float16", scope="shared.dyn")
            B_shared = T.alloc_buffer((3, 2, 4, 512), "float16", scope="shared.dyn")
            C_local = T.alloc_buffer((128,), scope="local")
            T.CreateListofMBarrierOp(128, 128, 128, 128, 128, 128)
            T.attr([128, 128], "kWarpSpecializationScope", 0)
            if v >= 128:
                T.SetMaxNReg(24, 0)
                for k in range(256):
                    T.MBarrierWaitParity(T.GetMBarrierOp(k % 3 + 3), T.bitwise_xor(k // 3 % 2, 1))
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(8):
                            A_shared[k % 3, 0, i * 4 + (v - 128) // 32, (v - 128) % 32 // 4 * 32 + ((v - 128) % 32 // 16 + (v - 128) % 4 // 2) % 2 * 16 + ((v - 128) % 16 // 8 + (v - 128) % 2) % 2 * 8 + vec] = A[by * 128 + i * 32 + (v - 128) // 4, k * 32 + (v - 128) % 4 * 8 + vec]
                    T.ptx_cp_async_barrier(T.GetMBarrierOp(k % 3))
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(8):
                            B_shared[k % 3, (v - 128) % 16 // 8, i, (v - 128) // 16 * 64 + ((v - 128) // 64 + (v - 128) % 8 // 4) % 2 * 32 + ((v - 128) % 64 // 32 + (v - 128) % 4 // 2) % 2 * 16 + ((v - 128) % 32 // 16 + (v - 128) % 2) % 2 * 8 + vec] = B[k * 32 + i * 8 + (v - 128) // 16, bx * 128 + (v - 128) % 16 * 8 + vec]
                    T.ptx_cp_async_barrier(T.GetMBarrierOp(k % 3))
                    T.ptx_arrive_barrier(T.GetMBarrierOp(k % 3))
            else:
                T.SetMaxNReg(240, 1)
                for i in T.unroll(128, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    C_local[i] = T.float32(0)
                for k in range(256):
                    T.MBarrierWaitParity(T.GetMBarrierOp(k % 3), k // 3 % 2)
                    T.call_extern("handle", "tl::gemm_ss<128, 128, 32, 2, 2, 0, 0>", T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, k % 3 * 4096, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, k % 3 * 4096, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 128, 3))
                    T.ptx_arrive_barrier(T.GetMBarrierOp(k % 3 + 3))
                for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    for vec in T.vectorized(2):
                        C[by * 128 + i % 8 // 2 * 32 + v % 64 // 32 * 16 + i % 2 * 8 + v % 32 // 4, bx * 128 + i // 8 * 16 + v // 64 * 8 + v % 4 * 2 + vec] = T.Cast("float16", C_local[i * 2 + vec])
