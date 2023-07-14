from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes



# Test cast from float32 to float16

x = Tensor([1.1, 0], dtype=dtypes.float32)
y = x.cast(dtypes.int32)
print(x.numpy())
print(y.numpy())
exit()


import numpy as np
from tinygrad.runtime.ops_cuda import CUDAProgram, RawCUDABuffer

prg = CUDAProgram("cast", """
.version 7.8
.target sm_61
.address_size 64
.visible .entry test(
    .param .u64 buf0, 
    .param .u64 buf1
) {
    .reg .u64 %B<2>;
    .reg .s32 %i<2>; // %i0 = ctaid.x, %i1 = %i0 * 4
    .reg .u64 %b<3>;
    .reg .f32 %f<1>;
                  
    ld.param.u64 %B0, [buf0];
    ld.param.u64 %B1, [buf1];
                  
    mov.u32 %i0, %ctaid.x;
    mul.lo.s32 %i1, %i0, 4;
    cvt.u64.s32 %b0, %i1;
    add.u64 %b1, %b0, %B1;
    ld.global.f32 %f0, [%b1+0];
    add.u64 %b2, %b0, %B0;
    st.global.f32 [%b2+0], %f0;
    ret;
}
""", binary=True)

input = RawCUDABuffer.fromCPU(np.array([1.0], np.float32))
prg([1,1,1], [1,1,1], input)
print(input.toCPU())



#if __name__ == "__main__":
#  
#
#
#  test = RawCUDABuffer.fromCPU(np.zeros(10, np.float32))
#  prg = CUDAProgram("test", """
#  .version 7.8
#  .target sm_61
#  .address_size 64
#  .visible .entry test(.param .u64 x) {
#    .reg .b32       %r<2>;
#    .reg .b64       %rd<3>;
#
#    ld.param.u64    %rd1, [x];
#    cvta.to.global.u64      %rd2, %rd1;
#    mov.u32         %r1, 0x40000000; // 2.0 in float
#    st.global.u32   [%rd2], %r1;
#    ret;
#  }""", binary=True)
#  prg([1], [1], test)
#  print(test.toCPU())

