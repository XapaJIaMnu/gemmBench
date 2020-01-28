# gemmBench
int8_t gemm benchmark between Eigen, kpu's [intgemm](https://github.com/kpu/intgemm), [dnnl](https://github.com/intel/mkl-dnn) and [fbgemm](https://github.com/pytorch/FBGEMM/)

## Compilation
```
mkdir build
cd build
cmake ..
make -j
```
## Usage
```
./benchmark [iterations=100] [arch=any] [use_eigen=0]
```

## Changing parameters
Some paramters are hardcoded
- The memory alignment can be changed here: https://github.com/XapaJIaMnu/gemmBench/blob/master/bench.cpp#L338
- Matrix sizes can be changed here: https://github.com/XapaJIaMnu/gemmBench/blob/master/bench.cpp#L372
- For intgemm to work, you need M and N to be a multiple of 8 and K to be a multiple of 32
- The number of iterations of the loop can be varied through command one.
- You can limit `arch` for `intgemm` and `dnnl`. Supported values: `ssse3`, `avx2`, `avx512`, `avx512vnni` and `any`
- Since `Eigen` is a lot slower than the other two, its execution is disabled by default. To enable it, provide the argument.

## Caveats
- Fbgemm only supports AVX2 processors or newer, so the test is skipped on older architectures.
- Fbgemm doesn't allow for limiting the arch type, so the test is skipped in case explicit arch is requested

