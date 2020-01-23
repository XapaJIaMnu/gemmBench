# gemmBench
int8_t gemm benchmark between Eigen, kpu's [intgemm](https://github.com/kpu/intgemm) and [mkl-dnn](https://github.com/intel/mkl-dnn)

## Compilation
- Download and compile [mkl-dnn](https://github.com/intel/mkl-dnn)
- Download and compile [intgemm](https://github.com/kpu/intgemm) 
- Install Eigen from your favourite provider

```
g++ bench.cpp ../intgemm/intgemm.cc -I../intgemm/build -ldnnl -march=native -O3
```

## Changing parameters
All the parameters are hardcoded
- The memory alignment can be changed here: https://github.com/XapaJIaMnu/gemmBench/blob/master/bench.cpp#L34
- Matrix sizes can be changed here: https://github.com/XapaJIaMnu/gemmBench/blob/master/bench.cpp#L57
- For intgemm to work, you need M and N to be a multiple of 8 and K to be a multiple of 32

## Caveats
The intgemms comparison is not a 100% fair, as it does less work. Eigen and MKL-dnn comparisons should be completely fair.

## OpenCL
There's a deprecated OpenCL version WiP in https://github.com/XapaJIaMnu/gemmBench/blob/master/bench_opencl_DEPRECATED.cpp

