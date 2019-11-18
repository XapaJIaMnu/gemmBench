# gemmBench
int8_t gemm benchmark between Eigen, kpu's [intgemm](https://github.com/kpu/intgemm) and [mkl-dnn](https://github.com/intel/mkl-dnn)

## Compilation
- Download and compile [mkl-dnn](https://github.com/intel/mkl-dnn)
- Download and compile [intgemm](https://github.com/kpu/intgemm) 
- Install Eigen from your favourite provider

```
g++ bench.cpp -lmkldnn -O3 -L./ -lintgemm -march=native
```

## Changing parameters
All the parameters are hardcoded
- The memory alignment can be changed here: https://github.com/XapaJIaMnu/gemmBench/blob/master/bench.cpp#L90
- Matrix sizes can be changed here: https://github.com/XapaJIaMnu/gemmBench/blob/master/bench.cpp#L101
- For intgemm to work, you need M and N to be a multiple of 8 and K to be a multiple of 32

## Caveats
The intgemms comparison is not a 100% fair, as it does less work. Eigen and MKL-dnn comparisons should be completely fair.
