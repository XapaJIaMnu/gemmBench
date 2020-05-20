#include "aligned.h"
#include "libs/FBGEMM/include/fbgemm/Fbgemm.h"
//#include "libs/FBGEMM/include/fbgemm/FbgemmSpMM.h"
#include <chrono>


//copy/paste from fbgemm source code
namespace fbgemm {
template <typename T>
void transpose_matrix(
    int M,
    int N,
    const T* src,
    int ld_src,
    T* dst,
    int ld_dst) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      dst[i * ld_dst + j] = src[i + j * ld_src];
    }
  } // for each output row
}
/*
std::chrono::duration<double> fbgemmSPMTimes(alloc::AlignedVector<uint8_t>& A,
                 alloc::AlignedVector<int8_t>& B,
                 alloc::AlignedVector<int32_t>& C,
                 int m,
                 int n,
                 int k) {
  auto aptr = A.get();
  auto bptr = B.get();

  int ldat = m;
  int ldbt = k;
  int ldct = m;

  // Transpose as if A is float so 4 columns are interleaved
  // transpose_matrix(m, k / 4, aData.data(), k / 4, atData.data(), ldat);
  // aligned_vector<float> btData(n * ldbt);
  // auto btptr = reinterpret_cast<int8_t*>(btData.data());
  // transpose_matrix(k, n, bptr, n, btptr, ldbt);
  // auto cDataJIT = getRandomSparseVector(n * ldct);
  // auto cptrJIT = reinterpret_cast<int32_t*>(cDataJIT.data());

  auto fn = generateSpMM<int32_t>(n, m, k, bptr, ldbt, ldat, ldct);
  auto start = std::chrono::system_clock::now();
  fn(aptr, C.get(), 0 ); // accum_flag 
  auto end = std::chrono::system_clock::now();

  return (end-start);
}*/


inline void col_offsets_with_zero_pt_s8acc32(
    bool transpose,
    int K,
    int N,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
    }
    col_offsets[n] = sum - B_zero_point[n / ncols_per_quant_group] * K;
  }
}

std::chrono::duration<double> fbgemmPackedTimes(alloc::AlignedVector<uint8_t>& A,
                 alloc::AlignedVector<int8_t>& B,
                 alloc::AlignedVector<int32_t>& C,
                 int m,
                 int n,
                 int k) {
  auto aptr = A.get();
  auto bptr = B.get();

  int ldat = m;
  int ldbt = k;
  int ldct = m;


matrix_op_t atrans = matrix_op_t::NoTranspose;
matrix_op_t btrans = matrix_op_t::NoTranspose; //matrix_op_t::NoTranspose;

PackAWithRowOffset<uint8_t> packAN(
              atrans,
              m,
              k,
              A.get(),
              (atrans == matrix_op_t::Transpose) ? m : k);

PackBMatrix<int8_t> packedBN(
          btrans,
          k,
          n,
          B.get(),
          (btrans == matrix_op_t::Transpose) ? k : n);

  //Some crap bogus bulshit values
  int32_t Aint8_zero_point = 43;
  float Aint8_scale = 0.11;

  alloc::AlignedVector<int32_t> Bint8_zero_point(n, 256);
  for (int i = 0; i < n; i++) {
    Bint8_zero_point[i] = 23;
  }

  alloc::AlignedVector<float> Bint8_scale(n, 256);
  for (int i = 0; i < n; i++) {
    Bint8_scale[i] = 0.43;
  }

  alloc::AlignedVector<int32_t> col_offsets(n, 256);
  
  col_offsets_with_zero_pt_s8acc32(
              false,
              k,
              n,
              B.get(),
              Bint8_zero_point.get(),
              col_offsets.get(),
              1);


  DoNothing<float, float> doNothingObj{};

  ReQuantizeForFloat<false> outputProcObj(
              doNothingObj,
              Aint8_scale,
              Bint8_scale.get(),
              Aint8_zero_point,
              Bint8_zero_point.get(),
              packAN.getRowOffsetBuffer(),
              col_offsets.get(),
              nullptr,
              n);

  //DoNothing<int32_t, int32_t> doNothingObj{};
  //memCopy<> outputProcObj(doNothingObj);
  auto start = std::chrono::system_clock::now();
  fbgemmPacked(
    packAN,
    packedBN,
    reinterpret_cast<float*>(C.get()),
    (int32_t*)C.get(),
    (int32_t)n,
    outputProcObj,
    0,
    1);
  auto end = std::chrono::system_clock::now();

  return (end-start);
}



}//namespace fbgemm

