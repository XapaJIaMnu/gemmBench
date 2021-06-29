#include <array>
#include <vector>
#include "Eigen/Dense"
#include <iostream>
#include "dnnl.h"
#include <chrono>
#include "intgemm/intgemm.h"
#include "aligned.h"
#include <unordered_map>
#include "fbgemm_tests.h"

#ifdef WITH_MKL
#  include <mkl.h>
#endif


void printDNNLStatus(dnnl_status_t& status) {
  if (status == dnnl_success) {
      std::cout << "DNNL success." << std::endl;
  } else if (status == dnnl_out_of_memory ) {
      std::cout << "The operation failed due to an out-of-memory condition." << std::endl;
  } else if (status == dnnl_invalid_arguments ) {
      std::cout << "The operation failed because of incorrect function arguments." << std::endl;
  } else if (status == dnnl_unimplemented) {
      std::cout << "The operation failed because requested functionality is not implemented." << std::endl;
  } else if (status == dnnl_iterator_ends) {
      std::cout << "Primitive iterator passed over last primitive descriptor." << std::endl;
  } else if (status == dnnl_iterator_ends) {
      std::cout << "Primitive or engine failed on execution." << std::endl;
  } else if (status == dnnl_not_required) {
      std::cout << "Queried element is not required for given primitive." << std::endl;
  }
}

struct matrix_size {
   const int M;
   const int K;
   const int N;

   friend std::ostream& operator<<(std::ostream& os, const matrix_size& m) {
    os << "Matrix size: M: " << m.M << " K: " << m.K << " N: " << m.N;
    return os;
   }
};

enum Arch { ssse3, avx2, avx512, avx512vnni, any };

static std::unordered_map<std::string, Arch> ArchMap = {
   {"ssse3", ssse3}, {"avx2", avx2}, {"avx512", avx512}, {"avx512vnni", avx512vnni}, {"any", any},
   {"SSSE3", ssse3}, {"AVX2", avx2}, {"AVX512", avx512}, {"AVX512VNNI", avx512vnni}, {"ANY", any},
};

template<Arch A> struct archInfo;

template<> struct archInfo<Arch::ssse3> {
  using intgemm_ = intgemm::SSSE3::Kernels8;
  using intgemmShift_ = intgemm::SSSE3::Kernels8;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_t::dnnl_cpu_isa_sse41;
#ifdef WITH_MKL
  int mkl_ = MKL_ENABLE_SSE4_2;
#endif
  std::string name = "SSSE3";
};

template<> struct archInfo<Arch::avx2> {
  using intgemm_ = intgemm::AVX2::Kernels8;
  using intgemmShift_ = intgemm::AVX2::Kernels8;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_avx2;
#ifdef WITH_MKL
  int mkl_ = MKL_ENABLE_AVX2;
#endif
  std::string name = "AVX2";
};

template<> struct archInfo<Arch::avx512> {
  using intgemm_ = intgemm::AVX512BW::Kernels8;;
  using intgemmShift_ = intgemm::AVX512BW::Kernels8;;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_avx512_core;
#ifdef WITH_MKL
  int mkl_ = MKL_ENABLE_AVX512;
#endif
  std::string name = "AVX512";
};

template<> struct archInfo<Arch::avx512vnni> {
  using intgemm_ = intgemm::AVX512VNNI::Kernels8;
  using intgemmShift_ = intgemm::AVX512VNNI::Kernels8;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_avx512_core_vnni;
#ifdef WITH_MKL
  int mkl_ = MKL_ENABLE_AVX512_E1;
#endif
  std::string name = "AVX512VNNI";
};

template<> struct archInfo<Arch::any> {
  using intgemm_ = intgemm::Int8;
  using intgemmShift_ = intgemm::Int8Shift;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_all;
#ifdef WITH_MKL
  int mkl_ = -1;
#endif
  std::string name = "any";
};


template<Arch A>
std::ostream& operator<<(std::ostream& os, const archInfo<A>& a) {
  os << a.name;
  return os;
}

template<Arch architecture>
void benchmarkLoop(int iterations, std::vector<matrix_size>& matrices, const size_t align, bool use_fbgemm, bool use_eigen, bool use_fp32) {

  archInfo<architecture> myarch;
  auto arch_status = dnnl_set_max_cpu_isa(myarch.dnnl_);

  if (arch_status != dnnl_success) {
    std::cerr << "We couldn't set arch: " << std::endl;
    printDNNLStatus(arch_status);
    return;
  }

#ifdef WITH_MKL
  if (myarch.mkl_ >= 0)
    mkl_enable_instructions(myarch.mkl_);
#endif


  std::chrono::duration<double> eigen_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> dnnl_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> dnnlU_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> dnnlS_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> dnnl32_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mkl_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mkl32_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kenn_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kennU_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> fbgemm_duration_loop = std::chrono::duration<double>::zero();
  //std::chrono::duration<double> fbgemmSPM_duration_loop = std::chrono::duration<double>::zero();

  for (auto&& sizes : matrices) {

    char offsetc = 'F';
    bool zero_oa = 1;
    bool zero_ob = 1;
    bool zero_oc = 0;
    char transA = 'N';
    char transB = 'N';
    const int M = sizes.M;
    const int K = sizes.K;
    const int N = sizes.N;
    float alpha = 1;
    float beta = 1;
    int lda = K;
    int ldb = N;
    int ldc = N;
    int8_t oa = 0;
    int8_t ob = 0;
    std::array<int32_t, 1> oc = {0};

    for (int i = 0; i<iterations + 1; i++) {

      //Construct matrices

      Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> A = Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic>::Random(M,K);
      Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> B = Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic>::Random(K,N);
      Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> C = Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic>::Random(M,N);

      //EIGEN
      if (use_eigen) {
        Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic> eigen_A_tmp = A.cast<int32_t>();
        Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic> eigen_B_tmp = B.cast<int32_t>();

        // Copy onto aligned memory
        alloc::AlignedVector<int32_t> A_EIGEN(M*K, align);
        alloc::AlignedVector<int32_t> B_EIGEN(K*N, align);
        alloc::AlignedVector<int32_t> C_EIGEN(M*N, align);

        std::copy(eigen_A_tmp.data(), eigen_A_tmp.data() + eigen_A_tmp.size(), A_EIGEN.get());
        std::copy(eigen_B_tmp.data(), eigen_B_tmp.data() + eigen_B_tmp.size(), B_EIGEN.get());
        std::copy(C.data(), C.data() + C.size(), C_EIGEN.get());

        //Eigen bug: https://stackoverflow.com/questions/54738495/eigenmapd-matrix-from-raw-buffer-gives-object-allocated-on-stack-is-too-big/
        Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_a(A_EIGEN.get(), M, K);
        Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_b(B_EIGEN.get(), K, N);
        Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_c(C_EIGEN.get(), M, N);



        auto eigen_start = std::chrono::system_clock::now();
        eigen_c.noalias() += (eigen_a*(int)alpha)*(eigen_b*(int)beta);
        auto eingen_end = std::chrono::system_clock::now();
        eigen_duration_loop += (eingen_end - eigen_start);
      }

      //MKL-DNN
      // Copy onto aligned memory
      alloc::AlignedVector<int8_t> A_DNNL(M*K, align);
      alloc::AlignedVector<int8_t> B_DNNL(K*N, align);
      alloc::AlignedVector<int32_t> C_DNNL(M*N, align);


      std::copy(A.data(), A.data() + A.size(), A_DNNL.get());
      std::copy(B.data(), B.data() + B.size(), B_DNNL.get());
      std::copy(C.data(), C.data() + C.size(), C_DNNL.get());

      auto dnnl_start = std::chrono::system_clock::now();

      auto status = dnnl_gemm_s8s8s32(transA, transB, offsetc,
              M, N, K, alpha, A_DNNL.get(), lda, oa, B_DNNL.get(), ldb, ob,
              beta, C_DNNL.get(), ldc, oc.data());
      auto dnnl_end = std::chrono::system_clock::now();

      dnnl_duration_loop += (dnnl_end - dnnl_start);
      if (status != dnnl_success) {
        std::cerr << "we died at " << i << std::endl;
        printDNNLStatus(status);
        break;
      }

#ifdef WITH_MKL
      {
        alloc::AlignedVector<int8_t> A_MKL(M*K, align);
        alloc::AlignedVector<int8_t> B_MKL(K*N, align);
        alloc::AlignedVector<int32_t> C_MKL(M*N, align);

        std::copy(A.data(), A.data() + A.size(), A_MKL.get());
        std::copy(B.data(), B.data() + B.size(), B_MKL.get());
        std::copy(C.data(), C.data() + C.size(), C_MKL.get());

        auto mkl_start = std::chrono::system_clock::now();
        cblas_gemm_s8u8s32(CblasRowMajor,
                           transA == 'N' ? CblasNoTrans : CblasTrans,
                           transB == 'N' ? CblasNoTrans : CblasTrans,
                           CblasFixOffset,
                           M, N, K,
                           alpha,
                           A_MKL.get(), lda, oa,
                           B_MKL.get(), ldb, ob,
                           beta,
                           C_MKL.get(), ldc, oc.data());
        auto mkl_end = std::chrono::system_clock::now();

        mkl_duration_loop += (mkl_end - mkl_start);
      }

      if (use_fp32) { // MKLcblas_Sgemm
        Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic> eigen_A_tmp = A.cast<float>();
        Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic> eigen_B_tmp = B.cast<float>();

        alloc::AlignedVector<float> A_MKL(M*K, align);
        alloc::AlignedVector<float> B_MKL(K*N, align);
        alloc::AlignedVector<float> C_MKL(M*N, align);

        std::copy(eigen_A_tmp.data(), eigen_A_tmp.data() + eigen_A_tmp.size(), A_MKL.get());
        std::copy(eigen_B_tmp.data(), eigen_B_tmp.data() + eigen_B_tmp.size(), B_MKL.get());
        std::copy(C.data(), C.data() + C.size(), C_MKL.get());

        auto mkl_start = std::chrono::system_clock::now();
        cblas_sgemm(CblasRowMajor,
                           transA == 'N' ? CblasNoTrans : CblasTrans,
                           transB == 'N' ? CblasNoTrans : CblasTrans,
                           /*CblasFixOffset,*/
                           M, N, K,
                           alpha,
                           A_MKL.get(), lda,// oa,
                           B_MKL.get(), ldb,// ob,
                           beta,
                           C_MKL.get(), ldc);// oc.data());
        auto mkl_end = std::chrono::system_clock::now();

        mkl32_duration_loop += (mkl_end - mkl_start);
      }
#endif

      if (use_fp32) { // oneDNN cblas_Sgemm
        Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic> eigen_A_tmp = A.cast<float>();
        Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic> eigen_B_tmp = B.cast<float>();

        alloc::AlignedVector<float> A_MKL(M*K, align);
        alloc::AlignedVector<float> B_MKL(K*N, align);
        alloc::AlignedVector<float> C_MKL(M*N, align);

        std::copy(eigen_A_tmp.data(), eigen_A_tmp.data() + eigen_A_tmp.size(), A_MKL.get());
        std::copy(eigen_B_tmp.data(), eigen_B_tmp.data() + eigen_B_tmp.size(), B_MKL.get());
        std::copy(C.data(), C.data() + C.size(), C_MKL.get());

        auto dnnlfp32_start = std::chrono::system_clock::now();
        cblas_sgemm(CblasRowMajor,
                           transA == 'N' ? CblasNoTrans : CblasTrans,
                           transB == 'N' ? CblasNoTrans : CblasTrans,
                           /*CblasFixOffset,*/
                           M, N, K,
                           alpha,
                           A_MKL.get(), lda,// oa,
                           B_MKL.get(), ldb,// ob,
                           beta,
                           C_MKL.get(), ldc);// oc.data());
        auto dnnlfp32_end = std::chrono::system_clock::now();

        dnnl32_duration_loop += (dnnlfp32_end - dnnlfp32_start);
      }

      //Now intgemm
      Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> kenneth_a_tmp = A.cast<float>();
      Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> kenneth_b_tmp = B.cast<float>();

      alloc::AlignedVector<float> A_proto(M * K, align);
      alloc::AlignedVector<float> B_proto(K * N, align);

      std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_proto.get());
      std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_proto.get());


      float quant_mult = 127.0 / 2.0;
      alloc::AlignedVector<int8_t> A_prepared(M * K, align);
      alloc::AlignedVector<int8_t> B_prepared(K * N, align);

      archInfo<architecture>::intgemm_::PrepareA(A_proto.get(), A_prepared.get(), quant_mult, M, K);
      // Quantize and reshape B.
      // Typically you will do this once when parameters are loaded, not every time.
      archInfo<architecture>::intgemm_::PrepareB(B_proto.get(), B_prepared.get(), quant_mult, K, N);

      alloc::AlignedVector<float> C_kenn(M*N, align);

      auto kenn_start = std::chrono::system_clock::now();
      archInfo<architecture>::intgemm_::Multiply(A_prepared.get(), B_prepared.get(), M, K, N, intgemm::callbacks::UnquantizeAndWrite(1.0 / (quant_mult * quant_mult), C_kenn.get()));
      auto kenn_end = std::chrono::system_clock::now();

      kenn_duration_loop += (kenn_end - kenn_start);
          
      //MKL-DNN SignedXunsigned
      // Copy onto aligned memory
      alloc::AlignedVector<uint8_t> A1_DNNL(M*K, align);
      alloc::AlignedVector<int8_t> B1_DNNL(K*N, align);
      alloc::AlignedVector<int32_t> C1_DNNL(M*N, align);


      std::copy(A.data(), A.data() + A.size(), A1_DNNL.get());
      std::copy(B.data(), B.data() + B.size(), B1_DNNL.get());
      std::copy(C.data(), C.data() + C.size(), C1_DNNL.get());

      auto dnnlU_start = std::chrono::system_clock::now();

      auto status1 = dnnl_gemm_u8s8s32(transA, transB, offsetc,
              M, N, K, alpha, A1_DNNL.get(), lda, oa, B1_DNNL.get(), ldb, ob,
              beta, C1_DNNL.get(), ldc, oc.data());
      auto dnnlU_end = std::chrono::system_clock::now();

      dnnlU_duration_loop += (dnnlU_end - dnnlU_start);
      if (status1 != dnnl_success) {
        std::cerr << "we died at " << i << std::endl;
        printDNNLStatus(status1);
        break;
      }

      //Now intgemm shifted
      alloc::AlignedVector<float> A_proto1(M * K, align);
      alloc::AlignedVector<float> B_proto1(K * N, align);
      alloc::AlignedVector<float> inputBias(K, align);
      std::fill(inputBias.get(), inputBias.get() + K, 0.0f);

      std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_proto1.get());
      std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_proto1.get());


      //float quant_mult = 127.0 / 2.0;
      alloc::AlignedVector<int8_t> A_prepared1(M * K, align); //@TODO API CHANGE
      alloc::AlignedVector<int8_t> B_prepared1(K * N, align);

      archInfo<architecture>::intgemmShift_::PrepareA(A_proto1.get(), A_prepared1.get(), quant_mult, M, K);
      // Quantize and reshape B.
      // Typically you will do this once when parameters are loaded, not every time.
      archInfo<architecture>::intgemmShift_::PrepareB(B_proto1.get(), B_prepared1.get(), quant_mult, K, N);

      float unquant_mult_forprep = (-1)*(2.0)*(2.0)/(127.0f);

      //PrepareBias
      archInfo<architecture>::intgemmShift_::PrepareBias(B_prepared1.get(), K, N, intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, inputBias.get(), inputBias.get()));

      alloc::AlignedVector<float> C_kenn1(M*N, align);

      auto kennU_start = std::chrono::system_clock::now();
      archInfo<architecture>::intgemmShift_::Multiply(A_prepared1.get(), B_prepared1.get(), M, K, N, intgemm::callbacks::UnquantizeAndAddBiasAndWrite(1.0 / (quant_mult * quant_mult), inputBias.get(), C_kenn1.get()));
      auto kennU_end = std::chrono::system_clock::now();

      kennU_duration_loop += (kennU_end - kennU_start);

      //DNNLDNN Single precision
      alloc::AlignedVector<float> A_DNNL_S(M*K, align);
      alloc::AlignedVector<float> B_DNNL_S(K*N, align);
      alloc::AlignedVector<float> C_DNNL_S(M*N, align);

      std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_DNNL_S.get());
      std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_DNNL_S.get());
      std::copy(C.data(), C.data() + C.size(), C_DNNL_S.get());

      auto dnnlS_start = std::chrono::system_clock::now();

      auto status2 = dnnl_sgemm(transA, transB,
              M, N, K, alpha, A_DNNL_S.get(), lda, B_DNNL_S.get(), ldb,
              beta, C_DNNL_S.get(), ldc);
      auto dnnlS_end = std::chrono::system_clock::now();

      dnnlS_duration_loop += (dnnlS_end - dnnlS_start);
      if (status2 != dnnl_success) {
        std::cerr << "we died at " << i << std::endl;
        printDNNLStatus(status2);
        break;
      }

      if (use_fbgemm) {
        //Now fbgemm
        alloc::AlignedVector<uint8_t> A_FBGEMM(M*K, align);
        alloc::AlignedVector<int8_t> B_FBGEMM(K*N, align);
        alloc::AlignedVector<int32_t> C_FBGEMM(M*N, align);


        std::copy(A.data(), A.data() + A.size(), A_FBGEMM.get());
        std::copy(B.data(), B.data() + B.size(), B_FBGEMM.get());
        std::copy(C.data(), C.data() + C.size(), C_FBGEMM.get());

        fbgemm_duration_loop += fbgemm::fbgemmPackedTimes(A_FBGEMM, B_FBGEMM, C_FBGEMM, M, N, K);

        //And fbgemm again

        alloc::AlignedVector<uint8_t> A_FBGEMM1(M*K, align);
        alloc::AlignedVector<int8_t> B_FBGEMM1(K*N, align);
        alloc::AlignedVector<int32_t> C_FBGEMM1(M*N, align);


        std::copy(A.data(), A.data() + A.size(), A_FBGEMM1.get());
        std::copy(B.data(), B.data() + B.size(), B_FBGEMM1.get());
        std::copy(C.data(), C.data() + C.size(), C_FBGEMM1.get());


        //fbgemmSPM_duration_loop += fbgemm::fbgemmSPMTimes(A_FBGEMM1, B_FBGEMM1, C_FBGEMM1, M, N, K);
      }
      /*First dnnl and fbgemm calls are slow, so ignore results from the first run of the loop*/
      if (i == 0) {
        eigen_duration_loop = std::chrono::duration<double>::zero();
        dnnl_duration_loop = std::chrono::duration<double>::zero();
        dnnlU_duration_loop = std::chrono::duration<double>::zero();
        dnnlS_duration_loop = std::chrono::duration<double>::zero();
        mkl_duration_loop = std::chrono::duration<double>::zero();
        kenn_duration_loop = std::chrono::duration<double>::zero();
        kennU_duration_loop = std::chrono::duration<double>::zero();
        fbgemm_duration_loop = std::chrono::duration<double>::zero();
        //fbgemmSPM_duration_loop = std::chrono::duration<double>::zero();
      }
    }
    std::cout << std::fixed;
    std::cout.precision(10);
    std::cout << "Arch: " << myarch << std::endl << sizes << " in loop, for " << iterations << " interations:" << std::endl;
    if (use_eigen)
      std::cout <<"      Eigen i32gemm took: " << eigen_duration_loop.count() << " seconds." << std::endl;

    std::cout <<  "  dnnl s8s8s32 gemm took: " << dnnl_duration_loop.count() << " seconds." << std::endl <<
                  "  dnnl u8s8s32 gemm took: " << dnnlU_duration_loop.count() << " seconds." << std::endl <<
                  "         dnnl sgemm took: " << dnnlS_duration_loop.count() << " seconds." << std::endl;
#ifdef WITH_MKL
      std::cout <<" cblas_gemm_s8u8s32 took: " << mkl_duration_loop.count() << " seconds." << std::endl;
    if (use_fp32)
      std::cout <<"    MKL cblas_sgemm took: " << mkl32_duration_loop.count() << " seconds." << std::endl;
#endif
    if (use_fp32)
      std::cout <<"   DNNL cblas_sgemm took: " << dnnl32_duration_loop.count() << " seconds." << std::endl;
  std::cout <<    "            Intgemm took: " << kenn_duration_loop.count() << " seconds." << std::endl <<
                  "    Intgemm Shifted took: " << kennU_duration_loop.count() << " seconds." << std::endl;
    if (use_fbgemm) {
      std::cout << 
                  //"fbgemm SparseXDense took: " << fbgemmSPM_duration_loop.count() << " seconds." << std::endl <<
                  "      fbgemm Packed took: " << fbgemm_duration_loop.count() << " seconds." << std::endl;
    }
                  
                     std::cout << "Alignment was: " << align << "." << std::endl;

  }
 }


int main(int argc, char const *argv[]) {

  //auto status = dnnl_set_max_cpu_isa(dnnl_cpu_isa_avx512_core);

  const size_t align = 256;

  int iterations = 1000;
  bool use_eigen = false;
  Arch myarch = any;
  if (argc == 1) {
    iterations = 1000;
    use_eigen = false;
  } else if (argc == 2) {
    iterations = std::atoi(argv[1]);
  } else if (argc == 3) {
    iterations = std::atoi(argv[1]);
    std::string archArg = std::string(argv[2]);
    if (ArchMap.find(archArg) != ArchMap.end()) {
      myarch = ArchMap[archArg];
    } else {
      std::cerr << "Unrecognised arch: " << archArg << std::endl << "Available options: ssse3 avx2 avx512 avx512vnni any" << std::endl;
      std::exit(1);
    }
  } else if (argc == 4) {
    iterations = std::atoi(argv[1]);
    std::string archArg = std::string(argv[2]);
    if (ArchMap.find(archArg) != ArchMap.end()) {
      myarch = ArchMap[archArg];
    } else {
      std::cerr << "Unrecognised arch: " << archArg << std::endl << "Available options: ssse3 avx2 avx512 avx512vnni any" << std::endl;
      std::exit(1);
    }
    use_eigen = std::atoi(argv[3]);
  } else {
    std::cerr << "Usage: " << argv[0] << " [iterations=1000] [arch=any] [use_eigen=0]" << std::endl;
    std::exit(1);
  }

  bool use_fp32 = false; // Compare the 32bit thingies.

  std::vector<matrix_size> matrices = {
    {1024, 1024, 1024},
    {256, 10368, 256},
    {256, 5312, 256},
    {8, 2048, 256},
    {320, 256, 256},
    {472, 256, 256},
    {248, 256, 256},
    {200, 256, 256},
    {1, 64, 8}};//zero, one, two, three, four, five, six, seven, eight};

  //fbgemm only supports AVX2 and above and doesn't support architecture limitations
  bool use_fbgemm = true;
  if (myarch != any) {
    use_fbgemm = false;
    std::cout << "Fbgemm tests will not run, because you requested a specific architecture and this is not supported by fbgemm." << std::endl;
  }

  if (intgemm::kCPU < intgemm::CPUType::AVX2) {
    use_fbgemm = false;
    std::cout << "Fbgemm tests will not run, because the architecture doesn't support it." << std::endl;
  }


  if (myarch==ssse3) {
    benchmarkLoop<ssse3>(iterations, matrices, align, use_fbgemm, use_eigen, use_fp32);
  } else if (myarch==avx2) {
    benchmarkLoop<avx2>(iterations, matrices, align, use_fbgemm, use_eigen, use_fp32);
  } else if (myarch==avx512) {
    benchmarkLoop<avx512>(iterations, matrices, align, use_fbgemm, use_eigen, use_fp32);
  } else if (myarch==avx512vnni) {
    benchmarkLoop<avx512vnni>(iterations, matrices, align, use_fbgemm, use_eigen, use_fp32);
  } else if (myarch==any) {
    benchmarkLoop<any>(iterations, matrices, align, use_fbgemm, use_eigen, use_fp32);
  }

  return 0;
}
