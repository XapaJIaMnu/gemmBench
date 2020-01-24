#include <array>
#include <vector>
#include "Eigen/Dense"
#include <iostream>
#include "dnnl.h"
#include <chrono>
#include "intgemm.h"
#include "aligned.h"
#include <unordered_map>


void printDNNLStatus(dnnl_status_t& status) {
  if (status == dnnl_success) {
      std::cout << "MKL success." << std::endl;
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
  using intgemm_ = intgemm::SSSE3_8bit;
  using intgemmShift_ = intgemm::SSSE3_8bit;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_t::dnnl_cpu_isa_sse41;
  std::string name = "SSSE3";
};

template<> struct archInfo<Arch::avx2> {
  using intgemm_ = intgemm::AVX2_8bit;
  using intgemmShift_ = intgemm::AVX2_8bit;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_avx2;
  std::string name = "AVX2";
};

template<> struct archInfo<Arch::avx512> {
  using intgemm_ = intgemm::AVX512_8bit;
  using intgemmShift_ = intgemm::AVX512_8bit;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_avx512_core;
  std::string name = "AVX512";
};

template<> struct archInfo<Arch::avx512vnni> {
  using intgemm_ = intgemm::AVX512VNNI_8bit;
  using intgemmShift_ = intgemm::AVX512VNNI_8bit;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_avx512_core_vnni;
  std::string name = "AVX512VNNI";
};

template<> struct archInfo<Arch::any> {
  using intgemm_ = intgemm::Int8;
  using intgemmShift_ = intgemm::Int8Shift;
  dnnl_cpu_isa_t dnnl_ = dnnl_cpu_isa_all;
  std::string name = "any";
};


template<Arch A>
std::ostream& operator<<(std::ostream& os, const archInfo<A>& a) {
  os << a.name;
  return os;
}

template<Arch architecture>
void benchmarkLoop(int iterations, std::vector<matrix_size>& matrices, const size_t align, bool use_eigen) {

  archInfo<architecture> myarch;
  auto arch_status = dnnl_set_max_cpu_isa(myarch.dnnl_);

  if (arch_status != dnnl_success) {
    std::cerr << "We couldn't set arch: " << std::endl;
    printDNNLStatus(arch_status);
    return;
  }


  std::chrono::duration<double> eigen_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mkl_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklU_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklS_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kenn_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kennU_duration_loop = std::chrono::duration<double>::zero();

  for (auto&& sizes : matrices) {

    char offsetc = 'F';
    bool zero_oa = 1;
    bool zero_ob = 1;
    bool zero_oc = 0;
    char transA = 'N';
    char transB = 'n';
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
      alloc::AlignedVector<int8_t> A_MKL(M*K, align);
      alloc::AlignedVector<int8_t> B_MKL(K*N, align);
      alloc::AlignedVector<int32_t> C_MKL(M*N, align);


      std::copy(A.data(), A.data() + A.size(), A_MKL.get());
      std::copy(B.data(), B.data() + B.size(), B_MKL.get());
      std::copy(C.data(), C.data() + C.size(), C_MKL.get());

      auto mkl_start = std::chrono::system_clock::now();

      auto status = dnnl_gemm_s8s8s32(transA, transB, offsetc,
              M, N, K, alpha, A_MKL.get(), lda, oa, B_MKL.get(), ldb, ob,
              beta, C_MKL.get(), ldc, oc.data());
      auto mkl_end = std::chrono::system_clock::now();

      mkl_duration_loop += (mkl_end - mkl_start);
      if (status != dnnl_success) {
        std::cerr << "we died at " << i << std::endl;
        printDNNLStatus(status);
        break;
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
      alloc::AlignedVector<uint8_t> A1_MKL(M*K, align);
      alloc::AlignedVector<int8_t> B1_MKL(K*N, align);
      alloc::AlignedVector<int32_t> C1_MKL(M*N, align);


      std::copy(A.data(), A.data() + A.size(), A1_MKL.get());
      std::copy(B.data(), B.data() + B.size(), B1_MKL.get());
      std::copy(C.data(), C.data() + C.size(), C1_MKL.get());

      auto mklU_start = std::chrono::system_clock::now();

      auto status1 = dnnl_gemm_u8s8s32(transA, transB, offsetc,
              M, N, K, alpha, A1_MKL.get(), lda, oa, B1_MKL.get(), ldb, ob,
              beta, C1_MKL.get(), ldc, oc.data());
      auto mklU_end = std::chrono::system_clock::now();

      mklU_duration_loop += (mklU_end - mklU_start);
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

      //MKLDNN Single precision
      alloc::AlignedVector<float> A_MKL_S(M*K, align);
      alloc::AlignedVector<float> B_MKL_S(K*N, align);
      alloc::AlignedVector<float> C_MKL_S(M*N, align);

      std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_MKL_S.get());
      std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_MKL_S.get());
      std::copy(C.data(), C.data() + C.size(), C_MKL_S.get());

      auto mklS_start = std::chrono::system_clock::now();

      auto status2 = dnnl_sgemm(transA, transB,
              M, N, K, alpha, A_MKL_S.get(), lda, B_MKL_S.get(), ldb,
              beta, C_MKL_S.get(), ldc);
      auto mklS_end = std::chrono::system_clock::now();

      mklS_duration_loop += (mklS_end - mklS_start);
      if (status2 != dnnl_success) {
        std::cerr << "we died at " << i << std::endl;
        printDNNLStatus(status2);
        break;
      }


      /*First mkl call is slow, so ignore results from the first run of the loop*/
      if (i == 0) {
        eigen_duration_loop = std::chrono::duration<double>::zero();
        mkl_duration_loop = std::chrono::duration<double>::zero();
        mklU_duration_loop = std::chrono::duration<double>::zero();
        mklS_duration_loop = std::chrono::duration<double>::zero();
        kenn_duration_loop = std::chrono::duration<double>::zero();
        kennU_duration_loop = std::chrono::duration<double>::zero();
      }
    }
    std::cout << std::fixed;
    std::cout.precision(10);
    std::cout << "Arch: " << myarch << std::endl << sizes << " in loop, for " << iterations << " interations:" << std::endl;
    if (use_eigen)
      std::cout <<"    Eigen i32gemm took: " << eigen_duration_loop.count() << " seconds." << std::endl;

    std::cout <<  "dnnl s8s8s32 gemm took: " << mkl_duration_loop.count() << " seconds." << std::endl <<
                  "dnnl u8s8s32 gemm took: " << mklU_duration_loop.count() << " seconds." << std::endl <<
                  "       dnnl sgemm took: " << mklS_duration_loop.count() << " seconds." << std::endl <<
                  "          Intgemm took: " << kenn_duration_loop.count() << " seconds." << std::endl <<
                  "  Intgemm Shifted took: " << kennU_duration_loop.count() << " seconds." << std::endl <<
                      "Alignment was: " << align << "." << std::endl;

  }
 }


int main(int argc, char const *argv[]) {

  //auto status = dnnl_set_max_cpu_isa(dnnl_cpu_isa_avx512_core);

  std::chrono::duration<double> eigen_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mkl_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklU_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklS_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kenn_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kennU_duration_loop = std::chrono::duration<double>::zero();
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


  if (myarch==ssse3) {
    benchmarkLoop<ssse3>(iterations, matrices, align, use_eigen);
  } else if (myarch==avx2) {
    benchmarkLoop<avx2>(iterations, matrices, align, use_eigen);
  } else if (myarch==avx512) {
    benchmarkLoop<avx512>(iterations, matrices, align, use_eigen);
  } else if (myarch==avx512vnni) {
    benchmarkLoop<avx512vnni>(iterations, matrices, align, use_eigen);
  } else if (myarch==any) {
    benchmarkLoop<any>(iterations, matrices, align, use_eigen);
  }

  return 0;
}
