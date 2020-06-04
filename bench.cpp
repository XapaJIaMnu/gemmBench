#include <array>
#include <vector>
#include "Eigen/Dense"
#include <iostream>
#include "dnnl.h"
#include <chrono>
#include "intgemm.h"
#include "aligned.h"
#include <unordered_map>
#include "fbgemm_tests.h"


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
void benchmarkLoop(int iterations, std::vector<matrix_size>& matrices, const size_t align, bool use_fbgemm, bool use_eigen) {

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
  std::chrono::duration<double> fbgemm_duration_loop = std::chrono::duration<double>::zero();
  //std::chrono::duration<double> fbgemmSPM_duration_loop = std::chrono::duration<double>::zero();

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
      alloc::AlignedVector<float> inputBias(K*2, align);
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
      /*First mkl and fbgemm calls are slow, so ignore results from the first run of the loop*/
      if (i == 0) {
        eigen_duration_loop = std::chrono::duration<double>::zero();
        mkl_duration_loop = std::chrono::duration<double>::zero();
        mklU_duration_loop = std::chrono::duration<double>::zero();
        mklS_duration_loop = std::chrono::duration<double>::zero();
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

    std::cout <<  "  dnnl s8s8s32 gemm took: " << mkl_duration_loop.count() << " seconds." << std::endl <<
                  "  dnnl u8s8s32 gemm took: " << mklU_duration_loop.count() << " seconds." << std::endl <<
                  "         dnnl sgemm took: " << mklS_duration_loop.count() << " seconds." << std::endl <<
                  "            Intgemm took: " << kenn_duration_loop.count() << " seconds." << std::endl <<
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

  std::chrono::duration<double> eigen_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mkl_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklU_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklS_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kenn_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kennU_duration_loop = std::chrono::duration<double>::zero();
  const size_t align = 64;

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
{32,256,256},
{32,1536,256},
{32,256,1536},
{1,256,256},
{31,256,256},
{2,256,256},
{3,256,256},
{30,256,256},
{6,256,256},
{28,256,256},
{1,1536,256},
{1,256,1536},
{29,256,256},
{26,256,256},
{4,256,256},
{9,256,256},
{8,256,256},
{27,256,256},
{12,256,256},
{11,256,256},
{5,256,256},
{31,1536,256},
{31,256,1536},
{10,256,256},
{7,256,256},
{17,256,256},
{25,256,256},
{23,256,256},
{19,256,256},
{14,256,256},
{24,256,256},
{2,1536,256},
{2,256,1536},
{21,256,256},
{20,256,256},
{16,256,256},
{13,256,256},
{22,256,256},
{18,256,256},
{15,256,256},
{3,1536,256},
{3,256,1536},
{30,1536,256},
{30,256,1536},
{6,1536,256},
{6,256,1536},
{576,256,256},
{28,1536,256},
{28,256,1536},
{29,1536,256},
{29,256,1536},
{992,256,256},
{928,256,256},
{832,256,256},
{640,256,256},
{26,1536,256},
{26,256,1536},
{736,256,256},
{704,256,256},
{544,256,256},
{4,1536,256},
{4,256,1536},
{416,256,256},
{32,13568,256},
{1120,256,256},
{9,1536,256},
{9,256,1536},
{8,1536,256},
{8,256,1536},
{32,12864,256},
{896,256,256},
{768,256,256},
{512,256,256},
{27,1536,256},
{27,256,1536},
{1216,256,256},
{1024,256,256},
{12,1536,256},
{12,256,1536},
{5,1536,256},
{5,256,1536},
{32,12480,256},
{32,11840,256},
{11,1536,256},
{11,256,1536},
{10,1536,256},
{10,256,1536},
{7,1536,256},
{7,256,1536},
{32,11840,256},
{32,12288,256},
{32,11648,256},
{17,1536,256},
{17,256,1536},
{32,11968,256},
{32,11520,256},
{25,1536,256},
{25,256,1536},
{23,1536,256},
{23,256,1536},
{19,1536,256},
{19,256,1536},
{14,1536,256},
{14,256,1536},
{32,11712,256},
{32,11264,256},
{32,10944,256},
{24,1536,256},
{24,256,1536},
{960,256,256},
{800,256,256},
{672,256,256},
{448,256,256},
{352,256,256},
{32,10688,256},
{32,10432,256},
{288,256,256},
{256,256,256},
{21,1536,256},
{21,256,1536},
{20,1536,256},
{20,256,1536},
{32,11264,256},
{32,10944,256},
{864,256,256},
{78,256,256},
{608,256,256},
{32,8128,256},
{32,10816,256},
{16,1536,256},
{16,256,1536},
{13,1536,256},
{13,256,1536},
{1088,256,256},
{1056,256,256},
{32,10112,256},
{32,10752,256},
{22,1536,256},
{22,256,1536},
{18,1536,256},
{18,256,1536},
{32,9920,256},
{32,9472,256},
{32,10432,256},
{32,10240,256},
{32,10048,256},
{480,256,256},
{384,256,256},
{3328,256,256},
{32,9792,256},
{32,10368,256},
{2176,256,256},
{1984,256,256},
{1856,256,256},
{1728,256,256},
{1664,256,256},
{1600,256,256},
{15,1536,256},
{15,256,1536},
{1568,256,256},
{1504,256,256},
{1472,256,256},
{1440,256,256},
{1408,256,256},
{1376,256,256},
{1344,256,256},
{1312,256,256},
{1280,256,256},
{1248,256,256},
{1184,256,256},
{1152,256,256},
{32,9984,256},
{32,9792,256},
{32,9728,256},
{32,9280,256},
{72,256,256},
{504,256,256},
{32,9600,256},
{32,9280,256},
{224,256,256},
{32,9216,256},
{32,8896,256},
{32,8640,256},
{364,256,256},
{32,8768,256},
{32,8704,256},
{32,8320,256},
{270,256,256},
{240,256,256},
{208,256,256},
{1,13568,256},
{180,256,256},
{132,256,256},
{32,8448,256},
{32,8320,256},
{32,8256,256},
{32,8064,256},
{32,7936,256},
{899,256,256},
{52,256,256},
{48,256,256},
{450,256,256},
{408,256,256},
{360,256,256},
{312,256,256},
{160,256,256},
{144,256,256},
{136,256,256},
{108,256,256},
{104,256,256},
{32,7616,256},
{32,7936,256},
{32,7360,256},
{32,7168,256},
{32,7104,256},
{992,1536,256},
{992,256,1536},
{928,1536,256},
{928,256,1536},
{90,256,256},
{896,1536,256},
{896,256,1536},
{870,256,256},
{840,256,256},
{837,256,256},
{832,1536,256},
{832,256,1536},
{806,256,256},
{76,256,256},
{768,1536,256},
{768,256,1536},
{736,1536,256},
{736,256,1536},
{728,256,256},
{704,1536,256},
{704,256,1536},
{700,256,256},
{696,256,256},
{676,256,256},
{66,256,256},
{640,1536,256},
{640,256,1536},
{630,256,256},
{62,256,256},
{620,256,256},
{60,256,256},
{600,256,256},
{58,256,256},
{588,256,256},
{576,1536,256},
{576,256,1536},
{560,256,256},
{546,256,256},
{544,1536,256},
{544,256,1536},
{532,256,256},
{512,1536,256},
{512,256,1536},
{496,256,256},
{460,256,256},
{456,256,256},
{442,256,256},
{416,1536,256},
{416,256,1536},
{414,256,256},
{38,256,256},
{378,256,256},
{372,256,256},
{36,256,256},
{336,256,256},
{32,6848,256},
{32,6528,256},
{32,6400,256},
{308,256,256},
{280,256,256},
{248,256,256},
{210,256,256},
{200,256,256},
{1,11968,256},
{198,256,256},
{174,256,256},
{150,256,256},
{130,256,256},
{1216,1536,256},
{1216,256,1536},
{1120,1536,256},
{1120,256,1536},
{1092,256,256},
{105,256,256},
{102,256,256},
{1024,1536,256},
{1024,256,1536},
{32,6464,256},
{32,6400,256},
{32,6272,256},
{32,5632,256},
{32,5824,256},
{980,256,256},
{96,256,256},
{961,256,256},
{936,256,256},
{930,256,256},
{92,256,256},
{924,256,256},
{868,256,256},
{84,256,256},
{816,256,256},
{812,256,256},
{810,256,256},
{80,256,256},
{784,256,256},
{775,256,256},
{770,256,256},
{75,256,256},
{756,256,256},
{725,256,256},
{720,256,256},
{714,256,256},
{713,256,256},
{70,256,256},
{690,256,256},
{68,256,256},
{665,256,256},
{660,256,256},
{651,256,256},
{648,256,256},
{646,256,256},
{644,256,256},
{638,256,256},
{624,256,256},
{598,256,256},
{594,256,256},
{570,256,256},
{558,256,256},
{552,256,256},
{550,256,256},
{54,256,256},
{540,256,256},
{527,256,256},
{525,256,256},
{522,256,256},
{520,256,256},
{510,256,256},
{500,256,256},
{493,256,256},
{484,256,256},
{476,256,256},
{465,256,256},
{464,256,256},
{44,256,256},
{440,256,256},
{435,256,256},
{434,256,256},
{42,256,256},
{420,256,256},
{40,256,256},
{406,256,256},
{403,256,256},
{400,256,256},
{396,256,256},
{374,256,256},
{35,256,256},
{350,256,256},
{34,256,256},
{340,256,256},
{33,256,256},
{330,256,256},
{320,256,256},
{319,256,256},
{315,256,256},
{310,256,256},
{300,256,256},
{276,256,256},
{272,256,256},
{252,256,256},
{228,256,256},
{216,256,256},
{204,256,256},
{1,11712,256},
{192,256,256},
{186,256,256},
{1768,256,256},
{168,256,256},
{1674,256,256},
{165,256,256},
{1612,256,256},
{156,256,256},
{152,256,256},
{1456,256,256},
{140,256,256},
{1404,256,256},
{1364,256,256},
{135,256,256},
{1352,256,256},
{1350,256,256},
{1334,256,256},
{1302,256,256},
{126,256,256},
{124,256,256},
{1240,256,256},
{1218,256,256},
{117,256,256},
{1170,256,256},
{114,256,256},
{1144,256,256},
{1102,256,256},
{1085,256,256},
{1054,256,256},
{1050,256,256},
{1040,256,256},
{1026,256,256},
{1015,256,256},
{100,256,256},
{1008,256,256},
{32,5632,256},
{32,4800,256},
{31,13568,256},
{2,11648,256},
{960,1536,256},
{960,256,1536},
{864,1536,256},
{864,256,1536},
{800,1536,256},
{800,256,1536},
{78,1536,256},
{78,256,1536},
{672,1536,256},
{672,256,1536},
{608,1536,256},
{608,256,1536},
{480,1536,256},
{480,256,1536},
{448,1536,256},
{448,256,1536},
{384,1536,256},
{384,256,1536},
{352,1536,256},
{352,256,1536},
{3328,1536,256},
{3328,256,1536},
{32,4096,256},
{32,3328,256},
{31,11264,256},
{2,8768,256},
{288,1536,256},
{288,256,1536},
{256,1536,256},
{256,256,1536},
{2176,1536,256},
{2176,256,1536},
{1,9600,256},
{1,8640,256},
{1,11264,256},
{1,10944,256},
{1984,1536,256},
{1984,256,1536},
{1856,1536,256},
{1856,256,1536},
{1728,1536,256},
{1728,256,1536},
{1664,1536,256},
{1664,256,1536},
{1600,1536,256},
{1600,256,1536},
{1568,1536,256},
{1568,256,1536},
{1504,1536,256},
{1504,256,1536},
{1472,1536,256},
{1472,256,1536},
{1440,1536,256},
{1440,256,1536},
{1408,1536,256},
{1408,256,1536},
{1376,1536,256},
{1376,256,1536},
{1344,1536,256},
{1344,256,1536},
{1312,1536,256},
{1312,256,1536},
{1280,1536,256},
{1280,256,1536},
{1248,1536,256},
{1248,256,1536},
{1184,1536,256},
{1184,256,1536},
{1152,1536,256},
{1152,256,1536},
{1088,1536,256},
{1088,256,1536},
{1056,1536,256},
{1056,256,1536},
{31,10368,256},
{1,11264,256},
{1,10432,256},
{99,256,256},
{999,256,256},
{990,256,256},
{98,256,256},
{988,256,256},
{987,256,256},
{986,256,256},
{975,256,256},
{957,256,256},
{952,256,256},
{950,256,256},
{94,256,256},
{943,256,256},
{931,256,256},
{925,256,256},
{920,256,256},
{918,256,256},
{903,256,256},
{900,256,256},
{897,256,256},
{893,256,256},
{88,256,256},
{884,256,256},
{882,256,256},
{880,256,256},
{875,256,256},
{874,256,256},
{861,256,256},
{855,256,256},
{851,256,256},
{841,256,256},
{82,256,256},
{820,256,256},
{819,256,256},
{817,256,256},
{799,256,256},
{798,256,256},
{792,256,256},
{783,256,256},
{782,256,256},
{77,256,256},
{759,256,256},
{750,256,256},
{748,256,256},
{744,256,256},
{741,256,256},
{722,256,256},
{6,13568,256},
{69,256,256},
{693,256,256},
{682,256,256},
{680,256,256},
{675,256,256},
{65,256,256},
{656,256,256},
{650,256,256},
{64,256,256},
{63,256,256},
{637,256,256},
{629,256,256},
{612,256,256},
{602,256,256},
{580,256,256},
{575,256,256},
{572,256,256},
{56,256,256},
{551,256,256},
{529,256,256},
{528,256,256},
{51,256,256},
{518,256,256},
{517,256,256},
{513,256,256},
{50,256,256},
{507,256,256},
{506,256,256},
{492,256,256},
{47,256,256},
{473,256,256},
{470,256,256},
{46,256,256},
{468,256,256},
{462,256,256},
{45,256,256},
{459,256,256},
{444,256,256},
{43,256,256},
{429,256,256},
{418,256,256},
{410,256,256},
{39,256,256},
{392,256,256},
{390,256,256},
{387,256,256},
{37,256,256},
{377,256,256},
{361,256,256},
{357,256,256},
{351,256,256},
{345,256,256},
{344,256,256},
{342,256,256},
{338,256,256},
{333,256,256},
{329,256,256},
{325,256,256},
{324,256,256},
{322,256,256},
{3224,256,256},
{31,7936,256},
{31,12288,256},
{31,11968,256},
{3120,256,256},
{30,12480,256},
{304,256,256},
{2,9216,256},
{2,7936,256},
{299,256,256},
{297,256,256},
{296,256,256},
{294,256,256},
{2912,256,256},
{289,256,256},
{286,256,256},
{285,256,256},
{282,256,256},
{2808,256,256},
{279,256,256},
{2704,256,256},
{26,11968,256},
{264,256,256},
{261,256,256},
{260,256,256},
{258,256,256},
{255,256,256},
{253,256,256},
{246,256,256},
{245,256,256},
{243,256,256},
{242,256,256},
{2392,256,256},
{238,256,256},
{234,256,256},
{232,256,256},
{231,256,256},
{230,256,256},
{225,256,256},
{222,256,256},
{221,256,256},
{220,256,256},
{215,256,256},
{2108,256,256},
{2080,256,256},
{205,256,256},
{2040,256,256},
{203,256,256},
{1,9984,256},
{1,9792,256},
{1,8768,256},
{1,6400,256},
{1,10048,256},
{1976,256,256},
{1972,256,256},
{190,256,256},
{189,256,256},
{188,256,256},
{1860,256,256},
{185,256,256},
{184,256,256},
{1836,256,256},
{1798,256,256},
{175,256,256},
{1740,256,256},
{171,256,256},
{162,256,256},
{1620,256,256},
{161,256,256},
{1566,256,256},
{1564,256,256},
{1550,256,256},
{1508,256,256},
{1500,256,256},
{1496,256,256},
{147,256,256},
{1470,256,256},
{145,256,256},
{1450,256,256},
{1426,256,256},
{1421,256,256},
{1392,256,256},
{1380,256,256},
{1360,256,256},
{1333,256,256},
{1320,256,256},
{1316,256,256},
{1300,256,256},
{129,256,256},
{1296,256,256},
{1292,256,256},
{1290,256,256},
{128,256,256},
{1274,256,256},
{1271,256,256},
{1269,256,256},
{1260,256,256},
{1247,256,256},
{1242,256,256},
{123,256,256},
{1215,256,256},
{120,256,256},
{1209,256,256},
{1204,256,256},
{1200,256,256},
{119,256,256},
{1189,256,256},
{1188,256,256},
{1178,256,256},
{1175,256,256},
{116,256,256},
{1150,256,256},
{1148,256,256},
{1147,256,256},
{1140,256,256},
{1134,256,256},
{1131,256,256},
{112,256,256},
{1125,256,256},
{111,256,256},
{1110,256,256},
{1104,256,256},
{1080,256,256},
{1078,256,256},
{1073,256,256},
{1066,256,256},
{1053,256,256},
{1044,256,256},
{1034,256,256},
{1032,256,256},
{1025,256,256},
{1023,256,256},
{1012,256,256},
{1000,256,256},
{8,10944,256},
{4,8640,256},
{4,7360,256},
{3,9600,256},
{3,9472,256},
{3,11648,256},
{3,10816,256},
{31,9920,256},
{31,9792,256},
{31,8768,256},
{31,8704,256},
{31,8128,256},
{31,7168,256},
{31,6400,256},
{31,10752,256},
{31,10432,256},
{2,9600,256},
{2,7168,256},
{2,5632,256},
{2,13568,256},
{2,12480,256},
{27,13568,256},
{27,12864,256},
{27,10432,256},
{1,9280,256},
{1,8896,256},
{1,12864,256},
{1,11840,256},
{13,1280,256},
{11,10752,256},
{9,13568,256},
{9,11840,256},
{8,13568,256},
{8,1280,256},
{8,11264,256},
{7,6848,256},
{7,12864,256},
{7,12288,256},
{7,10048,256},
{6,9728,256},
{6,6272,256},
{6,11968,256},
{5,9792,256},
{4,9792,256},
{4,12480,256},
{4,11520,256},
{4,10432,256},
{3,9984,256},
{3,8128,256},
{3,7616,256},
{3,5824,256},
{3,12288,256},
{3,11840,256},
{3,11264,256},
{3,10944,256},
{3,10368,256},
{3,10240,256},
{31,9792,256},
{31,9600,256},
{31,9280,256},
{31,8896,256},
{31,7616,256},
{31,6848,256},
{31,6272,256},
{31,10944,256},
{31,10816,256},
{31,10240,256},
{30,9216,256},
{30,8448,256},
{30,8256,256},
{30,8128,256},
{30,7360,256},
{30,6528,256},
{30,13568,256},
{30,11968,256},
{30,11648,256},
{30,11264,256},
{30,10944,256},
{30,10112,256},
{2,9792,256},
{2,8064,256},
{2,7936,256},
{2,11264,256},
{2,10816,256},
{2,10432,256},
{2,10432,256},
{2,10240,256},
{29,9600,256},
{29,6464,256},
{29,10432,256},
{28,9792,256},
{28,8128,256},
{28,7360,256},
{28,13568,256},
{28,12288,256},
{28,10816,256},
{28,10240,256},
{27,12480,256},
{27,11840,256},
{27,11712,256},
{26,11264,256},
{26,11264,256},
{23,10816,256},
{22,11968,256},
{20,12864,256},
{1,8320,256},
{1,7936,256},
{1,7936,256},
{1,7616,256},
{1,6400,256},
{1,12480,256},
{1,11840,256},
{1,11520,256},
{1,10944,256},
{1,10688,256},
{1,10240,256},
{17,9728,256},
{17,11840,256},
{15,10432,256},
{14,13568,256},
{12,10752,256},
{11,9280,256},
{11,10112,256},
{10,8320,256},
{10,13568,256},
{10,11520,256},
{9,9792,256},
{9,9600,256},
{9,9472,256},
{9,8896,256},
{9,8768,256},
{9,8640,256},
{9,8128,256},
{9,8064,256},
{9,7616,256},
{9,7360,256},
{9,6464,256},
{9,6400,256},
{9,6272,256},
{9,5824,256},
{9,5632,256},
{9,3328,256},
{9,12864,256},
{9,11840,256},
{9,10944,256},
{9,10944,256},
{9,10752,256},
{9,10688,256},
{9,10432,256},
{9,10368,256},
{9,10048,256},
{8,9984,256},
{8,9920,256},
{8,9792,256},
{8,9728,256},
{8,9280,256},
{8,9216,256},
{8,8768,256},
{8,8704,256},
{8,8448,256},
{8,7936,256},
{8,7104,256},
{8,6528,256},
{8,6400,256},
{8,12288,256},
{8,11840,256},
{8,11648,256},
{8,10944,256},
{8,10432,256},
{8,10368,256},
{8,10048,256},
{7,9984,256},
{7,9216,256},
{7,8640,256},
{7,8320,256},
{7,8128,256},
{7,7168,256},
{7,6400,256},
{7,4800,256},
{7,4096,256},
{7,12480,256},
{7,11968,256},
{7,11840,256},
{7,11712,256},
{7,11520,256},
{7,11264,256},
{6,9984,256},
{6,9920,256},
{6,9792,256},
{6,9280,256},
{6,9216,256},
{6,8768,256},
{6,8448,256},
{6,8320,256},
{6,8128,256},
{6,8064,256},
{6,7360,256},
{6,6528,256},
{6,6400,256},
{6,6400,256},
{6,5632,256},
{6,4800,256},
{6,12864,256},
{6,12480,256},
{6,11712,256},
{6,11648,256},
{6,11520,256},
{6,10944,256},
{6,10816,256},
{6,10688,256},
{6,10432,256},
{6,10368,256},
{6,10240,256},
{6,10048,256},
{5,9600,256},
{5,9472,256},
{5,9280,256},
{5,9280,256},
{5,8896,256},
{5,8768,256},
{5,8704,256},
{5,8320,256},
{5,7936,256},
{5,7936,256},
{5,7104,256},
{5,6400,256},
{5,5824,256},
{5,11968,256},
{5,11520,256},
{5,11264,256},
{5,10944,256},
{5,10944,256},
{5,10816,256},
{5,10688,256},
{5,10432,256},
{5,10368,256},
{5,10112,256},
{5,10048,256},
{4,9920,256},
{4,9728,256},
{4,9472,256},
{4,9280,256},
{4,8896,256},
{4,8704,256},
{4,8256,256},
{4,8128,256},
{4,7616,256},
{4,5632,256},
{4,12864,256},
{4,12288,256},
{4,11840,256},
{4,11712,256},
{4,11264,256},
{4,10688,256},
{3,9920,256},
{3,9792,256},
{3,8896,256},
{3,8768,256},
{3,8448,256},
{3,8320,256},
{3,8320,256},
{3,8256,256},
{3,7936,256},
{3,7168,256},
{3,6848,256},
{3,6528,256},
{3,6464,256},
{3,5632,256},
{3,4800,256},
{3,4096,256},
{3,3328,256},
{3,13568,256},
{3,11968,256},
{3,11840,256},
{3,11264,256},
{3,10752,256},
{3,10432,256},
{3,10048,256},
{31,9984,256},
{31,9728,256},
{31,9472,256},
{31,9280,256},
{31,9216,256},
{31,8640,256},
{31,8448,256},
{31,8320,256},
{31,7936,256},
{31,6464,256},
{31,6400,256},
{31,5824,256},
{31,5632,256},
{31,12864,256},
{31,11840,256},
{31,11520,256},
{31,10944,256},
{31,10688,256},
{31,10048,256},
{30,9984,256},
{30,9728,256},
{30,8640,256},
{30,8320,256},
{30,8064,256},
{30,7936,256},
{30,7936,256},
{30,7104,256},
{30,5632,256},
{30,4800,256},
{30,12864,256},
{30,11840,256},
{30,11840,256},
{30,11520,256},
{30,11264,256},
{30,10944,256},
{30,10752,256},
{30,10688,256},
{30,10432,256},
{30,10368,256},
{30,10240,256},
{30,10048,256},
{2,9984,256},
{2,9920,256},
{2,9728,256},
{2,9280,256},
{2,8896,256},
{2,8704,256},
{2,8320,256},
{2,7360,256},
{2,7104,256},
{2,6848,256},
{2,6464,256},
{2,6400,256},
{2,6272,256},
{2,5632,256},
{2,4800,256},
{2,12864,256},
{2,1280,256},
{2,12288,256},
{2,11968,256},
{2,11840,256},
{2,11712,256},
{2,10688,256},
{2,10112,256},
{2,10048,256},
{29,9984,256},
{29,9792,256},
{29,9792,256},
{29,9728,256},
{29,9472,256},
{29,9280,256},
{29,9216,256},
{29,8448,256},
{29,8320,256},
{29,8128,256},
{29,7104,256},
{29,6848,256},
{29,6400,256},
{29,6400,256},
{29,4096,256},
{29,12864,256},
{29,12480,256},
{29,11968,256},
{29,11840,256},
{29,11648,256},
{29,11520,256},
{29,10944,256},
{29,10944,256},
{29,10816,256},
{29,10752,256},
{29,10432,256},
{29,10368,256},
{29,10240,256},
{29,10048,256},
{28,9600,256},
{28,9280,256},
{28,9280,256},
{28,8768,256},
{28,8704,256},
{28,8640,256},
{28,8448,256},
{28,8320,256},
{28,8256,256},
{28,7936,256},
{28,7936,256},
{28,7616,256},
{28,7104,256},
{28,6464,256},
{28,5824,256},
{28,5632,256},
{28,11712,256},
{28,10944,256},
{28,10752,256},
{28,10688,256},
{28,10112,256},
{28,10048,256},
{27,9792,256},
{27,9728,256},
{27,9600,256},
{27,9216,256},
{27,8768,256},
{27,8448,256},
{27,6400,256},
{27,12288,256},
{27,11264,256},
{27,10944,256},
{27,10752,256},
{27,10432,256},
{27,10368,256},
{26,9920,256},
{26,9280,256},
{26,8896,256},
{26,8704,256},
{26,8320,256},
{26,8128,256},
{26,8064,256},
{26,6528,256},
{26,6400,256},
{26,6272,256},
{26,5632,256},
{26,3328,256},
{26,13568,256},
{26,12864,256},
{26,12480,256},
{26,12288,256},
{26,11840,256},
{26,11840,256},
{26,11648,256},
{26,10944,256},
{26,10816,256},
{26,10688,256},
{26,10432,256},
{26,10112,256},
{25,9728,256},
{25,9472,256},
{25,9280,256},
{25,9280,256},
{25,8896,256},
{25,8768,256},
{25,8320,256},
{25,8128,256},
{25,7616,256},
{25,7168,256},
{25,7104,256},
{25,6464,256},
{25,5632,256},
{25,5632,256},
{25,11712,256},
{25,11520,256},
{25,11264,256},
{25,10816,256},
{25,10752,256},
{25,10368,256},
{25,10048,256},
{24,9920,256},
{24,9216,256},
{24,8640,256},
{24,8320,256},
{24,8128,256},
{24,7936,256},
{24,7360,256},
{24,6528,256},
{24,4800,256},
{24,4096,256},
{24,11968,256},
{24,11840,256},
{24,11520,256},
{24,11264,256},
{24,10944,256},
{24,10944,256},
{24,10688,256},
{24,10240,256},
{24,10112,256},
{23,9984,256},
{23,9792,256},
{23,9792,256},
{23,8448,256},
{23,8320,256},
{23,8256,256},
{23,8128,256},
{23,7936,256},
{23,7168,256},
{23,6400,256},
{23,5824,256},
{23,13568,256},
{23,12864,256},
{23,11968,256},
{23,11840,256},
{23,11264,256},
{23,10752,256},
{23,10432,256},
{23,10368,256},
{22,9280,256},
{22,8320,256},
{22,8064,256},
{22,6272,256},
{22,5632,256},
{22,12864,256},
{22,12480,256},
{22,11712,256},
{22,11648,256},
{22,11264,256},
{22,10944,256},
{22,10240,256},
{22,10048,256},
{21,9920,256},
{21,9792,256},
{21,9728,256},
{21,9600,256},
{21,9472,256},
{21,8896,256},
{21,8768,256},
{21,8704,256},
{21,8128,256},
{21,6400,256},
{21,12480,256},
{21,12288,256},
{21,11840,256},
{21,11712,256},
{21,10944,256},
{21,10816,256},
{21,10752,256},
{21,10688,256},
{20,9984,256},
{20,9216,256},
{20,8768,256},
{20,8448,256},
{20,8128,256},
{20,7104,256},
{20,6848,256},
{20,5824,256},
{20,13568,256},
{20,12480,256},
{20,11840,256},
{20,11520,256},
{20,11264,256},
{20,10816,256},
{20,10112,256},
{1,9920,256},
{1,9792,256},
{1,9728,256},
{1,9472,256},
{1,9280,256},
{1,8704,256},
{1,8128,256},
{1,7168,256},
{1,7104,256},
{1,6848,256},
{1,6528,256},
{1,6464,256},
{1,5824,256},
{1,5632,256},
{1,4800,256},
{1,4096,256},
{1,12288,256},
{1,10432,256},
{1,10368,256},
{19,9920,256},
{19,9280,256},
{19,9280,256},
{19,8768,256},
{19,8640,256},
{19,8320,256},
{19,8064,256},
{19,7360,256},
{19,3328,256},
{19,13568,256},
{19,12864,256},
{19,11840,256},
{19,11712,256},
{19,11648,256},
{19,11264,256},
{19,10944,256},
{19,10944,256},
{19,10752,256},
{19,10432,256},
{19,10240,256},
{19,10048,256},
{18,9984,256},
{18,9792,256},
{18,9792,256},
{18,9472,256},
{18,9280,256},
{18,8768,256},
{18,8256,256},
{18,8128,256},
{18,7936,256},
{18,6272,256},
{18,4800,256},
{18,11840,256},
{18,11840,256},
{18,11648,256},
{18,11264,256},
{17,9984,256},
{17,9600,256},
{17,9216,256},
{17,8704,256},
{17,8320,256},
{17,8128,256},
{17,8064,256},
{17,7936,256},
{17,6400,256},
{17,5632,256},
{17,13568,256},
{17,12480,256},
{17,11968,256},
{17,11712,256},
{17,11520,256},
{17,10944,256},
{17,10432,256},
{17,10368,256},
{16,9792,256},
{16,7616,256},
{16,7168,256},
{16,6848,256},
{16,6400,256},
{16,6272,256},
{16,5632,256},
{16,4096,256},
{16,12864,256},
{16,12288,256},
{16,11840,256},
{16,11264,256},
{16,10816,256},
{16,10688,256},
{16,10432,256},
{16,10112,256},
{15,9728,256},
{15,9216,256},
{15,8896,256},
{15,7936,256},
{15,7360,256},
{15,7168,256},
{15,7104,256},
{15,6400,256},
{15,5632,256},
{15,11840,256},
{15,10240,256},
{15,10048,256},
{14,9920,256},
{14,9600,256},
{14,9472,256},
{14,9280,256},
{14,8448,256},
{14,8320,256},
{14,8064,256},
{14,7936,256},
{14,7616,256},
{14,6848,256},
{14,6528,256},
{14,6464,256},
{14,12864,256},
{14,12288,256},
{14,11264,256},
{14,10944,256},
{14,10752,256},
{14,10368,256},
{14,10112,256},
{13,9792,256},
{13,9216,256},
{13,8768,256},
{13,8704,256},
{13,8320,256},
{13,6528,256},
{13,13568,256},
{13,12480,256},
{13,12288,256},
{13,11648,256},
{13,11520,256},
{13,10752,256},
{13,10688,256},
{12,9920,256},
{12,9792,256},
{12,9728,256},
{12,9280,256},
{12,8896,256},
{12,8448,256},
{12,8128,256},
{12,7936,256},
{12,7360,256},
{12,7104,256},
{12,6464,256},
{12,5632,256},
{12,4800,256},
{12,4096,256},
{12,12864,256},
{12,1280,256},
{12,11968,256},
{12,11840,256},
{12,11840,256},
{12,11648,256},
{12,10944,256},
{12,10816,256},
{12,10432,256},
{12,10368,256},
{12,10240,256},
{11,9984,256},
{11,9472,256},
{11,9216,256},
{11,8640,256},
{11,8256,256},
{11,8128,256},
{11,7936,256},
{11,7616,256},
{11,6400,256},
{11,13568,256},
{11,12864,256},
{11,1280,256},
{11,12480,256},
{11,11968,256},
{11,11840,256},
{11,11712,256},
{11,11264,256},
{11,10944,256},
{11,10688,256},
{10,9920,256},
{10,9792,256},
{10,9792,256},
{10,9728,256},
{10,8640,256},
{10,8320,256},
{10,7168,256},
{10,6464,256},
{10,5824,256},
{10,12864,256},
{10,12480,256},
{10,12288,256},
{10,11968,256},
{10,11712,256},
{10,11264,256},
{10,11264,256},
{10,10816,256},
{10,10240,256}};


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
    benchmarkLoop<ssse3>(iterations, matrices, align, use_fbgemm, use_eigen);
  } else if (myarch==avx2) {
    benchmarkLoop<avx2>(iterations, matrices, align, use_fbgemm, use_eigen);
  } else if (myarch==avx512) {
    benchmarkLoop<avx512>(iterations, matrices, align, use_fbgemm, use_eigen);
  } else if (myarch==avx512vnni) {
    benchmarkLoop<avx512vnni>(iterations, matrices, align, use_fbgemm, use_eigen);
  } else if (myarch==any) {
    benchmarkLoop<any>(iterations, matrices, align, use_fbgemm, use_eigen);
  }

  return 0;
}
