#include <array>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <dnnl.h>
#include <chrono>
#include "intgemm/intgemm.h"
#include "aligned.h"


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


int main(int argc, char const *argv[]) {

  //@TODO limit ISA: dnnl_status_t DNNL_API dnnl_set_max_cpu_isa(dnnl_cpu_isa_t isa);

  std::chrono::duration<double> eigen_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mkl_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklU_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> mklS_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kenn_duration_loop = std::chrono::duration<double>::zero();
  std::chrono::duration<double> kennU_duration_loop = std::chrono::duration<double>::zero();
  const size_t align = 256;

  int iterations = 1000;
  bool use_eigen = false;
  if (argc == 1) {
    iterations = 1000;
    use_eigen = false;
  } else if (argc == 2) {
    iterations = std::atoi(argv[1]);
  } else if (argc == 3) {
    iterations = std::atoi(argv[1]);
    use_eigen = std::atoi(argv[2]);
  } else {
    std::cerr << "Usage: " << argv[0] << " [iterations=1000] [use_eigen=0]" << std::endl;
    std::exit(1);
  }

  struct matrix_size zero = {1024, 1024, 1024};
  struct matrix_size one = {256, 10368, 256};
  struct matrix_size two = {256, 5312, 256};
  struct matrix_size three = {8, 2048, 256};
  struct matrix_size four = {320, 256, 256};
  struct matrix_size five = {472, 256, 256};
  struct matrix_size six = {248, 256, 256};
  struct matrix_size seven = {200, 256, 256};
  struct matrix_size eight = {1, 64, 8};

  std::array<matrix_size, 9> matrices = {zero, one, two, three, four, five, six, seven, eight};

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
        alloc::AlignedVector<int32_t, align> A_EIGEN(M*K);
        alloc::AlignedVector<int32_t, align> B_EIGEN(K*N);
        alloc::AlignedVector<int32_t, align> C_EIGEN(M*N);

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
      alloc::AlignedVector<int8_t, align> A_MKL(M*K);
      alloc::AlignedVector<int8_t, align> B_MKL(K*N);
      alloc::AlignedVector<int32_t, align> C_MKL(M*N);


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

      alloc::AlignedVector<float, align> A_proto(M * K);
      alloc::AlignedVector<float, align> B_proto(K * N);

      std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_proto.get());
      std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_proto.get());


      float quant_mult = 127.0 / 2.0;
      alloc::AlignedVector<int8_t, align> A_prepared(M * K);
      alloc::AlignedVector<int8_t, align> B_prepared(K * N);

      intgemm::Int8::PrepareA(A_proto.get(), A_prepared.get(), quant_mult, M, K);
      // Quantize and reshape B.
      // Typically you will do this once when parameters are loaded, not every time.
      intgemm::Int8::PrepareB(B_proto.get(), B_prepared.get(), quant_mult, K, N);

      alloc::AlignedVector<float, align> C_kenn(M*N);

      auto kenn_start = std::chrono::system_clock::now();
      intgemm::Int8::Multiply(A_prepared.get(), B_prepared.get(), M, K, N, intgemm::callbacks::UnquantizeAndWrite(1.0 / (quant_mult * quant_mult), C_kenn.get()));
      auto kenn_end = std::chrono::system_clock::now();

      kenn_duration_loop += (kenn_end - kenn_start);
          
      //MKL-DNN SignedXunsigned
      // Copy onto aligned memory
      alloc::AlignedVector<uint8_t, align> A1_MKL(M*K);
      alloc::AlignedVector<int8_t, align> B1_MKL(K*N);
      alloc::AlignedVector<int32_t, align> C1_MKL(M*N);


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
      alloc::AlignedVector<float, align> A_proto1(M * K);
      alloc::AlignedVector<float, align> B_proto1(K * N);
      alloc::AlignedVector<float, align> inputBias(K);
      std::fill(inputBias.get(), inputBias.get() + K, 0.0f);

      std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_proto1.get());
      std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_proto1.get());


      //float quant_mult = 127.0 / 2.0;
      alloc::AlignedVector<int8_t, align> A_prepared1(M * K); //@TODO API CHANGE
      alloc::AlignedVector<int8_t, align> B_prepared1(K * N);

      intgemm::Int8Shift::PrepareA(A_proto1.get(), A_prepared1.get(), quant_mult, M, K);
      // Quantize and reshape B.
      // Typically you will do this once when parameters are loaded, not every time.
      intgemm::Int8Shift::PrepareB(B_proto1.get(), B_prepared1.get(), quant_mult, K, N);

      float unquant_mult_forprep = (-1)*(2.0)*(2.0)/(127.0f);

      //PrepareBias
      intgemm::Int8Shift::PrepareBias(B_prepared1.get(), K, N, intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, inputBias.get(), inputBias.get()));

      alloc::AlignedVector<float, align> C_kenn1(M*N);

      auto kennU_start = std::chrono::system_clock::now();
      intgemm::Int8Shift::Multiply(A_prepared1.get(), B_prepared1.get(), M, K, N, intgemm::callbacks::UnquantizeAndAddBiasAndWrite(1.0 / (quant_mult * quant_mult), inputBias.get(), C_kenn1.get()));
      auto kennU_end = std::chrono::system_clock::now();

      kennU_duration_loop += (kennU_end - kennU_start);

      //MKLDNN Single precision
      alloc::AlignedVector<float, align> A_MKL_S(M*K);
      alloc::AlignedVector<float, align> B_MKL_S(K*N);
      alloc::AlignedVector<float, align> C_MKL_S(M*N);

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
    std::cout << sizes << " in loop, for " << iterations << " interations:" << std::endl;
    if (use_eigen)
      std::cout <<"          Eigen took: " << eigen_duration_loop.count() << " seconds." << std::endl;

    std::cout << "            MKL took: " << mkl_duration_loop.count() << " seconds." << std::endl <<
                 "   MKL unsigned took: " << mklU_duration_loop.count() << " seconds." << std::endl <<
                 "      MKL float took: " << mklS_duration_loop.count() << " seconds." << std::endl <<
                 "        Intgemm took: " << kenn_duration_loop.count() << " seconds." << std::endl <<
                 "Intgemm Shifted took: " << kennU_duration_loop.count() << " seconds." << std::endl <<
                      "Alignment was: " << align << "." << std::endl;

  }

    return 0;
}
