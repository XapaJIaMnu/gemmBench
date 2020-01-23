#include <array>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mkldnn.h>
#include <chrono>
#include "intgemm/intgemm.h"
#include "aligned.h"


//FBGEMM

//
#include "fbgemm/FbgemmSpMM.h"

//OPENCL


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the CLBlast library (C interface)
#include <clblast_c.h>
/*
template <typename data_t>
void ref_gemm(const char *transa, const char *transb, int m, int n, int k,
        const data_t alpha, const data_t *a, int lda, const data_t *b,
        int ldb, data_t beta, data_t *c, int ldc) {

    const bool tr_a = transa && (*transa == 'T' || *transa == 't');
    const bool tr_b = transb && (*transb == 'T' || *transb == 't');

    auto pa = [=] (int i, int j) { return a[j*lda + i]; };
    auto pb = [=] (int i, int j) { return b[j*ldb + i]; };
    auto pc = [=] (int i, int j) { return c[j*ldc + i]; };

    mkldnn::impl::parallel_nd(m, n, [&](int im, int in) {
        data_t c_elem = (beta == 0.) ? 0. : pc(im, in) * beta;

        for (int ik = 0; ik < k; ik++) {
            const data_t a_elem = tr_a ? pa(ik, im) : pa(im, ik);
            const data_t b_elem = tr_b ? pb(in, ik) : pb(ik, in);
            c_elem += alpha * a_elem * b_elem;
        }
        c[in*ldc + im] = c_elem;
    });
}
*/

mkldnn_status_t check_gemm_input(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const int *lda,
        const int *ldb, const int *ldc, const float *alpha, const float *beta,
        const bool with_bias) {
    //if (utils::any_null(transa, transb, M, N, K, lda, ldb, ldc, alpha, beta))
    //    return mkldnn_invalid_arguments;
    if (with_bias && *beta != 0)
        return mkldnn_unimplemented;
    bool consistency = true
        //&& utils::one_of(*transa, 'T', 't', 'N', 'n')
        //&& utils::one_of(*transb, 'T', 't', 'N', 'n')
        && *M >= 0
        && *N >= 0
        && *K >= 0;

    if (!consistency)
        return mkldnn_invalid_arguments;
    bool isTransA = false;
    bool isTransB = false;
    if (*transa == 'T' || *transa == 't') {
     isTransA = true;
    }

    if (*transb == 'T' || *transb == 't') {
     isTransB = true;
    }

    int nrowA = isTransA ? *K : *M;
    int nrowB = isTransB ? *N : *K;
    consistency = true
        && *lda >= std::max(1, nrowA)
        && *ldb >= std::max(1, nrowB)
        && *ldc >= std::max(1, *M);
    if (!consistency)
        return mkldnn_invalid_arguments;

    return mkldnn_success;
}

mkldnn_status_t check_gemm_x8x8x32_input(const char *offsetc,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const int *ldc,
        const float *alpha, const float *beta, const bool with_bias) {
    if (offsetc == nullptr)
        return mkldnn_invalid_arguments;
    //if (!utils::one_of(*offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
    //    return mkldnn_invalid_arguments;

    return check_gemm_input(transa, transb, M, N, K, lda, ldb, ldc, alpha,
        beta, with_bias);
}


void printMKLdnnStatus(mkldnn_status_t& status) {
    if (status == mkldnn_success) {
        std::cout << "MKL success." << std::endl;
    } else if (status == mkldnn_out_of_memory) {
        std::cout << "The operation failed due to an out-of-memory condition." << std::endl;
    } else if (status == mkldnn_try_again) {
        std::cout << "The operation failed and should be retried." << std::endl;
    } else if (status == mkldnn_invalid_arguments ) {
        std::cout << "The operation failed because of incorrect function arguments." << std::endl;
    } else if (status == mkldnn_not_ready) {
        std::cout << "The operation failed because a primitive was not ready for execution." << std::endl;
    } else if (status == mkldnn_unimplemented) {
        std::cout << "The operation failed because requested functionality is not implemented." << std::endl;
    } else if (status == mkldnn_iterator_ends) {
        std::cout << "Primitive iterator passed over last primitive descriptor." << std::endl;
    } else if (status == mkldnn_runtime_error) {
        std::cout << "Primitive or engine failed on execution." << std::endl;
    } else if (status == mkldnn_not_required) {
        std::cout << "Queried element is not required for given primitive." << std::endl;
    }
}

int openCLGemm(size_t m, size_t n, size_t k) {
	// OpenCL platform/device settings
	const size_t platform_id = 0;
	const size_t device_id = 0;

	// Example SGEMM arguments
	//const size_t m = 128;
	//const size_t n = 64;
	//const size_t k = 512;
	const float alpha = 0.7f;
	const float beta = 1.0f;
	const size_t a_ld = k;
	const size_t b_ld = n;
	const size_t c_ld = n;

	// Initializes the OpenCL platform
	static cl_uint num_platforms;
	static cl_platform_id platform;
	static cl_device_id* devices;
	static cl_device_id device;
	static cl_uint num_devices;
	static cl_context context;
	static cl_command_queue queue;
	static cl_event event;
	static bool first = true;
	if (first) {
		clGetPlatformIDs(0, NULL, &num_platforms);
		cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
		clGetPlatformIDs(num_platforms, platforms, NULL);
		platform = platforms[platform_id];

		// Initializes the OpenCL device
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		device = devices[device_id];

		// Creates the OpenCL context, queue, and an event
		context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
		queue = clCreateCommandQueue(context, device, 0, NULL); //OpenCL 1.2 support
		event = NULL;
		first = false;
	}

	// Populate host matrices with some example data
	float* host_a = (float*)malloc(sizeof(float)*m*k);
	float* host_b = (float*)malloc(sizeof(float)*n*k);
	float* host_c = (float*)malloc(sizeof(float)*m*n);
	for (size_t i=0; i<m*k; ++i) { host_a[i] = 12.193f; }
	for (size_t i=0; i<n*k; ++i) { host_b[i] = -8.199f; }
	for (size_t i=0; i<m*n; ++i) { host_c[i] = 0.0f; }

	// Copy the matrices to the device
	cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, m*k*sizeof(float), NULL, NULL);
	cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*k*sizeof(float), NULL, NULL);
	cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m*k*sizeof(float), host_a, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n*k*sizeof(float), host_b, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, m*n*sizeof(float), host_c, 0, NULL, NULL);

	// Call the SGEMM routine.
	CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
	                                      CLBlastTransposeNo, CLBlastTransposeNo,
	                                      m, n, k,
	                                      alpha,
	                                      device_a, 0, a_ld,
	                                      device_b, 0, b_ld,
	                                      beta,
	                                      device_c, 0, c_ld,
	                                      &queue, &event);

	// Wait for completion
	if (status == CLBlastSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}

	// Clean-up
	//free(platforms);
	//free(devices);
	free(host_a);
	free(host_b);
	free(host_c);
	clReleaseMemObject(device_a);
	clReleaseMemObject(device_b);
	clReleaseMemObject(device_c);
	//clReleaseCommandQueue(queue);
	//clReleaseContext(context);

	return status;
}

int main(int argc, char const *argv[]) {
    /* old sanity checks
    alignas(64) const std::array<int8_t, 12> a = {1,2,3,4,
                                5,6,7,8,
                                9,10,11,12};

    alignas(64) const std::array<int8_t, 20> b = {13,14,15,16,17,
                                18,19,20,21,22,
                                23,24,25,26,27,
                                28,29,30,31,32};

    alignas(64) std::array<int32_t, 15> c = {-1,-2,-3,-4,-5,
                                -6,-7,-8,-9,-10,
                                -11,-12,-13,-14,-15};

    Eigen::Matrix<int32_t, 3,4> a1_eigen = Eigen::Map<const Eigen::Matrix<int8_t, 3, 4, Eigen::RowMajor>>(a.data()).cast<int32_t>();
    Eigen::Matrix<int32_t, 4,5> b1_eigen = Eigen::Map<const Eigen::Matrix<int8_t, 4, 5, Eigen::RowMajor>>(b.data()).cast<int32_t>();
    Eigen::Matrix<int32_t, 3,5> c1_eigen = Eigen::Map<Eigen::Matrix<int32_t, 3, 5, Eigen::RowMajor>>(c.data());

    Eigen::Matrix<int32_t, 3,4> a_eigen;
    a_eigen << 1,2,3,4,
               5,6,7,8,
               9,10,11,12;
    Eigen::Matrix<int32_t, 4,5> b_eigen;
    b_eigen << 13,14,15,16,17,
               18,19,20,21,22,
               23,24,25,26,27,
               28,29,30,31,32;
    Eigen::Matrix<int32_t, 3,5> c_eigen;
    c_eigen << -1,-2,-3,-4,-5,
               -6,-7,-8,-9,-10,
               -11,-12,-13,-14,-15;


    c1_eigen.noalias() += a1_eigen*b1_eigen;

    std::cout  << c1_eigen << std::endl;

    c_eigen.noalias() += a_eigen*b_eigen;

    std::cout << c_eigen << std::endl;

    //MKLGEMM
	{
		//New Matrix:
		//New Matrix:
		char offsetc = 'F';
		bool zero_oa = 1;
		bool zero_ob = 1;
		bool zero_oc = 0;
		char transA = 'N';
		char transB = 'n';
		const int M = 30;
		const int N = 20;
		const int K = 10;
		float alpha = 2;
		float beta = 1;
		int lda = M;
		int ldb = K;
		int ldc = M;
		bool expect_to_fail = 0;

		//Those are likely COL_MAJOR
		alignas(64) std::array<int8_t, M*K> A = {-10,-2,-10,8,3,-4,-6,6,10,3,
		                                        2,-8,6,0,1,7,-5,5,-2,-10,
		                                        6,6,9,1,0,6,-6,7,6,10,
		                                        9,6,-5,2,8,-3,3,-3,1,2,
		                                        4,2,-7,6,6,-10,-6,7,10,-1,
		                                        -10,-5,-3,-7,-8,-8,10,-8,-1,8,
		                                        3,-4,-5,-9,10,-8,-5,1,5,-5,
		                                        -6,0,-5,-2,-7,6,0,4,-5,6,
		                                        -7,-2,9,8,7,7,3,-1,-8,6,
		                                        -6,-8,-2,4,-8,-8,-7,1,7,8,
		                                        8,-6,2,-8,-4,9,8,7,8,6,
		                                        3,3,7,3,-7,-9,-8,3,10,-10,
		                                        -6,1,8,7,2,3,-6,9,3,3,
		                                        4,0,-6,0,-9,10,-6,10,-3,-7,
		                                        -5,4,6,7,9,-7,-3,-2,-3,6,
		                                        -8,-4,2,5,-9,8,-10,5,-7,-10,
		                                        0,7,8,-9,-5,8,1,-5,-6,3,
		                                        6,-4,7,4,-4,5,-1,0,4,3,
		                                        3,4,4,-4,2,-8,5,-4,2,-10,
		                                        7,-7,-8,2,-4,9,9,-2,1,-2,
		                                        -10,-5,10,2,5,1,1,-7,9,8,
		                                        5,-1,5,-8,1,2,7,3,2,5,
		                                        -8,-4,-7,-7,6,2,7,1,-9,6,
		                                        1,-7,-7,-6,-10,-8,-10,2,-9,2,
		                                        5,7,0,-3,10,-1,8,6,7,10,
		                                        10,-8,4,5,3,9,7,-5,2,-7,
		                                        -8,-5,-8,2,9,7,-8,-5,6,1,
		                                        5,1,-4,-6,3,3,-10,-7,9,-7,
		                                        9,-1,6,5,-2,-9,7,2,0,5,
		                                        7,3,8,-1,-2,0,2,9,-7,-7};
		                                        

		alignas(64) std::array<int8_t, N*K> B = {-2,6,-7,10,9,5,4,7,-8,10,
		                                        -5,-4,8,-9,0,2,-4,7,2,1,
		                                        7,-9,8,-10,-3,-2,7,7,-6,-5,
		                                        -2,10,3,-2,-9,10,-2,2,4,-7,
		                                        -6,4,9,-7,-7,8,7,-2,-3,-5,
		                                        -8,-8,7,-5,10,2,2,2,-8,10,
		                                        0,9,-5,-1,-4,-6,9,-9,-10,-9,
		                                        -10,-2,10,-3,-8,4,-7,4,-1,8,
		                                        -6,9,4,3,0,2,2,5,4,-9,
		                                        5,9,3,-3,9,4,6,2,3,5,
		                                        9,10,-8,4,-8,-2,7,1,-2,-5,
		                                        3,-9,9,-7,-8,4,8,-10,-7,7,
		                                        -2,7,-6,-3,6,-8,-10,5,-1,1,
		                                        -1,-9,-1,8,8,-10,1,-3,5,-10,
		                                        -10,6,10,-3,0,4,-4,5,4,9,
		                                        -3,1,-8,7,4,-3,5,1,6,-2,
		                                        3,6,-5,-4,-10,-6,-7,-3,4,-6,
		                                        8,8,-5,-2,3,0,3,7,-7,-8,
		                                        -10,4,6,8,3,8,-6,3,10,-2,
		                                        9,-6,7,5,10,5,-5,-1,-8,1};
		                                        

		alignas(64) std::array<int32_t, M*N> C = {-7,1,-8,-8,-5,0,3,-7,-2,-7,7,-5,8,5,2,-3,6,8,9,-6,
		                                        8,7,-3,-4,-8,8,10,10,6,-7,8,0,0,4,-10,-3,-2,-6,9,4,
		                                        -4,8,-8,4,-8,2,-2,3,-4,3,-4,10,-7,5,-7,-4,0,-8,-7,-4,
		                                        -5,9,4,4,2,5,-10,9,9,8,2,-6,5,2,7,-7,0,-7,7,-6,
		                                        1,0,3,4,-6,4,-3,8,-9,-5,6,8,5,5,-9,-3,-10,5,-1,5,
		                                        -2,6,5,-8,5,-9,3,-1,0,9,-3,-1,5,5,8,-7,-1,-2,-5,0,
		                                        1,6,1,-1,-2,6,-2,2,-6,-9,-10,8,8,-1,0,-5,-4,-6,6,6,
		                                        0,6,4,6,-8,8,10,-9,-1,4,-9,7,-6,9,-6,-3,8,-7,-1,3,
		                                        -3,-7,-6,2,-5,5,10,-1,4,-8,3,-2,-2,6,-1,-3,4,6,-2,8,
		                                        -5,9,4,2,-9,-4,8,1,-10,1,-2,-8,-4,6,8,0,2,2,-9,3,
		                                        -7,1,8,-3,2,9,9,-8,-4,6,-1,3,-9,-7,-6,-3,0,0,-2,-10,
		                                        7,4,-9,-6,3,-3,-10,-5,-4,7,9,6,-6,7,-9,3,-10,4,-1,10,
		                                        2,-10,4,4,9,0,-8,-7,3,-7,2,-7,9,5,-4,-6,-8,-8,5,6,
		                                        3,10,8,5,1,-6,-9,-1,8,-7,6,10,2,6,-8,-6,-5,10,8,0,
		                                        9,6,-7,-6,1,-5,5,-9,4,-10,1,2,8,5,4,-1,4,-7,6,-7,
		                                        -2,-10,5,-8,7,-10,7,-1,10,-9,-8,10,-2,5,-10,-3,4,-6,6,-1,
		                                        -2,1,1,-7,-6,0,4,3,3,-8,-2,-7,7,10,-4,6,-10,2,6,-9,
		                                        -9,6,5,5,-7,-2,6,10,-9,-3,-10,6,-6,-4,1,-5,8,-6,-6,8,
		                                        10,9,2,-10,10,8,1,-3,10,6,-10,8,-5,3,-3,-4,-7,-8,-7,-8,
		                                        -3,-7,6,2,-2,-4,7,-5,0,-10,-8,-3,10,2,-2,7,0,9,-9,-3,
		                                        2,6,3,7,3,7,-4,8,-8,-5,-3,-2,-4,0,-5,9,-3,2,-10,-4,
		                                        -6,1,-6,6,-3,0,3,1,9,10,7,-8,-3,10,-9,4,0,1,-10,-5,
		                                        1,-7,6,-7,-1,3,-1,3,10,3,-7,-9,-4,1,7,7,-8,-7,3,-2,
		                                        8,-2,0,8,-5,3,1,-8,9,-8,7,-8,-1,-4,-3,-5,-5,-8,-2,5,
		                                        4,-8,-1,5,-2,9,3,2,6,8,-1,-4,-6,8,3,-5,-10,1,1,-1,
		                                        8,0,10,1,-5,6,-3,5,-5,10,7,0,9,-10,2,-1,0,-5,-7,-1,
		                                        -3,-1,0,-6,-2,0,8,3,9,5,0,-3,-4,0,9,-6,1,-9,6,2,
		                                        10,2,-1,10,-7,1,1,-3,4,3,7,-4,7,6,9,4,2,-5,-4,-1,
		                                        6,1,10,-6,7,0,-4,6,10,-9,-9,0,-9,-3,-2,2,-8,-5,4,-9,
		                                        3,8,7,-8,3,-5,3,-5,10,-5,-8,-10,2,-3,7,-9,1,-6,6,-10};

		//Eigen sanity check matrices
		alignas(64) Eigen::Matrix<int32_t, M,K> A_eigen = Eigen::Map<const Eigen::Matrix<int8_t, M, K, Eigen::ColMajor>>(A.data()).cast<int32_t>();
		alignas(64) Eigen::Matrix<int32_t, K,N> B_eigen = Eigen::Map<const Eigen::Matrix<int8_t, K, N, Eigen::ColMajor>>(B.data()).cast<int32_t>();
		alignas(64) Eigen::Matrix<int32_t, M,N> C_eigen = Eigen::Map<Eigen::Matrix<int32_t, M, N, Eigen::ColMajor>>(C.data());

		//Kenneth's matrices
		Eigen::Matrix<int8_t, M,K, Eigen::RowMajor> A_kenn = Eigen::Map<const Eigen::Matrix<int8_t, M, K, Eigen::ColMajor>>(A.data());
		Eigen::Matrix<int8_t, K,N, Eigen::RowMajor> B_kenn = Eigen::Map<const Eigen::Matrix<int8_t, K, N, Eigen::ColMajor>>(B.data());
		Eigen::Matrix<int32_t, M,N, Eigen::RowMajor> C_kenn = Eigen::Map<const Eigen::Matrix<int32_t, M, N, Eigen::ColMajor>>(C.data());

		auto eigen_start = std::chrono::system_clock::now();
		C_eigen.noalias() += (A_eigen*(int)alpha)*(B_eigen*(int)beta);
		auto eingen_end = std::chrono::system_clock::now();
		std::cout << "EIGEN" << std::endl;
		std::cout << C_eigen << std::endl << std::endl << "MKL" << std::endl;
		                                        
		int8_t oa = 0;
		int8_t ob = 0;
		std::array<int32_t, 1> oc = {0};

		auto status_args = check_gemm_x8x8x32_input(&offsetc,
		        &transA, &transB, &M, &N, &K, &lda, &ldb, &ldc,
		        &alpha, &beta, false);
		printMKLdnnStatus(status_args);
		auto mkl_start = std::chrono::system_clock::now();
		auto status = mkldnn_gemm_s8s8s32(&transA, &transB, &offsetc,
		        &M, &N, &K, &alpha, A.data(), &lda, &oa, B.data(), &ldb, &ob,
		        &beta, C.data(), &ldc, oc.data());
		auto mkl_end = std::chrono::system_clock::now();
		printMKLdnnStatus(status);

		Eigen::Matrix<int32_t, M,N> C_MKL = Eigen::Map<Eigen::Matrix<int32_t, M, N, Eigen::ColMajor>>(C.data());
		std::cout << C_MKL << std::endl;
		std::chrono::duration<double> eigen_duration = eingen_end-eigen_start;
		std::chrono::duration<double> mkl_duration = mkl_end-mkl_start;

		std::cout << std::fixed;
		std::cout.precision(10);
		std::cout << "Eigen took: " << eigen_duration.count()
		              << " seconds. MKL took: " << mkl_duration.count() << " seconds." << std::endl;

		//intgemm::Int8::PrepareA(A_kenn.data(), A_prepared.data(), quant_mult, M, K);
		//intgemm::Int8::PrepareB(B_kenn.data(), B_prepared.data(), quant_mult, K, N);

		//hmmm
	} //Scope */

	std::chrono::duration<double> eigen_duration_loop = std::chrono::duration<double>::zero();
	std::chrono::duration<double> mkl_duration_loop = std::chrono::duration<double>::zero();
	std::chrono::duration<double> kenn_duration_loop = std::chrono::duration<double>::zero();
	std::chrono::duration<double> cblast_duration_loop = std::chrono::duration<double>::zero();
	const size_t align = 256;

	for (int i = 0; i<1000; i++) {

		char offsetc = 'F';
		bool zero_oa = 1;
		bool zero_ob = 1;
		bool zero_oc = 0;
		char transA = 'N';
		char transB = 'n';
		const int M = 320;
		const int N = 80;
		const int K = 320;
		float alpha = 2;
		float beta = 1;
		int lda = M;
		int ldb = K;
		int ldc = M;
		int8_t oa = 0;
		int8_t ob = 0;
		std::array<int32_t, 1> oc = {0};

		//Construct matrices

		Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic> A = Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic>::Random(M,K);
		Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic> B = Eigen::Matrix<int8_t, Eigen::Dynamic,Eigen::Dynamic>::Random(K,N);
		Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic> C = Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic>::Random(M,N);

		Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic> eigen_A_tmp = A.cast<int32_t>();
		Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic> eigen_B_tmp = B.cast<int32_t>();
		
		// Copy onto aligned memory
		alloc::AlignedVector<int8_t, align> A_MKL(M*K);
		alloc::AlignedVector<int8_t, align> B_MKL(K*N);
		alloc::AlignedVector<int32_t, align> C_MKL(M*N);

		alloc::AlignedVector<int32_t, align> A_EIGEN(M*K);
		alloc::AlignedVector<int32_t, align> B_EIGEN(K*N);
		alloc::AlignedVector<int32_t, align> C_EIGEN(M*N);

		//MKL
		std::copy(A.data(), A.data() + A.size(), A_MKL.get());
		std::copy(B.data(), B.data() + B.size(), B_MKL.get());
		std::copy(C.data(), C.data() + C.size(), C_MKL.get());

		//EIGEN
		std::copy(eigen_A_tmp.data(), eigen_A_tmp.data() + eigen_A_tmp.size(), A_EIGEN.get());
		std::copy(eigen_B_tmp.data(), eigen_B_tmp.data() + eigen_B_tmp.size(), B_EIGEN.get());
		std::copy(C.data(), C.data() + C.size(), C_EIGEN.get());

		//Eigen bug: https://stackoverflow.com/questions/54738495/eigenmapd-matrix-from-raw-buffer-gives-object-allocated-on-stack-is-too-big/
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_a(A_EIGEN.get(), M, K);
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_b(B_EIGEN.get(), K, N);
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_c(C_EIGEN.get(), M, N);

		//Sanity check
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > mkl_c_check(C_MKL.get(), M, N);



		//auto eigen_start = std::chrono::system_clock::now();
		//eigen_c.noalias() += (eigen_a*(int)alpha)*(eigen_b*(int)beta);
		//auto eingen_end = std::chrono::system_clock::now();
		//eigen_duration_loop += (eingen_end - eigen_start);

		auto mkl_start = std::chrono::system_clock::now();
		auto status = mkldnn_gemm_s8s8s32(&transA, &transB, &offsetc,
	        &M, &N, &K, &alpha, A_MKL.get(), &lda, &oa, B_MKL.get(), &ldb, &ob,
	        &beta, C_MKL.get(), &ldc, oc.data());
		auto mkl_end = std::chrono::system_clock::now();

		mkl_duration_loop += (mkl_end - mkl_start);
		if (status != mkldnn_success) {
			std::cout << "we died at " << i << std::endl;
			break;
		}

		//if (!eigen_c.isApprox(mkl_c_check)){
		//	std::cout << "WRONG RESULT at " << i << std::endl;
		//	break;
		//}

		//Now for kenneth's Matrices
		Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> kenneth_a_tmp = A.cast<float>();
		Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> kenneth_b_tmp = B.cast<float>();

		alloc::AlignedVector<float, align> A_proto(M * K);
    	alloc::AlignedVector<float, align> B_proto(K * N);

    	std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_proto.get());
		std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_proto.get());


		float quant_mult = 127.0 / 2.0; //Ask what's happening
    	alloc::AlignedVector<int8_t, align> A_prepared(M * K);
    	alloc::AlignedVector<int8_t, align> B_prepared(K * N);

    	intgemm::Int8::PrepareA(A_proto.get(), A_prepared.get(), quant_mult, M, K);
    	// Quantize and reshape B.
    	// Typically you will do this once when parameters are loaded, not every time.
    	intgemm::Int8::PrepareB(A_proto.get(), B_prepared.get(), quant_mult, K, N);

		alloc::AlignedVector<float, align> C_kenn(M*N);

		auto kenn_start = std::chrono::system_clock::now();
		intgemm::Int8::Multiply(A_prepared.get(), B_prepared.get(), intgemm::JustUnquantizeC(C_kenn.get(), 1.0 / (quant_mult * quant_mult)), M, K, N);
		auto kenn_end = std::chrono::system_clock::now();

		kenn_duration_loop += (kenn_end - kenn_start);

		auto cblast_start = std::chrono::system_clock::now();
		int multStatus = openCLGemm(M, N, K);
		if (multStatus != CLBlastSuccess) {
			std::cerr << "OpenCL multiply failed: " << multStatus << std::endl;
			break;
		}
		auto cblast_end = std::chrono::system_clock::now();
		cblast_duration_loop += (cblast_end - cblast_start);


	}
	std::cout << std::fixed;
	std::cout.precision(10);
	std::cout << "In loop, Eigen took: " << eigen_duration_loop.count() << " seconds." << std::endl << 
	             "           MKL took: " << mkl_duration_loop.count() << " seconds." << std::endl << 
	             "Kenneth's work took: " << kenn_duration_loop.count() << " seconds." << std::endl << 
	             " OpenCL cBLAST took: " << cblast_duration_loop.count() << " seconds" << std::endl <<
	             "Alignment was: " << align << "." << std::endl;


    return 0;
}
