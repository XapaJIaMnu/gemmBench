#include <array>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mkldnn.h>
#include <chrono>
#include "intgemm/intgemm.h"
#include "aligned.h"

<<<<<<< HEAD

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

=======
>>>>>>> 8c08a62aef17215aa910b62587b2cf8c4bbd7cb4
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


	std::chrono::duration<double> eigen_duration_loop = std::chrono::duration<double>::zero();
	std::chrono::duration<double> mkl_duration_loop = std::chrono::duration<double>::zero();
	std::chrono::duration<double> kenn_duration_loop = std::chrono::duration<double>::zero();
<<<<<<< HEAD
	std::chrono::duration<double> cblast_duration_loop = std::chrono::duration<double>::zero();
	const size_t align = 256;
=======
	const size_t align = 64;
    bool kenngemm = true;
>>>>>>> 8c08a62aef17215aa910b62587b2cf8c4bbd7cb4

	for (int i = 0; i<10; i++) {

		char offsetc = 'F';
		bool zero_oa = 1;
		bool zero_ob = 1;
		bool zero_oc = 0;
		char transA = 'N';
		char transB = 'n';
<<<<<<< HEAD
		const int M = 320;
		const int N = 80;
		const int K = 320;
		float alpha = 2;
=======
		const int M = 1024;
		const int N = 1024;
		const int K = 1024;
		float alpha = 1;//2; Eigen is a buggy
>>>>>>> 8c08a62aef17215aa910b62587b2cf8c4bbd7cb4
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

<<<<<<< HEAD
		//Eigen bug: https://stackoverflow.com/questions/54738495/eigenmapd-matrix-from-raw-buffer-gives-object-allocated-on-stack-is-too-big/
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_a(A_EIGEN.get(), M, K);
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_b(B_EIGEN.get(), K, N);
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > eigen_c(C_EIGEN.get(), M, N);

		//Sanity check
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > mkl_c_check(C_MKL.get(), M, N);
=======
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > eigen_a(A_EIGEN.get(), M, K);
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > eigen_b(B_EIGEN.get(), K, N);
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > eigen_c(C_EIGEN.get(), M, N);

		//Sanity check
		Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > mkl_c_check(C_MKL.get(), M, N);
>>>>>>> 8c08a62aef17215aa910b62587b2cf8c4bbd7cb4



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

        if (M%8 == 0 && N%8 == 0 && K%32 == 0) {
            float quant_mult = 127.0 / 2.0; //Ask what's happening
            alloc::AlignedVector<int8_t, align> A_prepared(M * K);
            alloc::AlignedVector<int8_t, align> B_prepared(K * N);

            intgemm::Int8::PrepareA(A_proto.get(), A_prepared.get(), quant_mult, M, K);
            // Quantize and reshape B.
            // Typically you will do this once when parameters are loaded, not every time.
            intgemm::Int8::PrepareB(A_proto.get(), B_prepared.get(), quant_mult, K, N);

            alloc::AlignedVector<float, align> C_kenn(M*N);

            auto kenn_start = std::chrono::system_clock::now();
            intgemm::Int8::Multiply(A_prepared.get(), B_prepared.get(), C_kenn.get(), 1.0 / (quant_mult * quant_mult), M, K, N);
            auto kenn_end = std::chrono::system_clock::now();

<<<<<<< HEAD
		auto kenn_start = std::chrono::system_clock::now();
		intgemm::Int8::Multiply(A_prepared.get(), B_prepared.get(), intgemm::JustUnquantizeC(C_kenn.get(), 1.0 / (quant_mult * quant_mult)), M, K, N);
		auto kenn_end = std::chrono::system_clock::now();
=======
            kenn_duration_loop += (kenn_end - kenn_start);
>>>>>>> 8c08a62aef17215aa910b62587b2cf8c4bbd7cb4

        } else {
            kenngemm = false;
        }

		auto cblast_start = std::chrono::system_clock::now();
		int multStatus = openCLGemm(M, N, K);
		if (multStatus != CLBlastSuccess) {
			std::cerr << "OpenCL multiply failed: " << multStatus << std::endl;
			break;
		}
		auto cblast_end = std::chrono::system_clock::now();
		cblast_duration_loop += (cblast_end - cblast_start);


	}
<<<<<<< HEAD
	std::cout << std::fixed;
	std::cout.precision(10);
	std::cout << "In loop, Eigen took: " << eigen_duration_loop.count() << " seconds." << std::endl << 
	             "           MKL took: " << mkl_duration_loop.count() << " seconds." << std::endl << 
	             "Kenneth's work took: " << kenn_duration_loop.count() << " seconds." << std::endl << 
	             " OpenCL cBLAST took: " << cblast_duration_loop.count() << " seconds" << std::endl <<
	             "Alignment was: " << align << "." << std::endl;
=======
    if (kenngemm) {
    	std::cout << std::fixed;
    	std::cout.precision(10);
    	std::cout << "In loop, Eigen took: " << eigen_duration_loop.count() << " seconds. " << std::endl << 
    	             "           MKL took: " << mkl_duration_loop.count() << " seconds. " << std::endl << 
    	             "Kenneth's work took: " << kenn_duration_loop.count() << std::endl << 
    	             "Alignment was: " << align << "." << std::endl;
    } else {
        std::cout << std::fixed;
        std::cout.precision(10);
        std::cout << "In loop, Eigen took: " << eigen_duration_loop.count() << " seconds. " << std::endl << 
                     "           MKL took: " << mkl_duration_loop.count() << " seconds. " << std::endl << 
                     "Kenneth's work was excluded from this benchmark due to matrix sizes not being appropriate." << std::endl << 
                     "Alignment was: " << align << "." << std::endl;
    }
>>>>>>> 8c08a62aef17215aa910b62587b2cf8c4bbd7cb4


    return 0;
}
