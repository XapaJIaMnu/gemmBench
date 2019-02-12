#include <array>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mkldnn.h>

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

int main(int argc, char const *argv[]) {
    /* code */
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
/*
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

*/
    c1_eigen.noalias() += a1_eigen*b1_eigen;

    std::cout  << c1_eigen << std::endl;

//    c_eigen.noalias() += a_eigen*b_eigen;

    //std::cout << c_eigen << std::endl;

    //MKLGEMM

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
int lda = 60;
int ldb = 50;
int ldc = 80;
bool expect_to_fail = 0;
alignas(64) const std::array<int8_t, M*N> A = {-4,2,-3,2,0,6,4,-2,8,2,1,-9,-8,-5,5,-10,6,9,1,0,
                                        -4,-3,0,4,8,1,-8,5,6,1,6,7,10,6,2,-3,-6,-10,6,8,
                                        -4,2,4,2,-8,-10,1,-6,10,-5,-3,-4,6,-10,-10,-7,-7,4,-7,-4,
                                        -4,0,8,-8,6,6,10,2,-6,6,5,2,-7,8,2,-2,8,-1,8,-8,
                                        -4,6,1,-10,6,9,-3,5,5,-7,-5,-8,-2,-6,6,1,-7,-5,-6,-1,
                                        -4,4,-8,1,10,-3,6,-10,-7,-7,-2,-3,8,-7,-10,-6,2,-4,7,3,
                                        -4,-2,5,-6,2,5,-10,7,2,1,6,-9,-10,1,1,-2,-4,5,-6,0,
                                        -4,8,6,10,-6,5,-7,2,8,8,-7,1,2,1,-4,2,8,6,8,-9,
                                        -4,2,1,-5,6,-7,-7,1,8,9,-6,-3,7,-3,-6,7,8,-1,-7,-2,
                                        -4,1,6,-3,5,-5,-2,6,-7,-6,-1,1,2,-7,8,6,2,2,-9,-4,
                                        -4,-9,7,-4,2,-8,-3,-9,1,-3,1,-6,-4,-9,-1,5,5,1,9,2,
                                        -4,-8,10,6,-7,-2,8,-10,2,7,2,-4,8,0,8,-9,-4,-2,0,2,
                                        -4,-5,6,-10,8,-6,-7,1,1,-3,-7,-9,0,-1,-8,5,1,2,6,-5,
                                        -4,5,2,-10,2,6,-10,1,-4,-6,8,-1,8,-8,10,-5,-1,2,10,-10,
                                        -4,-10,-3,-7,-2,1,-6,-2,2,7,6,5,-9,5,-5,-10,-4,-4,-10,-9,
                                        -4,6,-6,-7,8,-7,2,-4,8,8,2,5,-4,1,-1,-4,-7,1,10,6,
                                        -4,9,-10,4,-1,-5,-4,5,6,-1,2,1,-2,2,2,-4,1,7,-10,3,
                                        -4,1,6,-7,8,-6,7,-6,8,-7,-9,9,0,6,10,-10,10,-10,9,2,
                                        -4,0,8,-4,-8,-1,3,0,-9,-2,-4,2,2,-5,-10,-9,6,3,2,5,
                                        -4,6,5,-2,-7,-1,2,8,2,-9,-9,-5,-4,3,5,-8,1,8,1,3,
                                        -4,-6,-10,1,-10,-2,-6,-9,8,3,-5,-1,10,9,6,-6,4,9,5,-9,
                                        -4,7,2,-3,1,1,-4,-1,5,9,-5,7,-4,7,5,8,-7,-3,-3,-4,
                                        -4,6,4,-7,-5,-4,-1,5,5,-7,7,-8,4,8,-1,3,-10,-7,3,0,
                                        -4,10,-7,8,2,2,8,8,-4,0,-4,-4,10,-1,4,1,1,-5,-5,7,
                                        -4,9,-5,1,-1,6,6,8,-9,-10,-8,3,-8,9,-4,-9,-3,9,2,-1,
                                        -4,6,8,-7,1,-7,0,-8,1,6,3,7,-1,0,8,-5,7,-6,7,-7,
                                        -4,-5,-7,9,7,7,-7,3,0,3,-10,-5,9,-7,5,-10,-5,10,-1,-10,
                                        -4,2,2,-10,-4,8,8,10,-1,10,5,5,4,8,-8,-10,3,9,6,9,
                                        -4,8,9,6,-6,-7,0,-7,8,-1,-6,-9,5,-8,-8,-8,-10,-7,-2,4,
                                        -4,-3,-2,-6,2,6,-9,-5,-4,-10,-8,8,1,-5,-10,2,-1,-2,-6,-6};
                                        

alignas(64) const std::array<int8_t, N*K> B = {-4,2,-3,2,0,6,4,-2,8,2,
                                        -4,-3,0,4,8,1,-8,5,6,1,
                                        -4,2,4,2,-8,-10,1,-6,10,-5,
                                        -4,0,8,-8,6,6,10,2,-6,6,
                                        -4,6,1,-10,6,9,-3,5,5,-7,
                                        -4,4,-8,1,10,-3,6,-10,-7,-7,
                                        -4,-2,5,-6,2,5,-10,7,2,1,
                                        -4,8,6,10,-6,5,-7,2,8,8,
                                        -4,2,1,-5,6,-7,-7,1,8,9,
                                        -4,1,6,-3,5,-5,-2,6,-7,-6,
                                        -4,-9,7,-4,2,-8,-3,-9,1,-3,
                                        -4,-8,10,6,-7,-2,8,-10,2,7,
                                        -4,-5,6,-10,8,-6,-7,1,1,-3,
                                        -4,5,2,-10,2,6,-10,1,-4,-6,
                                        -4,-10,-3,-7,-2,1,-6,-2,2,7,
                                        -4,6,-6,-7,8,-7,2,-4,8,8,
                                        -4,9,-10,4,-1,-5,-4,5,6,-1,
                                        -4,1,6,-7,8,-6,7,-6,8,-7,
                                        -4,0,8,-4,-8,-1,3,0,-9,-2,
                                        -4,6,5,-2,-7,-1,2,8,2,-9};
                                        

alignas(64) std::array<int32_t, M*K> C = {-4,2,-3,2,0,6,4,-2,8,2,
                                        -4,-3,0,4,8,1,-8,5,6,1,
                                        -4,2,4,2,-8,-10,1,-6,10,-5,
                                        -4,0,8,-8,6,6,10,2,-6,6,
                                        -4,6,1,-10,6,9,-3,5,5,-7,
                                        -4,4,-8,1,10,-3,6,-10,-7,-7,
                                        -4,-2,5,-6,2,5,-10,7,2,1,
                                        -4,8,6,10,-6,5,-7,2,8,8,
                                        -4,2,1,-5,6,-7,-7,1,8,9,
                                        -4,1,6,-3,5,-5,-2,6,-7,-6,
                                        -4,-9,7,-4,2,-8,-3,-9,1,-3,
                                        -4,-8,10,6,-7,-2,8,-10,2,7,
                                        -4,-5,6,-10,8,-6,-7,1,1,-3,
                                        -4,5,2,-10,2,6,-10,1,-4,-6,
                                        -4,-10,-3,-7,-2,1,-6,-2,2,7,
                                        -4,6,-6,-7,8,-7,2,-4,8,8,
                                        -4,9,-10,4,-1,-5,-4,5,6,-1,
                                        -4,1,6,-7,8,-6,7,-6,8,-7,
                                        -4,0,8,-4,-8,-1,3,0,-9,-2,
                                        -4,6,5,-2,-7,-1,2,8,2,-9,
                                        -4,-6,-10,1,-10,-2,-6,-9,8,3,
                                        -4,7,2,-3,1,1,-4,-1,5,9,
                                        -4,6,4,-7,-5,-4,-1,5,5,-7,
                                        -4,10,-7,8,2,2,8,8,-4,0,
                                        -4,9,-5,1,-1,6,6,8,-9,-10,
                                        -4,6,8,-7,1,-7,0,-8,1,6,
                                        -4,-5,-7,9,7,7,-7,3,0,3,
                                        -4,2,2,-10,-4,8,8,10,-1,10,
                                        -4,8,9,6,-6,-7,0,-7,8,-1,
                                        -4,-3,-2,-6,2,6,-9,-5,-4,-10};


int8_t oa = 0;
int8_t ob = 0;
std::array<int32_t, 7> oc = {0,0,0,0,0,0,0};
auto status = mkldnn_gemm_s8s8s32(&transA, &transB, &offsetc,
        &M, &N, &K, &alpha, A.data(), &lda, &oa, B.data(), &ldb, &ob,
        &beta, C.data(), &ldc, oc.data());

printMKLdnnStatus(status);

    return 0;
}
