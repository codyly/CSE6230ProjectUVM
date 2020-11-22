#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>

#include "mtx_reader.h"
#include "utils.h"

const int BLK_SIZE = 256;

void SpMV_CSR(const SparseMatrixCSR A, const float * x, float * y)
{

    for (int row = 0; row < A.M; ++row) {

        float dotProduct = 0;
        const int row_start = A.row_indices[row];
        const int row_end = A.row_indices[row+1];
        for (int element = row_start; element < row_end; ++element) {

            dotProduct += A.values[element] * x[A.col_indices[element]];

        }

        y[row] = dotProduct;

    }

}

__global__
void SpMV_CSR_kernel(const SparseMatrixCSR A, const float * x, float * y)
{

    // TODO: write this function
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.M) {

        float dotProduct = 0.0f;
        const int row_start = A.row_indices[row];
        const int row_end = A.row_indices[row+1];
        for(int element=row_start; element < row_end; element++)
        {
            dotProduct += A.values[element] * x[A.col_indices[element]];
        }

        y[row] = dotProduct;
    }

}

void SpMV_ELL(const SparseMatrixELL A, const float * x, float * y)
{

    for (int row = 0; row < A.M; ++row) {

        float dotProduct = 0;

        for (int element = 0; element < A.K; ++element) {

            const int elementIndex = row + element* A.M;
            dotProduct += A.values[elementIndex] * x[A.col_indices[elementIndex]];

        }

        y[row] = dotProduct;

    }

}

__global__
void SpMV_ELL_kernel(const SparseMatrixELL A, const float * x, float * y)
{

    // TODO: write this function
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.M) {
        float dotProduct = 0;
        for (int element = 0; element < A.K; ++element) {
            const int elemIndex = row + element* A.M;
            if (A.values[elemIndex]) {
                dotProduct += A.values[elemIndex] * x[A.col_indices[elemIndex]];
            }
        }
        y[row] = dotProduct;
    }

}

void SpMV_COO(const SparseMatrixCOO A, const float * x, float * y)
{

    for (int element = 0; element < A.NNZ; ++element) {

        const int column = A.col_indices[element];
        const int row = A.row_indices[element];

        y[row] += A.values[element] * x[column];

    }

}

__global__
void SpMV_COO_kernel(const SparseMatrixCOO A, const float * x, float * y)
{

    // TODO: write this function
    for (int element = threadIdx.x + blockIdx.x * blockDim.x;
        element < A.NNZ;
        element += blockDim.x * gridDim.x) {

        const int column = A.col_indices[element];
        const int row = A.row_indices[element];
        
        atomicAdd(&y[row], A.values[element] * x[column]);
    }
}

void coospmv(const SparseMatrixCOO _coo, const float * x, float * y, float * y_exp, bool ongpu, bool sorted, bool check)
{
    SparseMatrixCOO coo = _coo;
    if (sorted)
        if (!sort_coo( _coo, &coo )) return ;

    set(y, coo.M, 0.0);

    float run_time;

    if ( !ongpu ) {
        double start_time = omp_get_wtime();
        SpMV_COO(coo, x, y);
        run_time = (float)(omp_get_wtime() - start_time);
    } else {
        float * x_d, * y_d;
        malloc_and_copy( (void **) &x_d, (void *)x, coo.N*sizeof(float));
        malloc_and_copy( (void **) &y_d, (void *)y, coo.M*sizeof(float));

        SparseMatrixCOO coo_d;
        coo_to_device(coo, &coo_d);

        int coarse = 32;
        int bs = BLK_SIZE;
        int cgs=(coo.NNZ-1)/(bs)/coarse+1;

        CUDA_TIMER_START;

        SpMV_COO_kernel<<<cgs, bs>>>(coo_d, x_d, y_d);

        CUDA_TIMER_END(run_time);

        cudaMemcpy(y, y_d, coo.M*sizeof(float), cudaMemcpyDeviceToHost);

        CUDA_CHK(cudaGetLastError());

        coo_free(coo_d, true);
        cudaFree(x_d);
        cudaFree(y_d);
    }

    if (sorted)
        coo_free(coo, false);

    print_res(sorted ? "coo_sorted" : "coo_unsorted", run_time, check, y, y_exp, coo.M);
}

void csrspmv(const SparseMatrixCOO coo, const float * x, float * y, float * y_exp, bool ongpu, bool check)
{
    SparseMatrixCSR csr;
    if (!coo_to_csr(coo, &csr)) return;

    float run_time;

    if ( !ongpu ) {
        double start_time = omp_get_wtime();
        SpMV_CSR(csr, x, y);
        run_time = omp_get_wtime() - start_time;
    } else {
        float * x_d, * y_d;
        malloc_and_copy( (void **) &x_d, (void *)x, coo.N*sizeof(float));
        malloc_and_copy( (void **) &y_d, (void *)y, coo.M*sizeof(float));

        SparseMatrixCSR csr_d;
        csr_to_device(csr, &csr_d);

        int coarse = 1;
        int bs = BLK_SIZE;
        int cgs=(coo.M-1)/(bs)/coarse+1;

        CUDA_TIMER_START;

        SpMV_CSR_kernel<<<cgs, bs>>>(csr_d, x_d, y_d);

        CUDA_TIMER_END(run_time);

        cudaMemcpy(y, y_d, coo.M*sizeof(float), cudaMemcpyDeviceToHost);

        CUDA_CHK(cudaGetLastError());

        csr_free(csr_d, true);
        cudaFree(x_d);
        cudaFree(y_d);
    }

    csr_free(csr, false);
    print_res("csr", run_time, check, y, y_exp, coo.M);
}

void ellspmv(const SparseMatrixCOO coo, const float * x, float * y, float * y_exp, bool ongpu, bool check)
{
    SparseMatrixELL ell;
    if (!coo_to_ell(coo, &ell)) return;

    float run_time;

    if ( !ongpu ) {
        double start_time = omp_get_wtime();
        SpMV_ELL(ell, x, y);
        run_time = omp_get_wtime() - start_time;
    } else {
        float * x_d, * y_d;
        malloc_and_copy( (void **) &x_d, (void *)x, coo.N*sizeof(float));
        malloc_and_copy( (void **) &y_d, (void *)y, coo.M*sizeof(float));

        SparseMatrixELL ell_d;
        ell_to_device(ell, &ell_d);

        int coarse = 1;
        int bs = BLK_SIZE;
        int cgs=(coo.M-1)/(bs)/coarse+1;

        CUDA_TIMER_START;

        SpMV_ELL_kernel<<<cgs, bs>>>(ell_d, x_d, y_d);

        CUDA_TIMER_END(run_time);

        cudaMemcpy(y, y_d, coo.M*sizeof(float), cudaMemcpyDeviceToHost);

        CUDA_CHK(cudaGetLastError());

        ell_free(ell_d, true);
        cudaFree(x_d);
        cudaFree(y_d);
    }

    ell_free(ell, false);
    print_res("ell", run_time, check, y, y_exp, coo.M);
}

void hybridspmv(const SparseMatrixCOO coo, const float * x, float * y, float * y_exp, bool ongpu, int K, bool check)
{
    // TODO: write this function

    // 1. convert input to Hybrid (ELL+COO) format

    SparseMatrixCOO hyb_coo;
    SparseMatrixELL hyb_ell;

    float *y_ell = (float*)calloc(coo.M, sizeof(float));
    for (int i=0; i<coo.M; ++i) {
        y_ell[i] = 0;
        y[i] = 0;
    }

    if(!coo_to_hybrid(coo, &hyb_ell, &hyb_coo, K)) return;

    float run_time;

    if ( !ongpu ) {
        double start_time = omp_get_wtime();
        // 2. run SpMV on ELL CPU
        SpMV_ELL(hyb_ell, x, y_ell);
        // 3. run SpMV on COO CPU
        SpMV_COO(hyb_coo, x, y);

        run_time = omp_get_wtime() - start_time;
        
    } else {
        float * x_d, * y_d, * y_ell_d;
        malloc_and_copy( (void **) &x_d, (void *)x, coo.N*sizeof(float));
        malloc_and_copy( (void **) &y_d, (void *)y, coo.M*sizeof(float));
        malloc_and_copy( (void **) &y_ell_d, (void *)y_ell, coo.M*sizeof(float));

        SparseMatrixELL ell_d;
        ell_to_device(hyb_ell, &ell_d);

        SparseMatrixCOO coo_d;
        coo_to_device(hyb_coo, &coo_d);

        int coarse = 1;
        int bs = BLK_SIZE;
        int cgs=(coo.M-1)/(bs)/coarse+1;

        CUDA_TIMER_START;

        // 2. run SpMV on ELL GPU
        SpMV_ELL_kernel<<<cgs, bs>>>(ell_d, x_d, y_ell_d);

        // 3. run SpMV on COO GPU
        SpMV_COO_kernel<<<cgs, bs>>>(coo_d, x_d, y_d);

        CUDA_TIMER_END(run_time);

        cudaMemcpy(y, y_d, coo.M*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(y_ell, y_ell_d, coo.M*sizeof(float), cudaMemcpyDeviceToHost);

        CUDA_CHK(cudaGetLastError());

        ell_free(ell_d, true);
        coo_free(coo_d, true);
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(y_ell_d);
    }

    for (int i=0; i<coo.M; ++i) {
        y[i] += y_ell[i];
    }

    // 4. print and check the results
    ell_free(hyb_ell, false);
    coo_free(hyb_coo, false);

// #ifdef DEBUG   
//     for (int i=0; i<coo.M; ++i) {
//         printf("%f %f\n", y[i], y_exp[i]);
//     }
// #endif
    print_res("hyb", run_time, check, y, y_exp, coo.M);
}

void run(const SparseMatrixCOO coo, const float * x, float * y, bool ongpu, int K, bool check)
{
    printf("%12s | %17s%13s", "code version", ongpu ? "gpu " : "cpu ", "run time(sec)");
    if (check)
        printf(" | %30s", "correctness");
    printf("\n");

    float * y_exp = NULL;
    if (check) {
        y_exp = (float *) malloc (coo.M * sizeof(float));
        csrspmv(coo, x, y_exp, NULL, false, false);
    }

    coospmv(coo, x, y, y_exp, ongpu, false, check);

    coospmv(coo, x, y, y_exp, ongpu, true, check);

    csrspmv(coo, x, y, y_exp, ongpu, check);

    ellspmv(coo, x, y, y_exp, ongpu, check);

    hybridspmv(coo, x, y, y_exp, ongpu, K, check);

    if (!y_exp) free(y_exp);
}

int main(int argc, char * argv[] )
{
    if (argc<2) {
        printf("usage: %s <matrix path name> [platform] [K] [check]\n", argv[0]);
        printf("       matrix path name : mtx matrix format path name\n");
        printf("       platform         : gpu(default), cpu\n");
        printf("       K                : K for ELL in Hybrid format, 8 as default\n");
        printf("       check            : check(default) or not, if check, verify results, otherwise not\n");
        exit(1);
    }

    const char * fname = argv[1];
    const char * platform = (argc > 2) ? argv[2] : "gpu";
    const int  K = (argc > 3) ? atoi(argv[3]) : 8;
    const char * c = (argc > 4) ? argv[4] : "check";
    // reading mtx format file to coo matrix
    SparseMatrixCOO coo = read_edgelist(fname);
    float *x, *y;
    data_init(coo, &x, &y);

    run(coo, x, y, strcmp(platform, "gpu") == 0, K, strcmp(c, "check") == 0);

    coo_free(coo, false);
    free(x);
    free(y);

    return 0;
}
