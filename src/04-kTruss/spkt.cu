#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>

#include "mtx_reader.h"
#include "utils.h"
#include "kt.h"

const int BLK_SIZE = 256;

__global__ 
void KT_Kernel_1(const int* IA, int* JA, float* M, const int NUM_VERTICE, const int K){
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < NUM_VERTICE){
        int i = row;
        int a12_start = *(IA+i);
        int a12_end = *(IA+i+1);
        int *JAL = JA + a12_start;

        for(int l = a12_start; *JAL !=0 && l!= a12_end; ++l){

            int A22_start = *(IA + *(JAL));
            int A22_end = *(IA + *(JAL) + l);
            JAL ++;

            float ML = 0;
            int *JAK = JAL;
            int *JAJ = JA + A22_start;
            float *MJ = M + A22_start;
            float *MK = M + l + 1;

            while(*JAK != 0 && *JAJ != 0 && JAK != JA + a12_end && JAJ !=JA + A22_end){
                int Jaj_val = *JAJ;
                int update_val = (Jaj_val == *JAK);

                if (update_val){
                    atomicAdd(MK, 1);
                    // ++(*MK);
                }

                ML += update_val;

                int tmp = *JAK;
                int advanceK = (tmp <= Jaj_val);
                int advanceJ = (Jaj_val <= tmp);

                JAK += advanceK;
                MK += advanceK;
                JAJ += advanceJ;

                if( update_val ){
                    atomicAdd(MJ, 1);
                    // ++(*MJ);
                }
                MJ += advanceJ;
            }
            
            // *(M+l) += ML;
            atomicAdd(M+l, ML);
        }
    }
}

__global__ 
void KT_Kernel_2(const int* IA, int* JA, float* M, const int NUM_VERTICE, const int K, int* not_finished){
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < NUM_VERTICE){

        int n = row;
        int st = *(IA+n);
        int end = *(IA+n+1);
        int *J = JA+st;
        int *Jk = JA+st;
        float *Mst = M + st;

        for(; *J != 0 && J!=JA+end; ++Mst, ++J){
            if (*Mst >= K-2) {
                *Jk = *J;
                Jk++;
            }
            *Mst = 0;
        }

        if(Jk<J){
            atomicMax(not_finished, 1);
            *Jk = 0;
        }


    }

}

void buildM(const SparseMatrixCSR A, const int K, const float * S, float * M){
    // #pragma omp parallel for
    for (int row = 0; row < A.M; ++row) {

        const int row_start = A.row_indices[row];
        const int row_end = A.row_indices[row+1];
        for (int element = row_start; element < row_end; ++element) {

            if(S[row*A.M + A.col_indices[element]] >= (K - 2.0)){
                M[element] = 1;    
            }
            else{
                M[element] = 0; 
            }

        }

    }

}

__global__
void buildM_kernel(const SparseMatrixCSR A, const int K, const float * S, float * M){

    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < A.M){
        const int row_start = A.row_indices[row];
        const int row_end = A.row_indices[row+1];
        for (int element = row_start; element < row_end; ++element) {
            if(S[row*A.M + A.col_indices[element]] >= K - 2){
                M[element] = 1;    
            }
            else{
                M[element] = 0; 
            }

        }
    }
}

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

void SpMM_CSR_ATA(const SparseMatrixCSR A, float * y)  // only for square
{

    for (int row = 0; row < A.M; ++row) {

        for (int col = 0; col < A.M; ++col) {
            float dotProduct = 0;
            const int row_start = A.row_indices[row];
            const int row_end = A.row_indices[row+1];

            const int col_start = A.row_indices[col];
            const int col_end = A.row_indices[col+1];

            int col_ptr = col_start;

            for (int element = row_start; element < row_end; ++element) {
                while( col_ptr < col_end && A.col_indices[col_ptr] < A.col_indices[element]){
                    ++col_ptr;
                }
                if(A.col_indices[col_ptr] == A.col_indices[element]){
                    dotProduct += A.values[col_ptr] * A.values[element];
                    // printf("YES: %f, %f \n", A.values[col_ptr], A.values[element]);
                }
            }

            int factor = 0;

            for (int element = col_start; element < col_end; ++element) {
                if(row == A.col_indices[element]){
                    factor = element;
                    break;
                }

            }

            y[factor] = dotProduct;
        }
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

__global__
void SpMM_CSR_ATA_kernel(const SparseMatrixCSR A, float * y)
{

    // TODO: write this function
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid / A.M;
    const int col = tid % A.M;

    if (row < A.M) {

        float dotProduct = 0;
        const int row_start = A.row_indices[row];
        const int row_end = A.row_indices[row+1];

        const int col_start = A.row_indices[col];
        const int col_end = A.row_indices[col+1];

        int col_ptr = col_start;

        for (int element = row_start; element < row_end; ++element) {
            while( col_ptr < col_end && A.col_indices[col_ptr] < A.col_indices[element]){
                ++col_ptr;
            }
            if(A.col_indices[col_ptr] == A.col_indices[element])
                dotProduct += A.values[col_ptr] * A.values[element];
        }

        int factor = 0;

        for (int element = col_start; element < col_end; ++element) {
            if(row == A.col_indices[element]){
                factor = element;
                break;
            }

        }

        y[factor] = dotProduct;
    }

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


void csrspmm_kt(const SparseMatrixCOO coo, int K, float * y, float * y_exp, bool ongpu, int opt, bool check)
{
    SparseMatrixCSR csr;
    if (!coo_to_csr(coo, &csr)) return;

    float run_time;

    // float* S = (float *) malloc (coo.NNZ * sizeof(float));
    // if ( !S ) exit (1);
    // set(S, coo.M * coo.M, 0);
    if ( !ongpu ) {
        double start_time = omp_get_wtime();
        SpMM_CSR_ATA(csr, y);
        // buildM(csr, K, S, y);
        int * col_indices = (int*) malloc(coo.NNZ*sizeof(int));
        memcpy(col_indices, csr.col_indices, coo.NNZ*sizeof(int));
        bool not_finished = true;
        while(not_finished){
            not_finished = serialKT(csr.row_indices, col_indices, y, csr.M, K );
        }
        run_time = omp_get_wtime() - start_time;
        free(col_indices);
    } else {
        int not_finished[] = {1};

        float * y_d;
        int * cid_d;
        // float * S_d;
        int * nf_d;

        if(opt == 0){
            malloc_and_copy( (void **) &y_d, (void *)y, coo.NNZ*sizeof(float));
            malloc_and_copy( (void **) &cid_d, (void *)csr.col_indices, coo.NNZ*sizeof(int));
            // malloc_and_copy( (void **) &S_d, (void *)S, coo.M*coo.M*sizeof(float));
            malloc_and_copy( (void **) &nf_d, (void *)not_finished, sizeof(int));
            SparseMatrixCSR csr_d;
            csr_to_device(csr, &csr_d);

            int coarse = 1;
            int bs = BLK_SIZE;
            int cgs=(coo.M*coo.M - 1)/(bs)/coarse+1;

            CUDA_TIMER_START;

            SpMM_CSR_ATA_kernel<<<cgs, bs>>>(csr_d, y_d);

            cgs=(coo.M - 1)/(bs)/coarse+1;

            // buildM_kernel<<<cgs, bs>>>(csr_d, K, S_d, y_d);

            while(not_finished[0] > 0){
                not_finished[0] = 0;
                cudaMemcpy(nf_d, not_finished, sizeof(int), cudaMemcpyHostToDevice);
                KT_Kernel_1<<<cgs, bs>>>(csr_d.row_indices, cid_d, y_d, csr_d.M, K);
                KT_Kernel_2<<<cgs, bs>>>(csr_d.row_indices, cid_d, y_d, csr_d.M, K, nf_d);
                cudaMemcpy(not_finished, nf_d, sizeof(int), cudaMemcpyDeviceToHost);
            }

            CUDA_TIMER_END(run_time);

            cudaMemcpy(y, y_d, coo.NNZ*sizeof(float), cudaMemcpyDeviceToHost);

            CUDA_CHK(cudaGetLastError());

            csr_free(csr_d, true);
        }
        else{
            cudaMallocManaged(&y_d, coo.NNZ*sizeof(float));
            // cudaMallocManaged(&S_d, coo.M*coo.M*sizeof(float));
            cudaMallocManaged(&cid_d, coo.NNZ*sizeof(int));
            cudaMallocManaged(&nf_d, sizeof(int));

            memcpy(y_d, y, coo.NNZ*sizeof(float));
            // memcpy(S_d, S, coo.M*coo.M*sizeof(float));
            memcpy(cid_d, csr.col_indices, coo.NNZ*sizeof(int));
            memcpy(nf_d, not_finished, sizeof(int));

            SparseMatrixCSR* csr_d;
            cudaMallocManaged(&csr_d, sizeof(SparseMatrixCSR));
            cudaMallocManaged(&(csr_d->row_indices), sizeof(int) * csr.N);
            cudaMallocManaged(&(csr_d->col_indices), sizeof(int) * csr.NNZ);
            cudaMallocManaged(&(csr_d->values), sizeof(float) * csr.NNZ);

            memcpy(csr_d->row_indices, csr.row_indices, sizeof(int) * csr.N);
            memcpy(csr_d->col_indices, csr.col_indices, sizeof(int) * csr.NNZ);
            memcpy(csr_d->values, csr.values, sizeof(float) * csr.NNZ);

            csr_d->M = csr.M;
            csr_d->N = csr.N;
            csr_d->NNZ = csr.NNZ;

            int coarse = 1;
            int bs = BLK_SIZE;
            int cgs=(coo.M*coo.M - 1)/(bs)/coarse+1;

            CUDA_TIMER_START;

            SpMM_CSR_ATA_kernel<<<cgs, bs>>>(*csr_d, y_d);

            cgs=(coo.M - 1)/(bs)/coarse+1;

            // buildM_kernel<<<cgs, bs>>>(*csr_d, K, S_d, y_d);

            while(nf_d[0] > 0){
                nf_d[0] = 0;
                KT_Kernel_1<<<cgs, bs>>>(csr_d->row_indices, cid_d, y_d, csr_d->M, K);
                KT_Kernel_2<<<cgs, bs>>>(csr_d->row_indices, cid_d, y_d, csr_d->M, K, nf_d);
            }

            CUDA_TIMER_END(run_time);

            memcpy(y, y_d, coo.NNZ*sizeof(float));

            CUDA_CHK(cudaGetLastError());
            cudaFree(csr_d->row_indices);
            cudaFree(csr_d->col_indices);
            cudaFree(csr_d);

        }
        // cudaFree(S_d);
        cudaFree(y_d);
        cudaFree(cid_d);
        cudaFree(nf_d);
    }
    // free(S);
    csr_free(csr, false);
    print_res("csr", run_time, check, y, y_exp, coo.NNZ);
}



// void run(const SparseMatrixCOO coo, const float * x, float * y, bool ongpu, int K, bool check)
// {
//     printf("%12s | %17s%13s", "code version", ongpu ? "gpu " : "cpu ", "run time(sec)");
//     if (check)
//         printf(" | %30s", "correctness");
//     printf("\n");

//     float * y_exp = NULL;
//     if (check) {
//         y_exp = (float *) malloc (coo.M * sizeof(float));
//         csrspmv(coo, x, y_exp, NULL, false, false);
//     }

//     csrspmv(coo, x, y, y_exp, ongpu, check);

//     if (!y_exp) free(y_exp);
// }


void runMM(const SparseMatrixCOO coo, int KT, float * y, bool ongpu, int op, bool check)
{
    printf("%12s | %17s%13s", "code version", ongpu ? "gpu " : "cpu ", "run time(sec)");
    if (check)
        printf(" | %30s", "correctness");
    printf("\n");

    float * y_exp = NULL;
    if (check) {
        y_exp = (float *) malloc (coo.NNZ * sizeof(float));
        set(y_exp, coo.NNZ, 0);
        csrspmm_kt(coo, KT, y_exp, NULL, false, op, false);
    }

    csrspmm_kt(coo, KT, y, y_exp, ongpu, op, check);

    // for(int i=0; i<coo.NNZ; ++i)
    // {
    //     if(y[i] > 0){
    //         printf("%.3f ", y[i]);
    //     }
    // }
    // printf("\n");
    if (!y_exp) free(y_exp);
}

void data_init_mm( SparseMatrixCOO coo, float ** y )
{
    *y = (float *) malloc (coo.NNZ * sizeof(float));
    if ( !*y ) exit (1);

    set(*y, coo.NNZ, 0);
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
    const int  KT = (argc > 3) ? atoi(argv[3]) : 2;
    const int  op = (argc > 4) ? atoi(argv[4]) : 2;
    const char * c = (argc > 5) ? argv[5] : "check";
    // reading mtx format file to coo matrix
    SparseMatrixCOO coo = read_edgelist(fname);
    float *y;
    // data_init(coo, &x, &y);
    data_init_mm(coo, &y);

    runMM(coo, KT, y, strcmp(platform, "gpu") == 0, op, strcmp(c, "check") == 0);
    // run(coo, x, y, strcmp(platform, "gpu") == 0, K, strcmp(c, "check") == 0);

    coo_free(coo, false);
    // free(x);
    free(y);

    return 0;
}