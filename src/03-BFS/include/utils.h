#ifndef _SPMV_UTILS
#define _SPMV_UTILS

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHK(cerr) do {cudaError_t _cerr = (cerr); if ((_cerr) != cudaSuccess) {fprintf(stderr,"[%s, %d] Cuda error %s\n", __FILE__, __LINE__, cudaGetErrorString(_cerr)); exit(1);}} while(0)

#define CUDA_TIMER_START \
       cudaEvent_t start, stop; \
       cudaEventCreate(&start);   cudaEventCreate(&stop); \
       cudaEventRecord(start, 0); \
       cudaEventSynchronize(start);

#define CUDA_TIMER_END(t) \
       cudaEventRecord(stop, 0); \
       cudaEventSynchronize(stop); \
       cudaEventElapsedTime(&t, start, stop); \
       t /= (float)1000; \
       cudaEventDestroy(start);  cudaEventDestroy(stop);

#define FREE(ptr, ongpu) do { \
       if (ongpu) cudaFree(ptr); else free(ptr); } while(0)

#include <math.h>

#define DEBUG

struct SparseMatrixCOO {
	float * values;
	int * col_indices;
	int * row_indices;
	int M;
	int N;
	int NNZ;
};

struct SparseMatrixCSR {
	float * values;
	int * col_indices;
	int * row_indices; // pointer of each row's starting position of col indices and values
	int M; // number of rows
	int N; // numver of cols
	int NNZ;
};

struct SparseMatrixELL {
	float * values;
	int * col_indices;
	int M;
	int N;
	int K;
};

void malloc_and_copy(void ** d, void * h, int size);

void coo_to_device(SparseMatrixCOO coo_h, SparseMatrixCOO * coo_d);

void coo_free(SparseMatrixCOO coo, bool ongpu);

int coo_max_deg(const SparseMatrixCOO coo);

void csr_to_device(SparseMatrixCSR csr_h, SparseMatrixCSR * csr_d);
void csr_to_device_uvm(SparseMatrixCSR csr_h, SparseMatrixCSR * csr_d);

void csr_free(SparseMatrixCSR csr, bool ongpu);

void ell_to_device(SparseMatrixELL ell_h, SparseMatrixELL * ell_d);

void ell_free(SparseMatrixELL ell, bool ongpu);


typedef struct Edge {
    int row_index;
    int col_index;
    float value;
} Edge;

int edge_comp (const void * a, const void * b);

// void set( float * y, int M, float v);

void set( int * y, int M, int v);

// void set( float * y, int M, float v);

void data_init( SparseMatrixCOO coo, float ** x, float ** y );

bool same( float * y_exp, float * y, int M );

void print_res(const char * codeversion, double run_time, bool check, float * y, float * y_exp, int M);

bool sort_coo(const SparseMatrixCOO coo, SparseMatrixCOO * sorted_coo);

bool coo_to_csr(SparseMatrixCOO coo, SparseMatrixCSR * csr);

int max_deg(const SparseMatrixCOO coo);

bool coo_to_ell(SparseMatrixCOO coo, SparseMatrixELL * ell);

bool coo_to_hybrid(const SparseMatrixCOO coo, SparseMatrixELL * ell_hyb, SparseMatrixCOO * coo_hyb, int K);

void print_res_bfs(const char * codeversion, double run_time, bool check, int * y, int * y_exp, int M);

void data_init_bfs( SparseMatrixCOO coo, int ** prev, int ** level );

#endif
