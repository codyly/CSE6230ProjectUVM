#include "utils.h"

void malloc_and_copy(void ** d, void * h, int size)
{
    cudaMalloc(d, size);
    cudaMemcpy(*d, h, size, cudaMemcpyHostToDevice);
}

void coo_to_device(SparseMatrixCOO coo_h, SparseMatrixCOO * coo_d)
{
    int size;

    size = (coo_h.NNZ) * sizeof(float);
    malloc_and_copy((void **) &((*coo_d).values), coo_h.values, size);

    size = (coo_h.NNZ) * sizeof(int);
    malloc_and_copy((void **) &((*coo_d).col_indices), coo_h.col_indices, size);

    size = (coo_h.NNZ) * sizeof(int);
    malloc_and_copy((void **) &((*coo_d).row_indices), coo_h.row_indices, size);

    (*coo_d).M = coo_h.M;
    (*coo_d).N = coo_h.N;
    (*coo_d).NNZ = coo_h.NNZ;
}

void coo_free(SparseMatrixCOO coo, bool ongpu)
{
    FREE(coo.values, ongpu);
    FREE(coo.col_indices, ongpu);
    FREE(coo.row_indices, ongpu);
}

void csr_to_device(SparseMatrixCSR csr_h, SparseMatrixCSR * csr_d)
{
    int size;

    size = (csr_h.row_indices[csr_h.M])*sizeof(float);
    malloc_and_copy((void **) &((*csr_d).values), csr_h.values, size);

    size = (csr_h.row_indices[csr_h.M])*sizeof(int);
    malloc_and_copy((void **) &((*csr_d).col_indices), csr_h.col_indices, size);

    size = (csr_h.M + 1)*sizeof(int);
    malloc_and_copy((void **) &((*csr_d).row_indices), csr_h.row_indices, size);

    (*csr_d).M = csr_h.M;
    (*csr_d).N = csr_h.N;
    (*csr_d).NNZ = csr_h.NNZ;
}

void csr_free(SparseMatrixCSR csr, bool ongpu)
{
    FREE(csr.values, ongpu);
    FREE(csr.row_indices, ongpu);
    FREE(csr.col_indices, ongpu);
}

int edge_comp (const void * a, const void * b)
{
    if ( (*(Edge*)a).row_index <  (*(Edge*)b).row_index ) return -1;
    else if ( (*(Edge*)a).row_index > (*(Edge*)b).row_index ) return 1;
    if ( (*(Edge*)a).col_index <  (*(Edge*)b).col_index ) return -1;
    else if ( (*(Edge*)a).col_index > (*(Edge*)b).col_index ) return 1;
    return 0;
}


void set( float * x, int M)
{
#pragma omp parallel for
    for (int i=0; i<M; ++i)
        x[i] = i;
}

void set( float * y, int M, float v)
{
#pragma omp parallel for
    for (int i=0; i<M; ++i)
        y[i] = v;
}

void data_init( SparseMatrixCOO coo, float ** x, float ** y )
{
    *x = (float *) malloc (coo.N * sizeof(float));
    *y = (float *) malloc (coo.M * sizeof(float));
    if ( !*x || !*y ) exit (1);

    set(*x, coo.N);
    set(*y, coo.M, 0);
}

bool same( float * y_exp, float * y, int M )
{
    if ( !y_exp || !y ) exit(1);

    for (int i=0; i<M; i++)
        if ((abs(y_exp[i] - y[i]) > 1e-3) && (abs(y_exp[i] - y[i]) > (1e-3) * max(abs(y_exp[i]), abs(y[i])))) {
        #ifdef DEBUG
            fprintf(stderr, "differs at: %d with values %f(expected) %f\n", i, y_exp[i], y[i]);
        #endif
            return false;
        }

    return true;
}

void print_res(const char * codeversion, double run_time, bool check, float * y, float * y_exp, int M)
{
   printf("%12s | %30.6lf", codeversion, run_time);
   if ( check )
       printf(" | %30s", same(y_exp, y, M)? "Y" : "N");
   printf("\n");
}

bool sort_coo(const SparseMatrixCOO coo, SparseMatrixCOO * sorted_coo)
{
    Edge * cooedges = (Edge *) malloc (coo.NNZ * sizeof(Edge));
    if (!cooedges) { fprintf(stderr, "out of space while sorting coo"); return false;}
#pragma omp parallel for
    for (int i=0; i<coo.NNZ; i++) {
        cooedges[i].row_index = coo.row_indices[i];
        cooedges[i].col_index = coo.col_indices[i];
        cooedges[i].value = coo.values[i];
    }

    qsort(cooedges, coo.NNZ, sizeof(Edge), edge_comp);

    sorted_coo->row_indices = (int *) malloc (coo.NNZ * sizeof(int));
    sorted_coo->col_indices = (int *) malloc (coo.NNZ * sizeof(int));
    sorted_coo->values = (float *) malloc (coo.NNZ * sizeof(float));
    if (!sorted_coo->row_indices || !sorted_coo->col_indices || !sorted_coo->values) {
        fprintf(stderr, "out of space while sorting coo");
        return false;
    }

    sorted_coo->NNZ = coo.NNZ;
    sorted_coo->M = coo.M;
    sorted_coo->N = coo.N;

#pragma omp parallel for
    for (int i=0; i<coo.NNZ; i++) {
        sorted_coo->row_indices[i] = cooedges[i].row_index;
        sorted_coo->col_indices[i] = cooedges[i].col_index;
        sorted_coo->values[i] = cooedges[i].value;
    }

#ifdef DEBUG
#pragma omp parallel for
    for (int i=1; i<sorted_coo->NNZ; i++) {
        if (sorted_coo->row_indices[i] < sorted_coo->row_indices[i-1]) exit(1);
        else if (sorted_coo->row_indices[i] == sorted_coo->row_indices[i-1]) {
            if (sorted_coo->col_indices[i] < sorted_coo->col_indices[i-1]) exit(1);
        }
    }
#endif

    free(cooedges);

    return true;
}

bool coo_to_csr(const SparseMatrixCOO coo, SparseMatrixCSR * csr)
{
    csr->M = coo.M;
    csr->N = coo.N;

    csr->row_indices = (int *) calloc( (coo.M + 1), sizeof(int) );
    csr->col_indices = (int *) malloc( coo.NNZ * sizeof(int) );
    csr->values = (float *) malloc( coo.NNZ * sizeof(float) );
    int * count = (int *) calloc ( coo.M, sizeof(int));
    if (!csr->row_indices || !csr->col_indices || !csr->values || !count) {
        fprintf(stderr, "out of space in transfering coo to csr\n");
        return false;
    }

    for (int i=0; i<coo.NNZ; ++i) {
        int r = coo.row_indices[i];
        if (r < coo.M - 1)
            csr->row_indices[r+2]++;
    }

    for (int i=1; i<=coo.M; ++i)
        csr->row_indices[i] += csr->row_indices[i-1];

    for (int i=0; i<coo.NNZ; ++i) {
        int r = coo.row_indices[i];
        csr->col_indices[csr->row_indices[r+1]] = coo.col_indices[i];
        csr->values[csr->row_indices[r+1]] = coo.values[i];
        csr->row_indices[r+1]++;
    }

    csr->NNZ = coo.NNZ;

    free(count);
#ifdef DEBUG
    if (csr->row_indices[coo.M] != coo.NNZ) { fprintf(stderr, "csr generated incorrectly\n"); exit(1); }
#endif
    return true;
}

int coo_max_deg(const SparseMatrixCOO coo)
{
    int * degrees = (int *) calloc (coo.M, sizeof(int));
    if (!degrees) exit(1);
    for (int i=0; i<coo.NNZ; ++i) {
        int r = coo.row_indices[i];
        degrees[r]++;
    }

    int max_deg = 0; 
    for (int i=0; i<coo.M; ++i) {
        if (degrees[i] > max_deg)
            max_deg = degrees[i];
    }

    free(degrees);

    return max_deg;
}
