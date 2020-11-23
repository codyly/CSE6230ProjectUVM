#include "utils.h"

void malloc_and_copy(void ** d, void * h, int size)
{
    cudaMalloc(d, size);
    cudaMemcpy(*d, h, size, cudaMemcpyHostToDevice);
}

void malloc_and_copy_uvm(void ** d, void * h, int size)
{
    cudaMallocManaged(d, size);
    // cudaMemcpy(*d, h, size, cudaMemcpyDefault);
    memcpy(*d, h, size);
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
}

void csr_to_device_uvm(SparseMatrixCSR csr_h, SparseMatrixCSR * csr_d)
{
    int size;

    size = (csr_h.row_indices[csr_h.M])*sizeof(float);
    malloc_and_copy_uvm((void **) &((*csr_d).values), csr_h.values, size);

    size = (csr_h.row_indices[csr_h.M])*sizeof(int);
    malloc_and_copy_uvm((void **) &((*csr_d).col_indices), csr_h.col_indices, size);

    size = (csr_h.M + 1)*sizeof(int);
    malloc_and_copy_uvm((void **) &((*csr_d).row_indices), csr_h.row_indices, size);

    (*csr_d).M = csr_h.M;
    (*csr_d).N = csr_h.N;
    (*csr_d).NNZ = csr_h.NNZ;
}

// void csr_free_uvm(SparseMatrixCSR csr, bool ongpu)
// {
//     FREE(csr.values, ongpu);
//     FREE(csr.row_indices, ongpu);
//     FREE(csr.col_indices, ongpu);
// }

void csr_free(SparseMatrixCSR csr, bool ongpu)
{
    FREE(csr.values, ongpu);
    FREE(csr.row_indices, ongpu);
    FREE(csr.col_indices, ongpu);
}

void ell_to_device(SparseMatrixELL ell_h, SparseMatrixELL * ell_d)
{
    int size;

    size = (ell_h.M * ell_h.K)*sizeof(int);
    malloc_and_copy((void **) &((*ell_d).col_indices), ell_h.col_indices, size);

    size = (ell_h.M * ell_h.K)*sizeof(float);
    malloc_and_copy((void **) &((*ell_d).values), ell_h.values, size);

    (*ell_d).M = ell_h.M;
    (*ell_d).N = ell_h.N;
    (*ell_d).K = ell_h.K;
}

void ell_free(SparseMatrixELL ell, bool ongpu)
{
    FREE(ell.col_indices, ongpu);
    FREE(ell.values, ongpu);
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

void set( bool * y, int M, bool v)
{
#pragma omp parallel for
    for (int i=0; i<M; ++i)
        y[i] = v;
}

void set( int * y, int M, int v)
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

void data_init_bfs( SparseMatrixCOO coo, int **prev, int ** level)
{
    // assert(("Adjacent Matrix should be square", coo.N == coo.M));

    if(coo.N != coo.M)
    {
        printf("Adjacent Matrix should be square");
        return;
    }

    *level = (int *) malloc (coo.N * sizeof(int));
    *prev = (int *) malloc (coo.N * sizeof(int));
    
    if ( !*level || !*prev ) exit (1);

    set(*level, coo.N, INT_MAX - 1);
    set(*prev, coo.N, -1);
}


// void data_init_bfs( SparseMatrixCOO coo, bool ** f, bool ** x, int ** c )
// {
//     // assert(("Adjacent Matrix should be square", coo.N == coo.M));

//     if(coo.N != coo.M)
//     {
//         printf("Adjacent Matrix should be square");
//         return;
//     }
    
//     *c = (int *) malloc (coo.N * sizeof(int));
//     *x = (bool *) malloc (coo.N * sizeof(bool));
//     *f = (bool *) malloc (coo.N * sizeof(bool));
    
//     if ( !*f || !*x || !*c) exit (1);

//     set(*f, coo.N, false);
//     set(*x, coo.N, false);
//     set(*c, coo.N, 0);
// }


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

bool same_bfs( int * y_exp, int * y, int M )
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

void print_res_bfs(const char * codeversion, double run_time, bool check, int * y, int * y_exp, int M)
{
   printf("%20s | %30.6lf", codeversion, run_time);
   if ( check )
       printf(" | %30s", same_bfs(y_exp, y, M)? "Y" : "N");
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
    csr->NNZ = coo.NNZ;

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

bool coo_to_ell(SparseMatrixCOO coo, SparseMatrixELL * ell)
{
     // TODO: write this code

    ell->M = coo.M;
    ell->N = coo.N;
    ell->K = 0;

    int * count = (int *) calloc ( coo.M, sizeof(int));
    if (!count) {
        fprintf(stderr, "out of space in transfering coo to ell\n");
        return false;
    }

    for(int i=0; i<coo.M; ++i)
        count[i] = 0;

    for(int i=0; i<coo.NNZ; ++i) {
        int r = coo.row_indices[i];
        count[r] += 1;
    }

    for(int i=0; i<coo.M; ++i) {
        if(ell->K < count[i]) {
            ell->K = count[i];
        }
        count[i] = 0;
    }

    ell->col_indices = (int *) malloc( ell->K * ell->M * sizeof(int) );
    ell->values = (float *) malloc( ell->K * ell->M * sizeof(float) );

    if (!ell->col_indices || !ell->values ) {
        fprintf(stderr, "out of space in transfering coo to ell\n");
        return false;
    }

    for(int i=0; i<ell->K * ell->M; ++i){
        ell->col_indices[i] = -1;
        ell->values[i] = 0;
    }


    for(int i=0; i<coo.NNZ; ++i) {
        int r = coo.row_indices[i];
        int c = coo.col_indices[i];
        float v = coo.values[i];
        ell->col_indices[ell->M * count[r] + r] = c;
        ell->values[ell->M * count[r] + r] = v;
        count[r] ++;
    }
    
    // printf("checkpoint\n");
    free(count);

// #ifdef DEBUG
//     if (ell->K * ell->M < coo.NNZ) { fprintf(stderr, "ell generated incorrectly\n"); exit(1); }
// #endif
    return true;

}

bool coo_to_hybrid(const SparseMatrixCOO coo, SparseMatrixELL * ell_hyb, SparseMatrixCOO * coo_hyb, int K)
{
     // TODO: write this code
    ell_hyb->M = coo.M;
    ell_hyb->N = coo.N;
    ell_hyb->K = K;

    coo_hyb->M = coo.M;
    coo_hyb->N = coo.N;


    int * count = (int *) calloc ( coo.M, sizeof(int));
    int count_NNZ = 0;
    if (!count) {
        fprintf(stderr, "out of space in transfering coo to hybrid\n");
        return false;
    }

    int min_col_to_K = coo.N;

    for(int i=0; i<coo.NNZ; ++i) {
        int r = coo.row_indices[i];
        int c = coo.col_indices[i];
        count[r]++;

        if(count[r] == K && c < min_col_to_K)
        {
            min_col_to_K = c;
        }
    }

// #ifdef DEBUG
//     printf("min_col_to_k: %d\n", min_col_to_K);
// #endif

    for(int i=0; i<coo.NNZ; ++i) {
        int c = coo.col_indices[i];

        if(c > min_col_to_K)
        {
            count_NNZ ++;
        }
    }

    coo_hyb->NNZ = count_NNZ;

    for(int i=0; i<coo.M; ++i) {
        count[i] = 0;
    }

    ell_hyb->col_indices = (int *) malloc( ell_hyb->K * ell_hyb->M * sizeof(int) );
    ell_hyb->values = (float *) malloc( ell_hyb->K * ell_hyb->M* sizeof(float) );

    coo_hyb->col_indices = (int *) malloc( coo_hyb->NNZ * sizeof(int) );
    coo_hyb->row_indices = (int *) malloc( coo_hyb->NNZ * sizeof(int) );
    coo_hyb->values = (float *) malloc( coo_hyb->NNZ * sizeof(float) );

    for(int i=0; i<ell_hyb->K * ell_hyb->M; ++i){
        ell_hyb->col_indices[i] = -1;
        ell_hyb->values[i] = 0;
    }

    int coo_hyb_ptr = 0;

    for(int i=0; i<coo.NNZ; ++i) {
        int r = coo.row_indices[i];
        int c = coo.col_indices[i];
        float v = coo.values[i];
        if(c <= min_col_to_K){
            ell_hyb->col_indices[ell_hyb->M * count[r] + r] = c;
            ell_hyb->values[ell_hyb->M * count[r] + r] = v;
            count[r] ++;
        }
        else{
            coo_hyb->values[coo_hyb_ptr] = v;
            coo_hyb->col_indices[coo_hyb_ptr] = c;
            coo_hyb->row_indices[coo_hyb_ptr] = r;
            coo_hyb_ptr ++;
        }
        
    }
    free(count);
    return true;
}
