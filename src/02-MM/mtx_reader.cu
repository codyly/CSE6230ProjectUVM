#include "mtx_reader.h"

SparseMatrixCOO read_edgelist(const char * fpath)
{
    FILE * f;
    f = fopen (fpath, "r");

    if (f == NULL) {
        perror("Error opening the matrix file");
        exit(1);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode)) {
        fprintf( stderr, "Could not process Matrix Market banner.\n" );
        exit(1);
    }
    if (mm_is_complex(matcode) || !mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        fprintf( stderr, "The .mtx file should contain a sparse real matrix.\n" );
        fprintf( stderr, "Market Market type: [%s] is not yet supported.\n", mm_typecode_to_str(matcode) );
        exit(1);
    }

    SparseMatrixCOO coo;
    int retcode;
    if ((retcode = mm_read_mtx_crd_size(f, &coo.M, &coo.N, &coo.NNZ)))
        exit(retcode);

    coo.row_indices = (int *) malloc ( coo.NNZ * sizeof(int) );
    coo.col_indices = (int *) malloc ( coo.NNZ * sizeof(int) );
    coo.values = (float *) malloc ( coo.NNZ * sizeof(float) );
    double * values;
    if (!coo.row_indices || !coo.col_indices || !coo.values) exit(1);
    if (!mm_is_pattern(matcode)) {
        values = (double *) malloc ( coo.NNZ * sizeof(double) );
        if (!values) exit(1);
    }

    if (mm_is_integer(matcode)) // mmio doesn't read integer matrices for some reason
        mm_set_real(&matcode);  // the code logic just doesn't accept integer matrices.
    if ((retcode = mm_read_mtx_crd_data(f, coo.M, coo.N, coo.NNZ, coo.row_indices, coo.col_indices, values, matcode)))
        exit(retcode);

#pragma omp parallel for
    for (int i = 0; i < coo.NNZ; ++i) {
        coo.row_indices[i] -= 1;
        coo.col_indices[i] -= 1;
        if (!mm_is_pattern(matcode))
            coo.values[i] = abs((float)values[i]); // absolte values to reduce relative error
        else
            coo.values[i] = (float)1;
    }

    fclose (f);
    return coo;
}
