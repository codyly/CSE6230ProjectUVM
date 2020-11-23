#ifndef _MTX_READER
#define _MTX_READER

#include "utils.h"
#include "mmio.h"

SparseMatrixCOO read_edgelist(const char * fpath);

#endif
