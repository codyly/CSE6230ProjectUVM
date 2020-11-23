#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>
#include <algorithm>
#include <queue>
#include <vector>

#include "mtx_reader.h"
#include "utils.h"

const int BLK_SIZE = 1024;


void SpMBFS_CSR(const SparseMatrixCSR A, int* prev, int *level, bool *done)
{
    std::queue<int> q;
    q.push(100);
        
    *done = true;

    while(!q.empty()) {
        int id = q.front();
        q.pop();

        // Xa[id] = true;

        // printf("%d\n", id);
        
        int start = A.row_indices[id];
        int end = A.row_indices[id+1];

        for (int i = start; i < end; i++) 
        {
            int nid = A.col_indices[i];
            int old = level[nid];

            level[nid] = min(level[nid], level[id] + 1);

            if (nid < A.N && old == INT_MAX-1)
            {
                
                prev[nid] = id;
                q.push(nid);
            }

        }


    }
}

struct idnode{
    int id;
    float harm;
    float phi;
};

void HALO_I(SparseMatrixCSR* A, int* prev, int *level, bool *done, int K, int source)
{

    // SpMBFS_CSR(*A, prev, level, done);
    std::vector<struct idnode> ids;

    for(int i=0; i<A->N; i++){
        struct idnode idi;
        idi.id = i;
        if(i!=source){
            idi.harm = 0;
            idi.harm = idi.harm + 1.0 / level[i];
        }
        ids.push_back(idi);
    }

    ids[source].harm = INT_MAX;

    sort( ids.begin( ), ids.end( ), [ ]( const struct idnode& lhs, const struct idnode& rhs ){
       return lhs.harm > rhs.harm;
    });

    float* new_value = (float*) malloc(sizeof(float)*A->N);
    int* new_row = (int*) malloc(sizeof(int)*A->N);
    int* new_col = (int*) malloc(sizeof(int)*A->NNZ);

    int start = 0;
    for(int i=0; i<A->N; i++){
        int id = ids[i].id;
        new_value[i] = A->values[id];
        new_row[i] = start;
        int num_neighbor = A->row_indices[id+1] - A->row_indices[id];
        memcpy(new_col + start, A->col_indices + A->row_indices[id], sizeof(int) * num_neighbor);
        start += num_neighbor;
    }

    memcpy(A->row_indices, new_row, sizeof(int)*A->N);
    memcpy(A->col_indices, new_col, sizeof(int)*A->NNZ);
    memcpy(A->values, new_value, sizeof(float)*A->N);


    free(new_value);
    free(new_col);
    free(new_row);


}



void HALO_II(SparseMatrixCSR* A, int* prev, int *level, bool *done, int K, int source)
{

    // SpMBFS_CSR(*A, prev, level, done);
    std::vector<struct idnode> ids;

    for(int i=0; i<A->N; i++){
        struct idnode idi;
        idi.id = i;
        if(i!=source){
            idi.harm = 0;
            idi.harm = idi.harm + 1.0 / level[i];
        }
        idi.phi = -1;
        ids.push_back(idi);
    }

    ids[source].harm = INT_MAX;

    sort( ids.begin( ), ids.end( ), [ ]( const struct idnode& lhs, const struct idnode& rhs ){
       return lhs.harm > rhs.harm;
    });

    int* map = (int*) malloc(sizeof(int)*A->N);
    for(int i=0; i<A->N; i++)
        map[ids[i].id] = i;

    int count = 0;

    for(int i=0; i<A->N; i++)
    {
        if(ids[i].phi == -1)
        {
            ids[i].phi = count ++;
        }
        int id = ids[i].id;
        for(int j=A->row_indices[id]; j<A->row_indices[id+1]; j++)
        {
            int new_id = map[A->col_indices[j]];
            if(ids[new_id].phi == -1)
            {
                ids[new_id].phi = count ++;
            }
        }
    }

    sort( ids.begin( ), ids.end( ), [ ]( const struct idnode& lhs, const struct idnode& rhs ){
        return lhs.phi < rhs.phi;
    });

     
    float* new_value = (float*) malloc(sizeof(float)*A->N);
    int* new_row = (int*) malloc(sizeof(int)*A->N);
    int* new_col = (int*) malloc(sizeof(int)*A->NNZ);

    int start = 0;
    for(int i=0; i<A->N; i++){
        int id = ids[i].id;
        new_value[i] = A->values[id];
        new_row[i] = start;
        int num_neighbor = A->row_indices[id+1] - A->row_indices[id];
        memcpy(new_col + start, A->col_indices + A->row_indices[id], sizeof(int) * num_neighbor);
        start += num_neighbor;
    }

    memcpy(A->row_indices, new_row, sizeof(int)*A->N);
    memcpy(A->col_indices, new_col, sizeof(int)*A->NNZ);
    memcpy(A->values, new_value, sizeof(float)*A->N);

    free(new_value);
    free(new_col);
    free(new_row);
    free(map);
}

__global__ 
void SpMBFS_CSR_kernel(const SparseMatrixCSR A, int* prev, int *level, int* curr, bool *done)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < A.M && level[id] == *curr)
	{
		// printf("%d ", A.NNZ); //This printf gives the order of vertices in BFS	
		// Fa[id] = false;
        // Xa[id] = true;
		// __syncthreads(); 
		int start = A.row_indices[id];
		int end = A.row_indices[id+1];
		for (int i = start; i < end; i++) 
		{
            int nid = A.col_indices[i];
            // int old_val = min(Ca[nid], Ca[id]+1);
            // if (atomicAnd(Xa[nid], 1) == false)
            int old_val = atomicMin(level + nid, *curr+1);
            if(old_val == INT_MAX - 1)
			{
                // Ca[nid] = old_val;
                // atomicMin(Ca + nid, Ca[id]+1);
                // Fa[nid] = true;
                prev[nid] = id;
                *done = false;
            }
		}

	}

}


void csrbfs(const SparseMatrixCOO coo, 
            int source, 
            int *prev, int *level, 
            int * level_exp, 
            int * level_back, 
            bool ongpu, 
            bool check, 
            int opt)
{
    SparseMatrixCSR csr;
    if (!coo_to_csr(coo, &csr)) return;

    float run_time;
    bool done = false;

    if(opt % 3 == 1){
        HALO_I(&csr, NULL, level_back, &done, 1, source);
    } else if( opt % 3 == 2 ){
        HALO_II(&csr, NULL, level_back, &done, 1, source);
    }
    
    if ( !ongpu ) {
        double start_time = omp_get_wtime();
        SpMBFS_CSR(csr, prev, level, &done);
        run_time = omp_get_wtime() - start_time;
    } else if (opt < 3) {
        int *level_d, * prev_d;
        bool * done_d;
        int *curr_d;
        int curr = -1;

        malloc_and_copy( (void **) &prev_d, (void *)prev, coo.N*sizeof(int));
        malloc_and_copy( (void **) &level_d, (void *)level, coo.N*sizeof(int));
        malloc_and_copy( (void  **) &curr_d, (void *)&curr, sizeof(int));
        malloc_and_copy( (void  **) &done_d, (void *)&done, sizeof(bool));

        SparseMatrixCSR csr_d;
        csr_to_device(csr, &csr_d);

        int coarse = 1;
        int bs = BLK_SIZE;
        int cgs=(coo.N-1)/(bs)/coarse+1;

        CUDA_TIMER_START;

        do{
            curr += 1;
            done = true;
            cudaMemcpy(curr_d, &curr, sizeof(int), cudaMemcpyHostToDevice);
    		cudaMemcpy(done_d, &done, sizeof(bool), cudaMemcpyHostToDevice);
            SpMBFS_CSR_kernel<<<cgs, bs>>>(csr_d, prev_d, level_d, curr_d, done_d);
            cudaMemcpy(&done, done_d , sizeof(bool), cudaMemcpyDeviceToHost);
        }while(!done);      

        cudaMemcpy(level, level_d, coo.N*sizeof(int), cudaMemcpyDeviceToHost);

        CUDA_TIMER_END(run_time);

        CUDA_CHK(cudaGetLastError());

        csr_free(csr_d, true);
        cudaFree(level_d);
        cudaFree(prev_d);
        cudaFree(curr_d);
        cudaFree(done_d);
    }
    else if (opt < 6) {

        int *prev_d, *level_d;
        cudaMallocManaged(&prev_d, coo.N*sizeof(int));
        cudaMallocManaged(&level_d, coo.N*sizeof(int));
        memcpy(level_d, level, coo.N*sizeof(int));  
        memcpy(prev_d, prev, coo.N*sizeof(int));  


        int *curr_d;
        bool *done_d;
        cudaMallocManaged(&curr_d, sizeof(int));
        cudaMallocManaged(&done_d, sizeof(bool));
        *curr_d = -1;
        *done_d = false;

        SparseMatrixCSR* csr_d;
        cudaMallocManaged(&csr_d, sizeof(SparseMatrixCSR));
        csr_to_device_uvm(csr, csr_d);

        cudaDeviceSynchronize();

        int coarse = 1;
        int bs = BLK_SIZE;
        int cgs=(coo.N-1)/(bs)/coarse+1;

        CUDA_TIMER_START;

        do{
            *curr_d = *curr_d + 1;
            *done_d = true;
            SpMBFS_CSR_kernel<<<cgs, bs>>>(*csr_d, prev_d, level_d, curr_d, done_d);
            cudaDeviceSynchronize();
        }while(!(*done_d));   

        cudaDeviceSynchronize();
        
        CUDA_TIMER_END(run_time);

        memcpy(level, level_d, coo.N*sizeof(int));     
        
        // printf("%d\n", csr_d->NNZ);

        CUDA_CHK(cudaGetLastError());

        csr_free(*csr_d, true);
        cudaFree(level_d);
        cudaFree(prev_d);
        cudaFree(curr_d);
        cudaFree(done_d);
        cudaFree(csr_d);
    }
    // if(check)
    //     for(int i=0; i<coo.N; ++i)
    //     {
    //         if(level[i] != level_exp[i])
    //         {
    //             printf("%d %d %d\n", i, level[i], level_exp[i]);
    //         }
    //     }

    const char* tag;

    if(!ongpu){
        tag = "bfs_on_host";
    }
    else{
        switch(opt){
            case 0:{
                tag = "bfs_on_device";
                break;
            }
            case 1:{
                tag = "bfs_on_device_halo_1";
                break;
            }
            case 2:{
                tag = "bfs_on_device_halo_2";
                break;
            }
            case 3:{
                tag = "bfs_uvm";
                break;
            }
            case 4:{
                tag = "bfs_uvm_halo_1";
                break;
            }
            case 5:{
                tag = "bfs_uvm_halo_2";
                break;
            }
            default:
            {
                tag = "bfs";
                break;
            }
        }
    }
    csr_free(csr, false);
    print_res_bfs(tag, run_time, check, level, level_exp, coo.N);
}



void run(const SparseMatrixCOO coo, int  SOURCE, int *prev, int* level, bool ongpu, int K, bool check)
{

    level[SOURCE] = 0;
    prev[SOURCE] = -1;

    printf("%20s | %17s%13s", "code version", ongpu ? "gpu " : "cpu ", "run time(sec)");
    if (check)
        printf(" | %30s", "correctness");
    printf("\n");

    int * prev_exp = NULL;
    int * level_exp = NULL;
    int * prev_exp_back = NULL;
    int * level_exp_back  = NULL;
    if (check) {
        prev_exp = (int *) malloc (coo.M * sizeof(int));
        level_exp = (int *) malloc (coo.M * sizeof(int));
        prev_exp_back = (int *) malloc (coo.M * sizeof(int));
        level_exp_back = (int *) malloc (coo.M * sizeof(int));
        memcpy(prev_exp, prev, coo.M * sizeof(int));
        memcpy(level_exp, level, coo.M * sizeof(int));
        // TODO: csrbfs-cpu
        csrbfs(coo, SOURCE, prev_exp, level_exp, NULL, NULL, ongpu, false, 0);
    }
    if(!ongpu)
    {
        csrbfs(coo, SOURCE, prev, level, level_exp, NULL, ongpu, check, 0);
    }
    else{
        csrbfs(coo, SOURCE, prev, level, level_exp, NULL, ongpu, check, 3);

        memcpy(prev_exp_back, prev_exp, coo.M * sizeof(int));
        memcpy(level_exp_back, level_exp, coo.M * sizeof(int));


        set(level, coo.N, INT_MAX - 1);
        set(prev, coo.N, -1);
        level[SOURCE] = 0;
        prev[SOURCE] = -1;
        if (check) {
            memcpy(prev_exp, prev, coo.M * sizeof(int));
            memcpy(level_exp, level, coo.M * sizeof(int));
            // TODO: csrbfs-cpu
            csrbfs(coo, SOURCE, prev_exp, level_exp, NULL, level_exp_back, ongpu, false, 1);
        }
        csrbfs(coo, SOURCE, prev, level, level_exp, level_exp_back, ongpu, check, 4);

        set(level, coo.N, INT_MAX - 1);
        set(prev, coo.N, -1);
        level[SOURCE] = 0;
        prev[SOURCE] = -1;
        if (check) {
            memcpy(prev_exp, prev, coo.M * sizeof(int));
            memcpy(level_exp, level, coo.M * sizeof(int));
            // TODO: csrbfs-cpu
            csrbfs(coo, SOURCE, prev_exp, level_exp, NULL,  level_exp_back, ongpu, false, 2);
        }
        csrbfs(coo, SOURCE, prev, level, level_exp, level_exp_back, ongpu, check, 5);
    }
    
    if (!level_exp_back) free(level_exp_back);
    if (!prev_exp_back) free(prev_exp_back);
    if (!level_exp) free(level_exp);
    if (!prev_exp) free(prev_exp);
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
    const int  source = (argc > 3) ? atoi(argv[3]) : 100;
    const int  K = (argc > 4) ? atoi(argv[4]) : 8;
    const char * check = (argc > 5) ? argv[5] : "check";
    // reading mtx format file to coo matrix
    SparseMatrixCOO coo = read_edgelist(fname);
    int *prev, *level;
    data_init_bfs(coo, &prev, &level);
               
    run(coo, source, prev, level, strcmp(platform, "gpu") == 0, K, strcmp(check, "check") == 0);

    coo_free(coo, false);
    free(prev);
    free(level);

    return 0;
}
