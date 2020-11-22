// ##include <bits/stdc++.h>
//sorting in thrust https://stackoverflow.com/questions/23541503/sorting-arrays-of-structures-in-cuda/23645954
//Merge SearchSet http://on-demand.gputechconf.com/gtc/2013/presentations/S3414-Efficient-Merge-Search-Set-Operations.pdf
//thrust SET https://thrust.github.io/doc/group__set__operations.html
//maximum int = 2,147,483,647
//min int = -2,147,483,648

//scp -r /home/awd/work/coursework/DS295/project/pp_project/parallel/* dtanwar@turing-gpu.cds.iisc.ac.in:/home/dtanwar/Project/Parallel_Programming_Project/parallel

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <functional>
#include <iostream>

#include <fstream>


#include <climits>
#include<cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#define GS 1024
#define BS 1024

using namespace std;

typedef int uui;
typedef int var;


typedef struct{
  var V;  //no of vertices
  var E;  //no of edges
  var n; //no of non empty rows
  //var E;

  uui *colind;  //nonzeroes in each row (colind)
  uui *roff;  //startig offset of each row (rowoff)
  uui *rlen;  //length of each row
  uui *rows;  //indices of the non empty rows
} G;

__device__ int L;

__global__ void reset_bitmap(var *bitmap , var blockid, var V){
    int index  = threadIdx.x + blockDim.x*blockIdx.x;

    if(index >= V*blockid && index < V*(blockid+1)){
      atomicAnd(bitmap + index , 0);
    }
}


__global__ void find(var *data, var value, /*int min_idx,*/ var io_s, var rlen_i){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx >= io_s && idx<= rlen_i){
      if(data[idx] == value)
          atomicMin(&L, idx);
    }
}


//cudamalloc colind , roff , rows , rlen , bitmap , E , V ,n , supp,k ;
__global__ void getmajorsupport(uui* d_colind, uui* d_roff, uui* d_rows, uui* d_rlen, var* bitmap, var E, var V, var n, uui* d_supp, var K){

  __shared__ int broadcast[BS]; //TODO: 2d array! why?

  /* if(threadIdx.x==0 && blockIdx.x==1)
  // {
  //   printf("\nkernel    threadId.x=%d blockid.x=%d E=%d V=%d n=%d K=%d\n",threadIdx.x,blockIdx.x,E,V,n,K );
  //   // for (var i=0;i<(n) ;i++)
  //   // printf("%d ",d_rows[i]);
  //   // printf("\n");
  //   // printf("rows\n");
  //   __syncthreads();
  //  printf("colind\n");
  //  for(var i=0;i<E;i++)
  //  printf("%d ",d_colind[i] );
  //  printf("\n");
  //  __syncthreads();
  //  // printf("roff\n" );
  //  // for(var i=0;i<V+1;i++)
  //  // printf("%d ",d_roff[i]);
  //  // printf("\n");
  //  // printf("rlen\n");
  //  // for(var i=0;i<V;i++)
  //  // printf("%d ",d_rlen[i]);
  //  // printf("\n");
  //
  //
  //
  // } */



  var i,io_s,io_e,j,jo_s,jo_e,jo,io,c,count,k;
  for (var s = 0 ; s<n ; s+=gridDim.x){
    printf("Inside kernel\n");

    i = d_rows[s];
    io_s = d_roff[i];
    io_e = io_s + d_rlen[i];
    printf("Inside 4\n");
    for (io=io_s ; io < io_e ; io += blockDim.x){
      printf("Inside 5, io=%d", io);
      c = (io + threadIdx.x < io_e) ? d_colind[io + threadIdx.x] : -1;
      printf("Inside 6, c=%d  ", c);
      if (c > -1){
        atomicOr ((bitmap + (V * blockIdx.x) +c) , 1);
        broadcast[threadIdx.x] = c;
        printf("Inside 1\n");
      }
      __syncthreads();

      for (var t=0 ; t < blockDim.x ; t++){
        j = broadcast[t];
        printf("Inside 2\n");
        if (j == -1) break;
        count = 0;
        jo_s = d_roff[j];
        jo_e = jo_s + d_rlen[j];
        for(jo = jo_s + threadIdx.x ; jo < jo_e ; jo += blockDim.x){
          k = d_colind[jo];
          if(bitmap[V * blockIdx.x + k] == 1){
            count ++;
            atomicAdd(d_supp + jo , 1);
            // find<<<  E/1024 +1, 1024 >>>(d_colind, k , /*&L,*/ io_s, d_rlen[i]);
            for(L=0; L <= d_rlen[i] ; L++)
              if (d_colind[io_s + L] == k)
                break;
            printf("Before: i=%d, j=%d, k=%d, l=%d\n",i,j,k,L);
            atomicAdd(d_supp + io_s + L , 1);
            printf("After: i=%d, j=%d, k=%d, l=%d\n",i,j,k,L);
          }
        }
        atomicAdd(d_supp + io + t , count);
      }
    }
    // for(var x = V*blockIdx.x, i=0; i<V/*x< V*(blockIdx.x + 1)*/ ; i++,x++){
    //   atomicAnd(bitmap + x , 0);
    // }
    atomicAnd(bitmap + (V * blockIdx.x) + c , 0);
    //reset_bitmap<<< GS,BS >>> (bitmap, blockIdx.x,V);
  }
  __syncthreads();
}



// #include "read_graph.hpp"
ifstream fin;
ofstream fout;
string infile, outfile;
void readGraph(string filename, G *g){
  // cout<<"inside readGraph"<<endl;
  // infile ="../../../input/"      + name + ".mmio" ; //  ../../../input/amazon0302_adj.mmio
  // outfile="../../output/serial/" + name + ".txt"  ; //  dataset+"-out.txt";
  infile =filename;
  cout<<infile<<endl;

  fin.open(infile.c_str());    // opening the input file
  fout.open(outfile.c_str());  // opening the output file

  string temp;
  //getline(fin,temp); // readint the description line 1
  //getline(fin,temp); // reading the description line 2

  var temp_edge;          // temperory edge because edge weight is useless
  var u,v;             // the v1,v2 of edges

  fin >> g->V >> g->E ;       // reading the MxN graph and edges
  cout<< g->V<<" "<< g->E<<endl;      // just checking if it worked



/**************************allocating & initializing all flag[V] to false**********************************/
  // bool flag[g->V];                // tells whether particular row is empty or not
  // for (var i=0 ; i < g->V ; i++) {
  //     flag[i] = false;            // false means empty
  // }

  thrust::device_vector<bool> flag(g->V);
  thrust::fill(flag.begin(), flag.end(),0);


/**************************allocating & initializing all roff[V+1] to zero**********************************/
  g->roff = (uui *) malloc((g->V + 1) * sizeof(uui));
  assert(g->roff != NULL);
  for (var i=0 ; i < g->V+1 ; i++) {
      g->roff[i] = 0;
      //cout<<g->roff[i]<<" ";
  };
  //cout<<endl;

/**************************increase row offset and set flag for non empty row********************************/
	for (var i=0; i<g->E; ++i) {           //thrust
		fin >> u >> v;
    //cout<< u <<" "<<v <<endl;

    if(u > v)
      g->roff[u+1]++ , flag[u] = true;
    else if(u < v)
      g->roff[v+1]++ , flag[v] = true;

	}

/**********************populates indexs of nonzero rows rows[n] and initilizes n (no of non empty rows)******/
  g->rows = (uui *) malloc((g->V) * sizeof(uui));
  g->n = 0;


  var k =0;
  for (var i = 0; i<g->V; i++){
     if (flag[i] == true){
       g->n++;                            //thrust
       g->rows[k++] = i;                    //thrust
     }
   }

/**********************************************************************************************************/
//converting the roff from degree holder to actual usage.
  uui *temp_num_edges = (uui *) malloc((g->V + 1) * sizeof(uui));
  assert(temp_num_edges != NULL);

  temp_num_edges[0] = 0;
  //g->E= 0;
  k=0;
  for(var i = 0; i < g->V; i++) {
    //  g->E += g->roff[i];
      k += g->roff[i+1];
      temp_num_edges[i+1] =k;
  }

  for(var i= 0; i < g->V+1; i++)
    g->roff[i] = temp_num_edges[i];

/**********************************************************************************************************/
  g->rlen = (uui *) malloc((g->V) * sizeof(uui));
  k =0;

  for (var i = 0; i<g->V; i++){
    if (flag[i] == true)
      g->rlen[k] = g->roff[i+1] - g->roff[i];
    else
      g->rlen[k] = 0;
    k++;
  }

/**********************************************************************************************************/
  //Allocate space for colind
  g->colind = (uui *) malloc(g->E * sizeof(uui));
  assert(g->colind != NULL);

  fin.close();
  fin.open(infile.c_str());
  // getline(fin,temp); // readint the description line 1
  // getline(fin,temp); // reading the description line 2

  //Read V and E
  //fscanf(infp, "%ld %ld\n", &(g->n), &g->E);
  fin>>g->V>>g->E;
  for(var i = 0; i < g->E; i++)
    g->colind[i] = 0;
  //Read the edges
  // while( fscanf(infp, "%u %u\n", &u, &v) != EOF ) {
  for(var i=0 ; i<g->E ; i++){


    fin>>u>>v;
    if(u>v){
      g->colind[ temp_num_edges[u]  ] = v;
      temp_num_edges[u]++;
    }
    else if (u<v){
      g->colind[ temp_num_edges[v] ] = u;
      temp_num_edges[v]++;
    }


  }
  fin.close();
  printf("readGraph E=%d V=%d n=%d \n",g->E,g->V,g->n );
cout<<"Read the graph"<<endl;
/**********************************************************************************************************/

}



int main(int argc, char *argv[]){

  G g;
  // cout<<endl<<"checkpoint 1"<<endl;
  char* file_path=argv[1];
  readGraph(file_path,&g);
  printf("main E=%d V=%d n=%d\n",g.E,g.V,g.n );
  // cout<<"checkpoint 2"<<endl;

  // cout<<"rows"<<endl;
  // for (var i=0;i<(g.n) ;i++){
  //   cout<<g.rows[i]<<" ";
  // }
  // cout<<endl;
  // cout<<"colind"<<endl;
  // for (var i=0;i<(g.E) ;i++){
  //   cout<<g.colind[i]<<" ";
  // }
  // cout<<endl;
  // cout<<"roff"<<endl;
  // for (var i=0;i<(g.V+1) ;i++){
  //   cout<<g.roff[i]<<" ";
  // }
  // cout<<endl;
  // cout<<"rlen"<<endl;
  // for (var i=0;i<(g.V) ;i++){
  //   cout<<g.rlen[i]<<" ";
  // }
  // cout<<endl;

  // cudaMalloc( (void **) &d_rows, size );
  // cudaMalloc( (void **) &d_colind, size );
  // cudaMalloc( (void **) &d_roff, size );
  // cudaMalloc( (void **) &d_rlen, size );g->
  //
  // for (var i=0;i< g->n ;i++)
  //   rows[i] =



  thrust::device_vector<uui> d_rows ( g.rows , g.rows + g.n);
  thrust::device_vector<uui> d_colind (g.colind , g.colind+ g.E);
  thrust::device_vector<uui> d_roff (g.roff , g.roff + g.V + 1 );
  thrust::device_vector<uui> d_rlen (g.rlen , g.rlen + g.V);
  thrust::device_vector<var> bitmap (GS*g.V);
  thrust::fill(bitmap.begin(), bitmap.end(),0);
  thrust::device_vector<uui> support(g.E);
  thrust::fill(support.begin(), support.end(),0);


  uui *d_rows1 = thrust::raw_pointer_cast(&d_rows[0]);
  uui *d_colind1 = thrust::raw_pointer_cast(&d_colind[0]);
  uui *d_roff1 = thrust::raw_pointer_cast(&d_roff[0]);
  uui *d_rlen1 = thrust::raw_pointer_cast(&d_rlen[0]);
  uui *d_support1 = thrust::raw_pointer_cast(&support[0]);
  var *d_bitmap1 = thrust::raw_pointer_cast(&bitmap[0]);
  cudaEvent_t start, stop;
   float elapsedTime;
  var k=3;
  var call=1;
  while(call){
    if (k>3)
      break;
    if(k==3)
    {
      cout<<"Calling Kernel"<<endl;
      printf("E=%d V=%d n=%d K=%d\n",g.E,g.V,g.n,k );
      cudaEventCreate(&start);
  cudaEventRecord(start,0);
      getmajorsupport<<<GS,BS>>>(d_colind1,d_roff1,d_rows1,d_rlen1,d_bitmap1,g.V,g.E,g.n,d_support1,k);
      cudaEventCreate(&stop);
 cudaEventRecord(stop,0);
 cudaEventSynchronize(stop);
      cudaDeviceSynchronize();
      cout<<"Out of kernel"<<endl;
      call=0;
    }
  }
  // int i;
  //   cout << "support[" << 0 << "] = " << support[0] << endl;
  // for( i = 0; i < support.size(); i++)
  //   cout << "support[" << i << "] = " << support[i] << endl;
  //   return 0;
  cudaEventElapsedTime(&elapsedTime, start,stop);
 printf("Elapsed time : %f ms\n" ,elapsedTime);

}
