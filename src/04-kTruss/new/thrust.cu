#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <functional>
#include <iostream>
#include <bits/stdc++.h>

#include <climits>
#include<cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/functional.h>



#include <chrono>  //this is from https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c

using namespace std;
using namespace std::chrono;


__global__ void line89 (int *d_dstp, int *d_srcp, int n){
  int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index >=0 && index < n)
    d_dstp [ d_srcp[index] ] =index;

}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}



int main(int argc, char *argv[])
{
    ifstream fin;
    ofstream fout;
    string infile, outfile;
    char* s=argv[1];
    char* t=argv[2];
    //cout<<"inside readGraph"<<endl;
  // infile ="../../../input/"      + name + ".mmio" ; //  ../../../input/amazon0302_adj.mmio
  // outfile="../../output/serial/" + name + ".txt"  ; //  dataset+"-out.txt";
  infile =s;
  outfile = t;
  fin.open(infile.c_str());    // opening the input file
  fout.open(outfile.c_str());  // opening the output file

  string temp;
  getline(fin,temp); // readint the description line 1
  getline(fin,temp); // reading the description line 2

  //int temp_e;          // temperory edge because edge weight is useless
  int u1,v1,n,m;             // the v1,v2 of edges

  fin >> n >> n >> m ;       // reading the MxN graph and edges
  //cout<< n<<" "<< m<<endl;

  //int hist[n], srcp[n], dstp[n];

  thrust::host_vector<int> hist1(n);
  thrust::device_vector<int> srcp(n);
  thrust::host_vector<int> dstp1(n);

  //cout<<"Allocated"<<endl;
  int i,temp_int;


  thrust::fill(hist1.begin(), hist1.begin() + n, 0);
  cout<<"Filled"<<endl;

  for(i=0;i<m;i++)
  {
      fin>>u1>>v1>>temp_int;
      hist1[u1]++;

      hist1[v1]++;

  }
  high_resolution_clock::time_point t0 = high_resolution_clock::now();
  //cout<<"Hist done"<<endl;
  thrust::device_vector<int> hist(hist1);



  // for( i = 0; i < hist.size(); i++)
  //       cout << "hist[" << i << "] = " << hist[i] << endl;

    thrust::sequence(srcp.begin(), srcp.end(),0);

    // for(i = 0; i < srcp.size(); i++)
    // cout << "srcp[" << i << "] = " << srcp[i] << endl;

    thrust::sort_by_key(hist.begin(), hist.begin() + n, srcp.begin(),thrust::greater<int>());

    // for(i = 0; i < srcp.size(); i++)
    // cout << "srcp[" << i << "] = " << srcp[i] << endl;

    // for(i=0;i<n;i++)
    // {
    //     dstp[srcp[i]]=i;
    // }
    thrust::device_vector<int> dstp(dstp1);
    int *d_dstp = thrust::raw_pointer_cast(&dstp[0]);
    int *d_srcp = thrust::raw_pointer_cast(&srcp[0]);
    line89<<< n/1024 + 1 , 1024>>>(d_dstp, d_srcp, n);


    thrust::device_vector<int> u(m);
    thrust::device_vector<int> v(m);
    //cout<<"u,v allocated"<<endl;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    fin.close();
    fin.open(infile.c_str());    // opening the input file
    // fout.open(outfile.c_str());  // opening the output file

    getline(fin,temp); // readint the description line 1
    getline(fin,temp); // reading the description line 2

    //int temp_e;          // temperory edge because edge weight is useless
    //int u1,v1,n,m;             // the v1,v2 of edges

    fin >> n >> n >> m ;
    //set< pair<int,int> > st;
    int u_,v_;


    thrust::host_vector< thrust::pair<int,int> > d_E(m);
    //cout<<"de allocated"<<endl;
    dstp1 = dstp;
    for(i=0;i<m;i++)
    {
        fin>>u1>>v1>>temp_int;
        u_ = dstp1[u1];
        v_ = dstp1[v1];
        d_E[i] = ( u_>v_ ? thrust::make_pair(u_,v_) : thrust::make_pair(v_,u_));//  make_pair(max(u_,v_),max(u_,v_)));
    }

    // std::set<int>::iterator it;
    // for(it = st.begin(); it != st.end(); it++){

    // }
    //thrust::host_vector< thrust::pair<int,int> > d_E (h_E.begin(),h_E.end());

    thrust::sort(d_E.begin(), d_E.end());
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    bool flag1=false;
    if(d_E[0].first == d_E[1].first && d_E[0].second == d_E[1].second)
      flag1 = true;
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    auto new_E = thrust::unique(d_E.begin(),d_E.end());

    //https://stackoverflow.com/questions/49856303/thrustunique-on-float3-tuple

    auto a = duration_cast<microseconds>( t1 - t0 ).count();
    auto b = duration_cast<microseconds>( t3 - t2 ).count();
    auto c = duration_cast<microseconds>( t3-t2 + t1-t0 ).count();

    //cout<<"a "<<a<<" b "<<b<<" c "<<c<<endl;

    ofstream fout1;
    string cmpt = "compaction_times";
    fout1.open(cmpt.c_str(),std::ios_base::app);
    fout1<<argv[3]<<" "<<a<<" "<<b<<" "<<c<<endl;
    fout1.close();

    // fout<<"hi"<<endl;
    // fout<<"hello"<<endl;
    if(flag1 == true)
    {
      fout<<n<<" "<<m/2<<endl;
      for(int i=0 ; i<(d_E.size()/2) ; i++)
        fout<<d_E[i].first << " "<<d_E[i].second<<endl;
    }

    else
    {
      fout<<n<<" "<<m<<endl;
      for(int i=0 ; i<(d_E.size()) ; i++)
        fout<<d_E[i].first << " "<<d_E[i].second<<endl;
    }

    fout.close();



return 0;


}
