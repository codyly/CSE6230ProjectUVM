#include "kt.h"

// int main(){

//     int IA[] = {0, 3, 5, 8, 9, 12};
//     int JA[] = {1, 2, 4, 0, 4, 0, 3, 4, 2, 0, 1, 2};

//     // int IA[] = {0, 3, 4, 5, 7, 7, 7};
//     // int JA[] = {1, 2, 4, 4, 3, 4};


//     // float S[] =  {1, 1, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1};

//     // float M[] =  {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1};
//     // float M[] =  {1, 1, 1, 1, 0, 1};
//     float M[] =     {0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0};

//     eagerKTruss(IA, JA, M, 5, 4);

//     for(int i=0; i<12; ++i)
//     {
//         printf("%u ", M[i]);
//     }
//     printf("\n");
//     return 0;

// }


bool serialKT(int *IA, int *JA, float *M, int NUM_VERTICES, int K){

    bool notEqual = false;

    for(int i=0; i <NUM_VERTICES; ++i){
        int a12_start = *(IA+i);
        int a12_end = *(IA+i+1);
        int *JAL = JA + a12_start;

        for(int l = a12_start; *JAL !=0 && l!= a12_end; ++l){

            int A22_start = *(IA + *(JAL));
            int A22_end = *(IA + *(JAL) + 1);
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
                    ++(*MK);
                }

                ML += update_val;

                int tmp = *JAK;
                int advanceK = (tmp <= Jaj_val);
                int advanceJ = (Jaj_val <= tmp);

                JAK += advanceK;
                MK += advanceK;
                JAJ += advanceJ;

                if( update_val ){
                    ++(*MJ);
                }
                MJ += advanceJ;
            }
            *(M+l) += ML;
        }
    }

    for(int n = 0; n<NUM_VERTICES; ++n){
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
            notEqual = 1;
            *Jk = 0;
        }
    }
    return notEqual;
}


bool openmpKT(int *IA, int *JA, float *M, int NUM_VERTICES, int K){

    bool notEqual = false;

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, CHUNK)
    for(int i=0; i <NUM_VERTICES; ++i){
        int a12_start = *(IA+i);
        int a12_end = *(IA+i+1);
        register int *JAL = JA + a12_start;

        for(int l = a12_start; *JAL !=0 && l!= a12_end; ++l){

            int A22_start = *(IA + *(JAL));
            int A22_end = *(IA + *(JAL) + 1);
            JAL ++;

            float ML = 0;
            int *JAK = JAL;
            int *JAJ = JA + A22_start;
            float *MJ = M + A22_start;
            float *MK = M + l + 1;

            while(*JAK != 0 && *JAJ != 0 && JAK != JA + a12_end && JAJ !=JA + A22_end){
                register int Jaj_val = *JAJ;
                register int update_val = (Jaj_val == *JAK);

                if (update_val){
                    #pragma omp atomic
                    ++(*MK);
                }

                ML += update_val;

                int tmp = *JAK;
                int advanceK = (tmp <= Jaj_val);
                int advanceJ = (Jaj_val <= tmp);

                JAK += advanceK;
                MK += advanceK;
                JAJ += advanceJ;

                if( update_val ){
                    #pragma omp atomic
                    ++(*MJ);
                }
                MJ += advanceJ;
            }
            #pragma omp atomic
            *(M+l) += ML;
        }
    }

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, CHUNK)
    for(int n = 0; n<NUM_VERTICES; ++n){
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
            notEqual = 1;
            *Jk = 0;
        }
    }

    return notEqual;
}
