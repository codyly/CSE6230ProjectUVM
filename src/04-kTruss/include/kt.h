#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <algorithm>
#include <queue>
#include <vector>

#include<time.h>
#include<sys/time.h>
#include<string.h>
#include<stdlib.h>
#include <stdint.h>
#define NUM_THREADS 4
#define CHUNK 1

bool openmpKT(int *IA, int *JA, float *M, int NUM_VERTICES, int K);
bool serialKT(int *IA, int *JA, float *M, int NUM_VERTICES, int K);
