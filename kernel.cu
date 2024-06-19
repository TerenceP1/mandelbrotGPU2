#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma comment (lib,"cuda.lib")

#include <stdio.h>
#include <fstream>
#include <iostream>
using namespace std;

typedef void* Decimal;

// High decimal presision code:

__device__ void CreateDecimal(Decimal* d, int decp) {
    /* Decimal layout:
    * First item of array int stores decimal precision
    * Treat as unsigned int array
    * First item to left of decimal point (signed)
    * Rest to the right of decimal point
    */
    cudaMalloc(d, (sizeof(int) * (decp + 2)));
    (*((unsigned int**)d))[0] = decp;
}

__device__ void DestroyDecimal(Decimal* d) {
    cudaFree(d);
}

__device__ void InitDecimal(Decimal* d) {
    int dp = (*((int**)d))[0];
    for (int i = 1;i < dp + 2;i++) {
        (*((int**)d))[i] = 0;
    }
}

__device__ void AddDecimal(Decimal* a, Decimal* b, Decimal* c) {
    // Adds a and b and stores result in c
    unsigned long pass = 0;
    int decp = ((unsigned int*)a)[0];
    for (int i = decp + 1;i > 0;i--) {
        bool pas = (pass && 0x100000000l) >> 8;
        pass = (unsigned long)(((unsigned int*)a)[i]) + (unsigned long)(((unsigned int*)b)[i]);
        if (pas) {
            pass++;
        }
        ((unsigned int*)c)[i] = (unsigned int)pass;
    }
}

__device__ void NegDecimal(Decimal* a, Decimal* b) {
    // Negates a and stores result in b
    int decp = ((unsigned int*)a)[0];
}

__device__ void SubDecimal(Decimal* a, Decimal* b, Decimal* c) {
    // c=a-b
    bool pass;
    unsigned int subR;
    int decp = ((unsigned int*)a)[0];
    for (int i = decp;i > 1;i--) {
        subR = (((unsigned int*)a)[i]) - (((unsigned int*)b)[i]);
        if ((((unsigned int*)a)[i]) < (((unsigned int*)b)[i])) {
            pass = true;
        }
        if (pass) {
            subR--;
        }
        ((unsigned int*)c)[i] = (unsigned int)pass;
    }
    subR = (((int*)a)[1]) - (((int*)b)[1]);
    if (pass) {
        subR--;
    }
    ((int*)c)[1] = (int)subR;
}

__global__ void calcRow(CUdeviceptr arr) {
    
}

int main()
{
    cuInit(0);
    cout << "start!\n";
    ifstream in("in.txt");
    double re, im, zoom;
    in >> re >> im >> zoom;
    const int xSz = 4096, ySz = 2160;
    CUdeviceptr arr=0;
    int res = cuMemAlloc(&arr, xSz * ySz * 3); // In case not enough memory
    if (res != cudaSuccess) {
        cerr << "ERROR!"<<res;
        return res;
    }
    cout << "Malloc\n";
    calcRow <<<1, ySz >>> (arr);
    cudaDeviceSynchronize();
    cuMemFree(arr);
    cout << "Done!";
    return 0;
}