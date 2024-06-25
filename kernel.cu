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
    unsigned long long pass = 0;
    int decp = ((unsigned int*)a)[0];
    for (int i = decp + 1;i > 0;i--) {
        bool pas = (pass && 0x100000000l) >> 32;
        pass = (unsigned long long)(((unsigned int*)a)[i]) + (unsigned long long)(((unsigned int*)b)[i]);
        if (pas) {
            pass++;
        }
        ((unsigned int*)c)[i] = (unsigned int)pass;
    }
}

__device__ void NegDecimal(Decimal* a, Decimal* b) {
    // Negates a and stores result in b
    int decp = ((unsigned int*)a)[0];
    for (int i = 1;i < decp + 2;i++) {
        ((unsigned int*)b)[i] = ~(((unsigned int*)a)[i]);
    }
    for (int i = decp + 1;i > 0;i--) {
        ((unsigned int*)b)[i]++;
        if ((((unsigned int*)b)[i]) != 0) {
            break;
        }
    }
}

__device__ void SubDecimal(Decimal* a, Decimal* b, Decimal* c) {
    // c=a-b
    NegDecimal(b, b);
    unsigned long long pass = 0;
    int decp = ((unsigned int*)a)[0];
    for (int i = decp + 1;i > 0;i--) {
        bool pas = (pass && 0x100000000l) >> 32;
        pass = (unsigned long long)(((unsigned int*)a)[i]) + (unsigned long long)(((unsigned int*)b)[i]);
        if (pas) {
            pass++;
        }
        ((unsigned int*)c)[i] = (unsigned int)pass;
    }
    NegDecimal(b, b);
}

__device__ void MulDecimal(Decimal* a, Decimal* b, Decimal* c) {
    // c=a*b
    // Consumes auxillary space
    int decp = ((unsigned int*)a)[0];
    unsigned int* temp = new unsigned int[decp * 2 + 2];
    unsigned int* ai = *((unsigned int**)a);
    unsigned int* bi = *((unsigned int**)b);
    unsigned int* ci = *((unsigned int**)c);
    for (int i = 0;i <= decp * 2;i++) {
        temp[i] = 0;
    }
    for (int i = 0;i < decp;i++) {
        for (int j = 0;j < decp;j++) {
            unsigned long long res = ((unsigned long long)(ai[i])) + ((unsigned long long)(bi[i]));
            unsigned int gRes = (res << 32) >> 32;
            unsigned int lRes = res;
            // Add lres
            temp[i + j] += lRes;
            if (temp[i + j] < lRes) {
                for (int k = i + j;k < decp * 2 + 2;k++) {
                    temp[k]++;
                    if (temp[k] != 0) {
                        break;
                    }
                }
            }
            // Add gres
            temp[i + j + 1] += gRes;
            if (temp[i + j + 1] < lRes) {
                for (int k = i + j + 1;k < decp * 2 + 2;k++) {
                    temp[k]++;
                    if (temp[k] != 0) {
                        break;
                    }
                }
            }
        }
    }
    int ind = 1;
    for (int i = decp * 2;i > decp - 1;i--) {
        ci[i] = ind;
        ind++;
    }
    delete[] temp;
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
