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
// Note: Decimals are always signed
// Note: Impementation utilizes two's complement

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
        bool pas = (pass & 0x100000000l) >> 32;
        pass = (unsigned long long)(((unsigned int*)a)[i]) + (unsigned long)(((unsigned int*)b)[i]);
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
        bool pas = (pass & 0x100000000l) >> 32;
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
    for (int i = 0;i <= decp;i++) {
        for (int j = 0;j <= decp;j++) {
            unsigned long long res = ((unsigned long long)(ai[i + 1])) * ((unsigned long long)(bi[i + 1]));
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

__device__ void Dv2Decimal(Decimal* a) {
    // Divide a by 2 (via bitshifting)
    unsigned int* ai = *((unsigned int**)a);
    unsigned int* decp = ai[0];
    for (int i = decp + 1;i > 0;i--) {
        ai[i] = (ai[i] >> 1) | (ai[i - 1] << 31);
    }
    ai[0] >>= 1;
}

__device__ int CmpDecimal(Decimal* a, Decimal* b) {
    // Return 1 if a>b, 0 if a==b, and -1 if a<b
    unsigned int* ai = *((unsigned int**)a);
    unsigned int* bi = *((unsigned int**)b);
    unsigned int decp = ai[0];
    int af = ai[1];
    int bf = bi[1];
    if (af > bf) {
        return 1;
    }
    if (af < bf) {
        return -1;
    }
    for (int i = 2;i < decp + 2;i++) {
        if (ai[i] > bi[i]) {
            return 1;
        }
        if (ai[i] < bi[i]) {
            return -1;
        }
    }
    return 0;
}

__device__ void RecDecimal(unsigned int a, Decimal* b) {
    // Take reciprical of a and store it in b
    // Consumes auxillary space
    unsigned int* bi = *((unsigned int**)b);
    unsigned int decp = bi[0];
    Decimal tmp;
    CreateDecimal(&tmp, decp);
    InitDecimal(&tmp);
    Decimal inc;
    CreateDecimal(&inc, decp);
    InitDecimal(&inc);
    ((unsigned int*)inc)[1] = a;
    Dv2Decimal(inc);
    Decimal one;
    CreateDecimal(&one, decp);
    InitDecimal(&one);
    ((unsigned int*)one)[1] = 1;
    bi[1] = 0;
    for (int i = 2;i < decp + 2;i++) {
        for (int j = 31;j >= 0;j--) {
            unsigned int pl = (1u << j);
            AddDecimal(&tmp, &inc, &tmp);
            if (CmpDecimal(&tmp, &one) == 1) {
                SubDecimal(&tmp, &inc, &tmp);
                bi[i] &= ~pl;
            }
            else {
                bi[i] |= pl;
            }
            Dv2Decimal(&tmp);
        }
    }
}

__global__ void calcRow(CUdeviceptr arr, char* re, char* im, int reLen, int imLen, int prec) {
    // Get 0.1 in binary to convert base 10 to binary
    Decimal tenth;
    CreateDecimal(&tenth, prec);
    InitDecimal(&tenth);
    ((unsigned int*)tenth)[1] = 0x00000000U;
    ((unsigned int*)tenth)[2] = 0x19999999U;
    for (int i = 3;i < prec + 2;i++) {
        ((unsigned int*)tenth)[i] = 0x99999999U;
    }
    // Convert the re from string to binary decimal
    Decimal dRe;
    CreateDecimal(&dRe, prec);
    InitDecimal(&dRe);
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
    //calcRow <<<1, ySz >>> (arr);
    cudaDeviceSynchronize();
    cuMemFree(arr);
    cout << "Done!";
    return 0;
}
