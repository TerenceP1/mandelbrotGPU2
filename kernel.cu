﻿#include "cuda.h"
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
    for (int i = 0;i < decp * 2 + 2;i++) {
        temp[i] = 0;
    }
    for (int i = 0;i < decp;i++) {
        for (int j = 0;j < decp;j++) {
            unsigned long long res = ((unsigned long long)(ai[i + 1])) * ((unsigned long long)(bi[i + 1]));
            unsigned int gRes = (res >> 32);
            unsigned int lRes = res;
            // Add lres
            temp[i + j] += lRes;
            if (temp[i + j] < lRes) {
                for (int k = i + j + 1;k < decp * 2 + 2;k++) {
                    temp[k]++;
                    if (temp[k] != 0) {
                        break;
                    }
                }
            }
            // Add gres
            temp[i + j + 1] += gRes;
            if (temp[i + j + 1] < gRes) {
                for (int k = i + j + 2;k < decp * 2 + 2;k++) {
                    temp[k]++;
                    if (temp[k] != 0) {
                        break;
                    }
                }
            }
        }
    }
    int ind = 1;
    for (int i = decp * 2;i >= decp;i--) {
        ci[ind] = temp[i];
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
# i n c l u d e   " c u d a . h "  
 # i n c l u d e   " c u d a _ r u n t i m e . h "  
 # i n c l u d e   " d e v i c e _ l a u n c h _ p a r a m e t e r s . h "  
 # p r a g m a   c o m m e n t   ( l i b , " c u d a . l i b " )  
  
 # i n c l u d e   < s t d i o . h >  
 # i n c l u d e   < f s t r e a m >  
 # i n c l u d e   < i o s t r e a m >  
 u s i n g   n a m e s p a c e   s t d ;  
  
 t y p e d e f   v o i d *   D e c i m a l ;  
  
 / /   H i g h   d e c i m a l   p r e s i s i o n   c o d e :  
 / /   N o t e :   D e c i m a l s   a r e   a l w a y s   s i g n e d  
 / /   N o t e :   I m p e m e n t a t i o n   u t i l i z e s   t w o ' s   c o m p l e m e n t  
  
 _ _ d e v i c e _ _   v o i d   C r e a t e D e c i m a l ( D e c i m a l *   d ,   i n t   d e c p )   {  
         / *   D e c i m a l   l a y o u t :  
         *   F i r s t   i t e m   o f   a r r a y   i n t   s t o r e s   d e c i m a l   p r e c i s i o n  
         *   T r e a t   a s   u n s i g n e d   i n t   a r r a y  
         *   F i r s t   i t e m   t o   l e f t   o f   d e c i m a l   p o i n t   ( s i g n e d )  
         *   R e s t   t o   t h e   r i g h t   o f   d e c i m a l   p o i n t  
         * /  
         c u d a M a l l o c ( d ,   ( s i z e o f ( i n t )   *   ( d e c p   +   2 ) ) ) ;  
         ( * ( ( u n s i g n e d   i n t * * ) d ) ) [ 0 ]   =   d e c p ;  
 }  
  
 _ _ d e v i c e _ _   v o i d   D e s t r o y D e c i m a l ( D e c i m a l *   d )   {  
         c u d a F r e e ( d ) ;  
 }  
  
 _ _ d e v i c e _ _   v o i d   I n i t D e c i m a l ( D e c i m a l *   d )   {  
         i n t   d p   =   ( * ( ( i n t * * ) d ) ) [ 0 ] ;  
         f o r   ( i n t   i   =   1 ; i   <   d p   +   2 ; i + + )   {  
                 ( * ( ( i n t * * ) d ) ) [ i ]   =   0 ;  
         }  
 }  
  
 _ _ d e v i c e _ _   v o i d   A d d D e c i m a l ( D e c i m a l *   a ,   D e c i m a l *   b ,   D e c i m a l *   c )   {  
         / /   A d d s   a   a n d   b   a n d   s t o r e s   r e s u l t   i n   c  
         u n s i g n e d   l o n g   l o n g   p a s s   =   0 ;  
         i n t   d e c p   =   ( ( u n s i g n e d   i n t * ) a ) [ 0 ] ;  
         f o r   ( i n t   i   =   d e c p   +   1 ; i   >   0 ; i - - )   {  
                 b o o l   p a s   =   ( p a s s   & &   0 x 1 0 0 0 0 0 0 0 0 l )   > >   8 ;  
                 p a s s   =   ( u n s i g n e d   l o n g   l o n g ) ( ( ( u n s i g n e d   i n t * ) a ) [ i ] )   +   ( u n s i g n e d   l o n g ) ( ( ( u n s i g n e d   i n t * ) b ) [ i ] ) ;  
                 i f   ( p a s )   {  
                         p a s s + + ;  
                 }  
                 ( ( u n s i g n e d   i n t * ) c ) [ i ]   =   ( u n s i g n e d   i n t ) p a s s ;  
         }  
 }  
  
 _ _ d e v i c e _ _   v o i d   N e g D e c i m a l ( D e c i m a l *   a ,   D e c i m a l *   b )   {  
         / /   N e g a t e s   a   a n d   s t o r e s   r e s u l t   i n   b  
         i n t   d e c p   =   ( ( u n s i g n e d   i n t * ) a ) [ 0 ] ;  
         f o r   ( i n t   i   =   1 ; i   <   d e c p   +   2 ; i + + )   {  
                 ( ( u n s i g n e d   i n t * ) b ) [ i ]   =   ~ ( ( ( u n s i g n e d   i n t * ) a ) [ i ] ) ;  
         }  
         f o r   ( i n t   i   =   d e c p   +   1 ; i   >   0 ; i - - )   {  
                 ( ( u n s i g n e d   i n t * ) b ) [ i ] + + ;  
                 i f   ( ( ( ( u n s i g n e d   i n t * ) b ) [ i ] )   ! =   0 )   {  
                         b r e a k ;  
                 }  
         }  
 }  
  
 _ _ d e v i c e _ _   v o i d   S u b D e c i m a l ( D e c i m a l *   a ,   D e c i m a l *   b ,   D e c i m a l *   c )   {  
         / /   c = a - b  
         N e g D e c i m a l ( b ,   b ) ;  
         u n s i g n e d   l o n g   l o n g   p a s s   =   0 ;  
         i n t   d e c p   =   ( ( u n s i g n e d   i n t * ) a ) [ 0 ] ;  
         f o r   ( i n t   i   =   d e c p   +   1 ; i   >   0 ; i - - )   {  
                 b o o l   p a s   =   ( p a s s   & &   0 x 1 0 0 0 0 0 0 0 0 l )   > >   3 2 ;  
                 p a s s   =   ( u n s i g n e d   l o n g   l o n g ) ( ( ( u n s i g n e d   i n t * ) a ) [ i ] )   +   ( u n s i g n e d   l o n g   l o n g ) ( ( ( u n s i g n e d   i n t * ) b ) [ i ] ) ;  
                 i f   ( p a s )   {  
                         p a s s + + ;  
                 }  
                 ( ( u n s i g n e d   i n t * ) c ) [ i ]   =   ( u n s i g n e d   i n t ) p a s s ;  
         }  
         N e g D e c i m a l ( b ,   b ) ;  
 }  
  
 _ _ d e v i c e _ _   v o i d   M u l D e c i m a l ( D e c i m a l *   a ,   D e c i m a l *   b ,   D e c i m a l *   c )   {  
         / /   c = a * b  
         / /   C o n s u m e s   a u x i l l a r y   s p a c e  
         i n t   d e c p   =   ( ( u n s i g n e d   i n t * ) a ) [ 0 ] ;  
         u n s i g n e d   i n t *   t e m p   =   n e w   u n s i g n e d   i n t [ d e c p   *   2   +   2 ] ;  
         u n s i g n e d   i n t *   a i   =   * ( ( u n s i g n e d   i n t * * ) a ) ;  
         u n s i g n e d   i n t *   b i   =   * ( ( u n s i g n e d   i n t * * ) b ) ;  
         u n s i g n e d   i n t *   c i   =   * ( ( u n s i g n e d   i n t * * ) c ) ;  
         f o r   ( i n t   i   =   0 ; i   < =   d e c p   *   2 ; i + + )   {  
                 t e m p [ i ]   =   0 ;  
         }  
         f o r   ( i n t   i   =   0 ; i   <   d e c p ; i + + )   {  
                 f o r   ( i n t   j   =   0 ; j   <   d e c p ; j + + )   {  
                         u n s i g n e d   l o n g   l o n g   r e s   =   ( ( u n s i g n e d   l o n g   l o n g ) ( a i [ i ] ) )   +   ( ( u n s i g n e d   l o n g   l o n g ) ( b i [ i ] ) ) ;  
                         u n s i g n e d   i n t   g R e s   =   ( r e s   < <   3 2 )   > >   3 2 ;  
                         u n s i g n e d   i n t   l R e s   =   r e s ;  
                         / /   A d d   l r e s  
                         t e m p [ i   +   j ]   + =   l R e s ;  
                         i f   ( t e m p [ i   +   j ]   <   l R e s )   {  
                                 f o r   ( i n t   k   =   i   +   j ; k   <   d e c p   *   2   +   2 ; k + + )   {  
                                         t e m p [ k ] + + ;  
                                         i f   ( t e m p [ k ]   ! =   0 )   {  
                                                 b r e a k ;  
                                         }  
                                 }  
                         }  
                         / /   A d d   g r e s  
                         t e m p [ i   +   j   +   1 ]   + =   g R e s ;  
                         i f   ( t e m p [ i   +   j   +   1 ]   <   l R e s )   {  
                                 f o r   ( i n t   k   =   i   +   j   +   1 ; k   <   d e c p   *   2   +   2 ; k + + )   {  
                                         t e m p [ k ] + + ;  
                                         i f   ( t e m p [ k ]   ! =   0 )   {  
                                                 b r e a k ;  
                                         }  
                                 }  
                         }  
                 }  
         }  
         i n t   i n d   =   1 ;  
         f o r   ( i n t   i   =   d e c p   *   2 ; i   >   d e c p   -   1 ; i - - )   {  
                 c i [ i ]   =   i n d ;  
                 i n d + + ;  
         }  
         d e l e t e [ ]   t e m p ;  
 }  
  
 _ _ g l o b a l _ _   v o i d   c a l c R o w ( C U d e v i c e p t r   a r r ,   c h a r *   r e ,   c h a r *   i m ,   i n t   r e L e n ,   i n t   i m L e n ,   i n t   p r e c )   {  
         / /   G e t   0 . 1   i n   b i n a r y   t o   c o n v e r t   b a s e   1 0   t o   b i n a r y  
         D e c i m a l   t e n t h ;  
         C r e a t e D e c i m a l ( & t e n t h ,   p r e c ) ;  
         I n i t D e c i m a l ( & t e n t h ) ;  
         ( ( u n s i g n e d   i n t * ) t e n t h ) [ 1 ]   =   0 x 0 0 0 0 0 0 0 0 U ;  
         ( ( u n s i g n e d   i n t * ) t e n t h ) [ 2 ]   =   0 x 1 9 9 9 9 9 9 9 U ;  
         f o r   ( i n t   i   =   3 ; i   <   p r e c   +   2 ; i + + )   {  
                 ( ( u n s i g n e d   i n t * ) t e n t h ) [ i ]   =   0 x 9 9 9 9 9 9 9 9 U ;  
         }  
         / /   C o n v e r t   t h e   r e   f r o m   s t r i n g   t o   b i n a r y   d e c i m a l  
         D e c i m a l   d R e ;  
         C r e a t e D e c i m a l ( & d R e ,   p r e c ) ;  
         I n i t D e c i m a l ( & d R e ) ;  
 }  
  
 i n t   m a i n ( )  
 {  
         c u I n i t ( 0 ) ;  
         c o u t   < <   " s t a r t ! \ n " ;  
         i f s t r e a m   i n ( " i n . t x t " ) ;  
         d o u b l e   r e ,   i m ,   z o o m ;  
         i n   > >   r e   > >   i m   > >   z o o m ;  
         c o n s t   i n t   x S z   =   4 0 9 6 ,   y S z   =   2 1 6 0 ;  
         C U d e v i c e p t r   a r r = 0 ;  
         i n t   r e s   =   c u M e m A l l o c ( & a r r ,   x S z   *   y S z   *   3 ) ;   / /   I n   c a s e   n o t   e n o u g h   m e m o r y  
         i f   ( r e s   ! =   c u d a S u c c e s s )   {  
                 c e r r   < <   " E R R O R ! " < < r e s ;  
                 r e t u r n   r e s ;  
         }  
         c o u t   < <   " M a l l o c \ n " ;  
         c a l c R o w   < < < 1 ,   y S z   > > >   ( a r r ) ;  
         c u d a D e v i c e S y n c h r o n i z e ( ) ;  
         c u M e m F r e e ( a r r ) ;  
         c o u t   < <   " D o n e ! " ;  
         r e t u r n   0 ;  
 }  
 