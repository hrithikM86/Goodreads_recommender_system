#include <cmath>
#include <random>
extern "C" void train(int n,int I,int K,int B,int epochs,const int*u,const int*i,const int*b,const float*t,const float*P,const float*Q,float*D,float*C,int seed){
std::mt19937 rng(seed);std::uniform_int_distribution<int> row(0,n-1),item(0,I-1);
for(int ep=0;ep<epochs;ep++)for(int z=0;z<n;z++){
 int r=row(rng),v=u[r],a=i[r],j=item(rng),bin=b[r];if(a==j)continue;float time=t[r],s=C[a*B+bin]-C[j*B+bin];
 for(int k=0;k<K;k++)s+=P[v*K+k]*(Q[a*K+k]-Q[j*K+k]);
 for(int k=0;k<16;k++)s+=time*D[v*16+k]*(Q[a*K+k]-Q[j*K+k]);
 float g=1/(1+std::exp(fmax(-30,fmin(30,s)))),lr=.015f;
 C[a*B+bin]+=lr*(g-.1f*C[a*B+bin]);C[j*B+bin]+=lr*(-g-.1f*C[j*B+bin]);
 for(int k=0;k<16;k++)D[v*16+k]+=lr*(g*time*(Q[a*K+k]-Q[j*K+k])-.1f*D[v*16+k]);
}}
