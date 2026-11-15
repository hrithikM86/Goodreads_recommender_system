#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <cmath>
// Residual MF: (p_u + dev_u(t)*a_u + scale_{u,b}*s_{u,b}) dot q_i.
extern "C" void train_mf(int n,int k,const int* u,const int* i,const int* period,
 const float* dev,const float* scale,const float* y,float* P,float* A,float* S,float* Q,
 int mode,int start,int epochs,int seed,float lr,float rp,float rq,float ra,float rs) {
 std::vector<int> order(n);
 for(int epoch=start;epoch<start+epochs;epoch++){
  std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+epoch);std::shuffle(order.begin(),order.end(),rng);
  float rate=lr*std::pow(.98f,epoch);
  for(int row:order){
   int uu=u[row],ii=i[row],bb=period[row];float d=mode?dev[row]:0.f;float w=mode?scale[row]:0.f;
   float* p=P+uu*k;float* a=A+uu*k;float* s=S+bb*k;float* q=Q+ii*k;float pred=0;
   for(int f=0;f<k;f++)pred+=(p[f]+d*a[f]+w*s[f])*q[f];
   float error=y[row]-pred;
   for(int f=0;f<k;f++){
    float oldq=q[f],eff=p[f]+d*a[f]+w*s[f];
    p[f]+=rate*(error*oldq-rp*p[f]);
    if(mode){a[f]+=rate*(error*oldq*d-ra*a[f]);s[f]+=rate*(error*oldq*w-rs*s[f]);}
    q[f]+=rate*(error*eff-rq*oldq);
   }
  }
 }
}
extern "C" void predict_mf(int n,int k,int nu,int ni,const int* u,const int* i,
 const int* period,const float* dev,const float* scale,const int* uc,const int* ic,
 const float* P,const float* A,const float* S,const float* Q,int mode,float* out){
 for(int z=0;z<n;z++){
  int uu=u[z],ii=i[z],b=period[z];float v=0;
  if(uu>=0&&uu<nu&&ii>=0&&ii<ni&&uc[uu]>0&&ic[ii]>0){
   for(int f=0;f<k;f++){
    float eff=P[uu*k+f];if(mode)eff+=dev[z]*A[uu*k+f]+(b>=0?scale[z]*S[b*k+f]:0.f);
    v+=eff*Q[ii*k+f];
   }
  }
  out[z]=v;
 }
}
