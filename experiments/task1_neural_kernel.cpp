#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
// Five-class neural collaborative filtering: tanh(Pu * Qi) -> softmax.
static void step(int u,int i,int y,int k,const int* cu,const int* ci,float* P,float* Q,float* bu,float* bi,float* W,float* b,float* HP,float* HQ,float* Hbu,float* Hbi,float* HW,float* Hb,float lambda){
 std::vector<float> h(k),dh(k);float p[5],d[5],mx=-1e30f,total=0,mean=0;
 for(int f=0;f<k;f++)h[f]=std::tanh(P[u*k+f]*Q[i*k+f]);
 for(int c=0;c<5;c++){float v=b[c]+bu[u*5+c]+bi[i*5+c];for(int f=0;f<k;f++)v+=W[c*k+f]*h[f];p[c]=v;mx=std::max(mx,v);}
 for(int c=0;c<5;c++){p[c]=std::exp(p[c]-mx);total+=p[c];}for(int c=0;c<5;c++){p[c]/=total;mean+=p[c]*(c+1);}
 for(int c=0;c<5;c++)d[c]=p[c]-(y==c?1.f:0.f)+2*lambda*(mean-(y+1))*p[c]*((c+1)-mean);
 for(int f=0;f<k;f++){float g=0;for(int c=0;c<5;c++)g+=d[c]*W[c*k+f];dh[f]=g*(1-h[f]*h[f]);}
 auto update=[](float& value,float& hist,float grad){hist+=grad*grad;value-=.05f*grad/std::sqrt(1.f+hist);};
 for(int f=0;f<k;f++){int a=u*k+f,z=i*k+f;float pu=P[a],qi=Q[z];update(P[a],HP[a],dh[f]*qi+.01f*pu);update(Q[z],HQ[z],dh[f]*pu+.01f*qi);}
 for(int c=0;c<5;c++){int a=u*5+c,z=i*5+c;update(bu[a],Hbu[a],d[c]+5.f/std::max(cu[u],1)*bu[a]);update(bi[z],Hbi[z],d[c]+5.f/std::max(ci[i],1)*bi[z]);update(b[c],Hb[c],d[c]);for(int f=0;f<k;f++){int q=c*k+f;update(W[q],HW[q],d[c]*h[f]+.00001f*W[q]);}}
}
extern "C" void neural_train(int n,int k,const int* u,const int* i,const int* y,const int* cu,const int* ci,float* P,float* Q,float* bu,float* bi,float* W,float* b,float* HP,float* HQ,float* Hbu,float* Hbi,float* HW,float* Hb,int start,int epochs,int seed,float lambda){
 std::vector<int> order(n);for(int epoch=start;epoch<start+epochs;epoch++){std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+epoch);std::shuffle(order.begin(),order.end(),rng);for(int row:order)step(u[row],i[row],y[row],k,cu,ci,P,Q,bu,bi,W,b,HP,HQ,Hbu,Hbi,HW,Hb,lambda);}
}
