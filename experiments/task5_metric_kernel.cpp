#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
// kind=0: PRME-style personalized metric; kind=1: TransRec translation.
extern "C" void metric_train(int U,int k,const int* ptr,const int* seq,const int* seenptr,const int* seen,const int* pool,int poolsize,const int* active,int na,float* P,float* Q,float* global,float* bias,int kind,int start,int epochs,int seed,float lr,float reg){
 int n=poolsize-U;std::mt19937 rng(seed);std::vector<float> gp(k),gl(k),gi(k),gj(k);
 auto project=[&](float* v){float norm=0;for(int f=0;f<k;f++)norm+=v[f]*v[f];if(norm>1){float scale=1/std::sqrt(norm);for(int f=0;f<k;f++)v[f]*=scale;}};
 for(int epoch=start;epoch<start+epochs;epoch++){rng.seed(seed+epoch);float rate=lr*std::pow(.99f,epoch);
  for(int z=0;z<n;z++){int u=rng()%U,lo=ptr[u],hi=ptr[u+1];if(hi-lo<2)continue;int pos=lo+1+rng()%(hi-lo-1);int l=seq[pos-1],i=seq[pos],j;int tries=0;
   do{j=(rng()%2)?pool[rng()%poolsize]:active[rng()%na];tries++;}while((j==i||std::binary_search(seen+seenptr[u],seen+seenptr[u+1],j))&&tries<100);
   if(j==i||std::binary_search(seen+seenptr[u],seen+seenptr[u+1],j))continue;
   float diff=bias[i]-bias[j];
   for(int f=0;f<k;f++){float p=P[u*k+f],lq=Q[l*k+f],a=Q[i*k+f],b=Q[j*k+f];if(kind==0){diff+=.5f*((p-b)*(p-b)-(p-a)*(p-a)+(lq-b)*(lq-b)-(lq-a)*(lq-a));gp[f]=a-b;gl[f]=a-b;gi[f]=p+lq-2*a;gj[f]=-p-lq+2*b;}else{float c=p+lq+global[f];diff+=(c-b)*(c-b)-(c-a)*(c-a);gp[f]=2*(a-b);gl[f]=gp[f];gi[f]=2*(c-a);gj[f]=-2*(c-b);}}
   float g=1/(1+std::exp(std::clamp(diff,-30.f,30.f)));bias[i]+=rate*(g-reg*bias[i]);bias[j]+=rate*(-g-reg*bias[j]);
   for(int f=0;f<k;f++){P[u*k+f]+=rate*(g*gp[f]-reg*P[u*k+f]);Q[l*k+f]+=rate*(g*gl[f]-reg*Q[l*k+f]);Q[i*k+f]+=rate*(g*gi[f]-reg*Q[i*k+f]);Q[j*k+f]+=rate*(g*gj[f]-reg*Q[j*k+f]);if(kind==1)global[f]+=rate*(g*gp[f]-reg*global[f]);}
   project(P+u*k);project(Q+l*k);project(Q+i*k);project(Q+j*k);if(kind==1)project(global);
  }
 }
}
