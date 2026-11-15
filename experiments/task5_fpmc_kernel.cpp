#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
// Joint BPR optimization of score(u,j|l)=bias[j]+P[u].Q[j]+A[l].B[j].
extern "C" void train_fpmc(int n,int k,const int* users,const int* last,const int* next,
 const int* ptr,const int* seen,const int* pool,int poolsize,const int* active,int active_size,
 float* P,float* Q,float* A,float* B,float* bias,int start,int epochs,int seed,
 float lr,float reg,float popularity_mix,int uniform_users){
 std::vector<int> order(n);int U=users[n-1]+1;std::vector<int> transitionptr(U+1,0);
 for(int z=0;z<n;z++)transitionptr[users[z]+1]++;for(int u=1;u<=U;u++)transitionptr[u]+=transitionptr[u-1];
 for(int epoch=start;epoch<start+epochs;epoch++){
  std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+epoch);if(!uniform_users)std::shuffle(order.begin(),order.end(),rng);float rate=lr*std::pow(.98f,epoch);
  for(int sampled:order){int row=sampled;if(uniform_users){int u=rng()%U;row=transitionptr[u]+rng()%(transitionptr[u+1]-transitionptr[u]);}int u=users[row],l=last[row],i=next[row],j;int tries=0;
   do {j=(double(rng())/rng.max()<popularity_mix)?pool[rng()%poolsize]:active[rng()%active_size];tries++;}
   while((j==i||std::binary_search(seen+ptr[u],seen+ptr[u+1],j))&&tries<100);
   if(j==i||std::binary_search(seen+ptr[u],seen+ptr[u+1],j))continue;
   float diff=bias[i]-bias[j];
   for(int f=0;f<k;f++)diff+=P[u*k+f]*(Q[i*k+f]-Q[j*k+f])+A[l*k+f]*(B[i*k+f]-B[j*k+f]);
   float g=1.f/(1.f+std::exp(std::min(30.f,std::max(-30.f,diff))));
   bias[i]+=rate*(g-reg*bias[i]);bias[j]+=rate*(-g-reg*bias[j]);
   for(int f=0;f<k;f++){
    float p=P[u*k+f],qi=Q[i*k+f],qj=Q[j*k+f],a=A[l*k+f],bi=B[i*k+f],bj=B[j*k+f];
    P[u*k+f]+=rate*(g*(qi-qj)-reg*p);Q[i*k+f]+=rate*(g*p-reg*qi);Q[j*k+f]+=rate*(-g*p-reg*qj);
    A[l*k+f]+=rate*(g*(bi-bj)-reg*a);B[i*k+f]+=rate*(g*a-reg*bi);B[j*k+f]+=rate*(-g*a-reg*bj);
   }
  }
 }
}
