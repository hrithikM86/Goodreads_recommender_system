#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
// Truncated FOSSIL: shared history embeddings + personalized high-order Markov weights.
extern "C" void fossil_train(int U,int k,const int* ptr,const int* seq,const int* sp,const int* seen,const int* active,int na,int start,int epochs,int seed,float lr,float reg,float* A,float* B,float* wu,float* wg,float* bias){
 int total=ptr[U];std::vector<float> x(k),delta(k),ad(20*k),wd(3);std::vector<int> hist(20);std::vector<float> coef(20);std::mt19937 rng(seed);
 for(int epoch=start;epoch<start+epochs;epoch++){rng.seed(seed+epoch);float rate=lr*std::pow(.98f,epoch);
  for(int z=0;z<total-U;z++){int u=rng()%U,lo=ptr[u],hi=ptr[u+1];if(hi-lo<2)continue;int pos=lo+1+rng()%(hi-lo-1),i=seq[pos],j,tries=0;
   do{j=(rng()%2)?seq[rng()%total]:active[rng()%na];tries++;}while((j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))&&tries<100);if(j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))continue;
   int n=std::min(20,pos-lo);float norm=1/std::sqrt(float(n));std::fill(x.begin(),x.end(),0);
   for(int h=0;h<n;h++){hist[h]=seq[pos-1-h];coef[h]=norm+(h<3?wg[h]+wu[u*3+h]:0);for(int f=0;f<k;f++)x[f]+=coef[h]*A[hist[h]*k+f];}
   float diff=bias[i]-bias[j];for(int f=0;f<k;f++)diff+=x[f]*(B[i*k+f]-B[j*k+f]);float g=1/(1+std::exp(std::clamp(diff,-30.f,30.f)));
   for(int f=0;f<k;f++)delta[f]=B[i*k+f]-B[j*k+f];
   for(int h=0;h<std::min(3,n);h++){float v=0;for(int f=0;f<k;f++)v+=A[hist[h]*k+f]*delta[f];wd[h]=g*v;}
   for(int h=0;h<n;h++)for(int f=0;f<k;f++)A[hist[h]*k+f]+=rate*(g*coef[h]*delta[f]-reg*A[hist[h]*k+f]);
   for(int f=0;f<k;f++){B[i*k+f]+=rate*(g*x[f]-reg*B[i*k+f]);B[j*k+f]+=rate*(-g*x[f]-reg*B[j*k+f]);}
   for(int h=0;h<std::min(3,n);h++){wu[u*3+h]+=rate*(wd[h]-reg*wu[u*3+h]);wg[h]+=rate*.1f*(wd[h]-reg*wg[h]);}
   bias[i]+=rate*(g-reg*bias[i]);bias[j]+=rate*(-g-reg*bias[j]);
  }
 }
}
