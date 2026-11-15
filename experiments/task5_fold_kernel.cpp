#include <algorithm>
#include <cmath>
#include <random>
// Adapt only query-user factors from known recent books; item factors stay frozen.
extern "C" void fold_users(int N,int k,const int* ptr,const int* seq,const int* sp,const int* seen,const int* users,const int* active,int na,const int* known,const float* P,const float* Q,const float* bias,float* result,int cap,int start,int steps){
 for(int row=0;row<N;row++){int u=users[row],lo=ptr[row],hi=ptr[row+1];std::mt19937 rng(97+row+start*31+cap*13);int begin=cap?std::max(lo,hi-cap):lo;
  for(int step=0;step<steps;step++){int i=-1;for(int tries=0;tries<100;tries++){int candidate=seq[begin+rng()%(hi-begin)];if(known[candidate]){i=candidate;break;}}if(i<0)continue;int j,tries=0;do{j=active[rng()%na];tries++;}while(std::binary_search(seen+sp[row],seen+sp[row+1],j)&&tries<100);if(std::binary_search(seen+sp[row],seen+sp[row+1],j))continue;
   float diff=bias[i]-bias[j];for(int f=0;f<k;f++)diff+=result[row*k+f]*(Q[i*k+f]-Q[j*k+f]);float g=1/(1+std::exp(std::clamp(diff,-30.f,30.f)));
   for(int f=0;f<k;f++)result[row*k+f]+=.02f*(g*(Q[i*k+f]-Q[j*k+f])-.1f*(result[row*k+f]-P[u*k+f]));
  }
 }
}
