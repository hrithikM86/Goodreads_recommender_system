#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

// Exact second-order FM: b + w*x + .5*sum_f[(sum_j V_jf*x_j)^2-sum_j(V_jf*x_j)^2].
extern "C" void fm_predict(int n,int m,int k,const int* ids,const float* x,
 const float* linear,const float* V,float mean,float* output){
 std::vector<float> sum(k),square(k);
 for(int row=0;row<n;row++){
  std::fill(sum.begin(),sum.end(),0);std::fill(square.begin(),square.end(),0);float pred=mean;
  for(int a=0;a<m;a++){int j=ids[row*m+a];float value=x[row*m+a];if(j<0||value==0)continue;
   pred+=linear[j]*value;
   for(int f=0;f<k;f++){float v=V[j*k+f]*value;sum[f]+=v;square[f]+=v*v;}
  }
  for(int f=0;f<k;f++)pred+=.5f*(sum[f]*sum[f]-square[f]);output[row]=pred;
 }
}
extern "C" void fm_train(int n,int m,int k,const int* ids,const float* x,const float* y,
 float* linear,float* V,float mean,int start,int epochs,int seed,float lr,float reg,float linear_reg){
 std::vector<int> order(n);std::vector<float> sum(k),square(k);
 for(int epoch=start;epoch<start+epochs;epoch++){
  std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+epoch);std::shuffle(order.begin(),order.end(),rng);
  float rate=lr*std::pow(.98f,epoch);
  for(int row:order){
   std::fill(sum.begin(),sum.end(),0);std::fill(square.begin(),square.end(),0);float pred=mean;
   for(int a=0;a<m;a++){int j=ids[row*m+a];float value=x[row*m+a];if(j<0||value==0)continue;
    pred+=linear[j]*value;
    for(int f=0;f<k;f++){float v=V[j*k+f]*value;sum[f]+=v;square[f]+=v*v;}
   }
   for(int f=0;f<k;f++)pred+=.5f*(sum[f]*sum[f]-square[f]);float error=y[row]-pred;
   for(int a=0;a<m;a++){int j=ids[row*m+a];float value=x[row*m+a];if(j<0||value==0)continue;
    linear[j]+=rate*(error*value-linear_reg*linear[j]);
    for(int f=0;f<k;f++){float old=V[j*k+f];V[j*k+f]+=rate*(error*value*(sum[f]-old*value)-reg*old);}
   }
  }
 }
}
