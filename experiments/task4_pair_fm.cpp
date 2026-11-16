#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
// Second-order FM with logistic interaction classification, evaluated as paired score differences.
static float score(int m,int k,const int* ids,const float*w,const float*V,std::vector<float>&s){
 std::fill(s.begin(),s.end(),0.f);float r=0;
 for(int a=0;a<m;a++){int j=ids[a];if(j<0)continue;r+=w[j];for(int f=0;f<k;f++){float v=V[j*k+f];r-=.5f*v*v;s[f]+=v;}}
 for(int f=0;f<k;f++)r+=.5f*s[f]*s[f];return r;
}
extern "C" void predict(int n,int m,int k,const int*ids,const float*w,const float*V,float*out){std::vector<float>s(k);for(int r=0;r<n;r++)out[r]=score(m,k,ids+r*m,w,V,s);}
extern "C" void train(int n,int m,int k,int I,int item_offset,const int*ids,const int*pool,float*w,float*V,int start,int epochs,int seed){
 std::vector<float>s(k);std::vector<int>x(m);std::vector<int>order(n);
 for(int ep=start;ep<start+epochs;ep++){
  std::mt19937 rng(seed+ep);std::uniform_int_distribution<int>pop(0,n-1),uniform(0,I-1);for(int r=0;r<n;r++)order[r]=r;std::shuffle(order.begin(),order.end(),rng);
  float lr=.02f*std::pow(.97f,ep);
  for(int row:order){for(int label=1;label>=0;label--){std::copy(ids+row*m,ids+(row+1)*m,x.begin());
   if(!label){int j;do{j=(rng()%2)?uniform(rng):pool[pop(rng)];}while(item_offset+j==x[1]);x[1]=item_offset+j;}
   float pred=score(m,k,x.data(),w,V,s);float g=label-1.f/(1.f+std::exp(-std::max(-30.f,std::min(30.f,pred))));
   for(int a=0;a<m;a++){int j=x[a];if(j<0)continue;w[j]+=lr*(g-.02f*w[j]);for(int f=0;f<k;f++){float old=V[j*k+f];V[j*k+f]+=lr*(g*(s[f]-old)-.04f*old);}}
  }}
 }
}
