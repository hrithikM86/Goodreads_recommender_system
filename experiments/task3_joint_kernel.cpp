#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
// User-block SGD for joint temporal SVD++. Cached implicit updates are exact
// equivalents of updating all history Y vectors after every rating in a block.
extern "C" void joint_train(int U,int k,const int* ptr,const int* items,const int* day,
 const int* iper,const int* uper,const int* month,const float* dev,const float* y,
 const int* ic,const int* dc,const int* ipc,const int* upc,const int* mc,float mu,
 float* P,float* Q,float* Y,float* A,float* bu,float* bi,float* du,float* bt,float* ut,float* mt,float* alpha,
 int start,int epochs,int seed,float lr,float reg,float rday,float rperiod,float ruserperiod){
 std::vector<int> users(U);std::vector<int> order;std::vector<float> z(k),initial(k);
 for(int ep=start;ep<start+epochs;ep++){
  std::iota(users.begin(),users.end(),0);std::mt19937 rng(seed+ep);std::shuffle(users.begin(),users.end(),rng);float eta=lr*std::pow(.985f,ep);
  for(int u:users){int n=ptr[u+1]-ptr[u];if(!n)continue;float norm=1.f/std::sqrt(float(n));std::fill(z.begin(),z.end(),0.f);
   for(int a=ptr[u];a<ptr[u+1];a++)for(int f=0;f<k;f++)z[f]+=Y[items[a]*k+f]*norm;initial=z;
   order.resize(n);std::iota(order.begin(),order.end(),ptr[u]);std::shuffle(order.begin(),order.end(),rng);
   for(int a:order){int j=items[a],d=day[a],b=iper[a],v=uper[a],m=month[a];float t=dev[a];float pred=mu+bu[u]+bi[j]+du[d]+bt[b]+ut[v]+mt[m]+alpha[u]*t;
    for(int f=0;f<k;f++)pred+=(P[u*k+f]+z[f]+t*A[u*k+f])*Q[j*k+f];float e=y[a]-pred;
    bu[u]+=eta*(e-3.f/n*bu[u]);bi[j]+=eta*(e-5.f/ic[j]*bi[j]);du[d]+=eta*(e-rday/dc[d]*du[d]);bt[b]+=eta*(e-rperiod/ipc[b]*bt[b]);ut[v]+=eta*(e-ruserperiod/upc[v]*ut[v]);mt[m]+=eta*(e-100.f/mc[m]*mt[m]);alpha[u]+=eta*(e*t-10.f/n*alpha[u]);
    for(int f=0;f<k;f++){int p=u*k+f,q=j*k+f;float oldq=Q[q],eff=P[p]+z[f]+t*A[p];P[p]+=eta*(e*oldq-reg*P[p]);A[p]+=eta*(e*oldq*t-.15f*A[p]);Q[q]+=eta*(e*eff-reg*oldq);z[f]+=eta*(e*oldq-reg*z[f]);}
   }
   float decay=std::pow(1.f-eta*reg,n);
   for(int a=ptr[u];a<ptr[u+1];a++)for(int f=0;f<k;f++){int j=items[a]*k+f;Y[j]=decay*Y[j]+(z[f]-decay*initial[f])*norm;}
  }
 }
}
