#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
// Exact cached implicit updates for ratings traversed in per-user blocks.
extern "C" void svdpp_train(int U,int k,const int* ptr,const int* item,const float* rating,
 const int* ic,float mu,float* P,float* Q,float* Y,float* bu,float* bi,
 int start,int epochs,int seed,float lr,float reg,int shuffle){
 std::vector<int> users(U),order;std::vector<float> z(k),initial(k);
 for(int epoch=start;epoch<start+epochs;epoch++){
  std::iota(users.begin(),users.end(),0);std::mt19937 rng(seed+epoch);if(shuffle)std::shuffle(users.begin(),users.end(),rng);float eta=lr*std::pow(.985f,epoch);
  for(int u:users){int n=ptr[u+1]-ptr[u];if(!n)continue;float norm=1.f/std::sqrt(float(n));std::fill(z.begin(),z.end(),0);
   for(int row=ptr[u];row<ptr[u+1];row++)for(int f=0;f<k;f++)z[f]+=Y[item[row]*k+f]*norm;initial=z;order.resize(n);std::iota(order.begin(),order.end(),ptr[u]);if(shuffle)std::shuffle(order.begin(),order.end(),rng);
   for(int row:order){int j=item[row];float pred=mu+bu[u]+bi[j];for(int f=0;f<k;f++)pred+=(P[u*k+f]+z[f])*Q[j*k+f];float e=rating[row]-pred;bu[u]+=eta*(e-10.f/n*bu[u]);bi[j]+=eta*(e-10.f/ic[j]*bi[j]);
    for(int f=0;f<k;f++){int p=u*k+f,q=j*k+f;float oldq=Q[q],effective=P[p]+z[f];P[p]+=eta*(e*oldq-reg*P[p]);Q[q]+=eta*(e*effective-reg*oldq);z[f]+=eta*(e*oldq-reg*z[f]);}
   }
   float decay=std::pow(1.f-eta*reg,n);for(int row=ptr[u];row<ptr[u+1];row++)for(int f=0;f<k;f++){int j=item[row]*k+f;Y[j]=decay*Y[j]+(z[f]-decay*initial[f])*norm;}
  }
 }
}
