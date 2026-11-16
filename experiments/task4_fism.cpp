#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
// Nearest known training context, target excluded; normalized exponential time weights.
extern "C" void prepare(int n,int cap,const int*users,const int*items,const double*time,const int*ptr,int*nb,float*coef,int*count,int*context_count){
 for(int r=0;r<n;r++){int u=users[r],lo=ptr[u],hi=ptr[u+1],l=r-1,h=r+1,c=0;float sum=0;
  while(c<cap&&(l>=lo||h<hi)){int j;if(l>=lo&&(h>=hi||time[r]-time[l]<=time[h]-time[r]))j=l--;else j=h++;nb[r*cap+c]=items[j];float w=std::exp(-std::abs(time[j]-time[r])/(86400.*180));coef[r*cap+c]=w;sum+=w;context_count[items[j]]++;c++;}
  count[r]=c;if(!c)continue;if(sum<1e-8){sum=c;for(int a=0;a<c;a++)coef[r*cap+a]=1;}
  float norm=1/std::sqrt(sum);for(int a=0;a<c;a++)coef[r*cap+a]*=norm;
 }
}
extern "C" void train(int n,int k,int cap,int I,const int*users,const int*items,const int*ptr,const int*seen_ptr,const int*seen,const int*nb,const float*coef,const int*count,float*A,float*B,float*bias,int start,int epochs,int seed){
 std::vector<int>order(n);std::vector<float>x(k),delta(k);
 for(int ep=start;ep<start+epochs;ep++){
  std::mt19937 rng(seed+ep);for(int r=0;r<n;r++)order[r]=r;std::shuffle(order.begin(),order.end(),rng);float lr=.025f*std::pow(.98f,ep);
  for(int r:order){int c=count[r];if(!c)continue;int u=users[r],i=items[r],j,tries=0;
   do{j=(rng()%2)?items[rng()%n]:rng()%I;tries++;}while((j==i||std::binary_search(seen+seen_ptr[u],seen+seen_ptr[u+1],j))&&tries<100);
   if(j==i||std::binary_search(seen+seen_ptr[u],seen+seen_ptr[u+1],j))continue;
   std::fill(x.begin(),x.end(),0.f);for(int a=0;a<c;a++){int h=nb[r*cap+a];float w=coef[r*cap+a];for(int f=0;f<k;f++)x[f]+=w*A[h*k+f];}
   float diff=bias[i]-bias[j];for(int f=0;f<k;f++){delta[f]=B[i*k+f]-B[j*k+f];diff+=x[f]*delta[f];}float g=1/(1+std::exp(std::max(-30.f,std::min(30.f,diff))));
   for(int a=0;a<c;a++){int h=nb[r*cap+a];float w=coef[r*cap+a];for(int f=0;f<k;f++)A[h*k+f]+=lr*(g*w*delta[f]-.01f*A[h*k+f]);}
   for(int f=0;f<k;f++){B[i*k+f]+=lr*(g*x[f]-.01f*B[i*k+f]);B[j*k+f]+=lr*(-g*x[f]-.01f*B[j*k+f]);}
   bias[i]+=lr*(g-.01f*bias[i]);bias[j]+=lr*(-g-.01f*bias[j]);
  }
 }
}
extern "C" void score(int n,int k,int cap,const int*ptr,const int*items,const double*times,const int*qu,const int*qi,const double*qt,const float*A,const float*B,const float*bias,float*out){
 std::vector<float>x(k),uniform(k),past(k);std::vector<int>ids;std::vector<double>dist;
 for(int r=0;r<n;r++){
  int u=qu[r],j=qi[r];float*o=out+r*4;std::fill(o,o+4,0.f);std::fill(x.begin(),x.end(),0.f);std::fill(uniform.begin(),uniform.end(),0.f);std::fill(past.begin(),past.end(),0.f);ids.clear();dist.clear();
  int lo=ptr[u],hi=ptr[u+1];bool valid=qt[r]>=0&&qt[r]<4102444800.;
  for(int h=lo;h<hi;h++)if(items[h]!=j){ids.push_back(h);dist.push_back(valid?std::abs(times[h]-qt[r]):0.);}
  std::vector<int>order(ids.size());for(int h=0;h<(int)order.size();h++)order[h]=h;int c=std::min(cap,int(order.size()));if(!c)continue;
  std::partial_sort(order.begin(),order.begin()+c,order.end(),[&](int a,int b){return dist[a]!=dist[b]?dist[a]<dist[b]:ids[a]<ids[b];});float sum=0;int pc=0;
  for(int a=0;a<c;a++){int pos=ids[order[a]],h=items[pos];float w=valid?std::exp(-dist[order[a]]/(86400.*180)):1;sum+=w;for(int f=0;f<k;f++){x[f]+=w*A[h*k+f];uniform[f]+=A[h*k+f];}}
  if(sum<1e-8){x=uniform;sum=c;}
  // Independently choose up to cap closest known predecessors.
  for(int a=hi-1;a>=lo&&pc<cap;a--)if(items[a]!=j&&valid&&times[a]<qt[r]){int h=items[a];for(int f=0;f<k;f++)past[f]+=A[h*k+f];pc++;}
  float xn=0,bn=0,dot=0,sc=0,ps=0;for(int f=0;f<k;f++){x[f]/=std::sqrt(sum);xn+=x[f]*x[f];bn+=B[j*k+f]*B[j*k+f];dot+=x[f]*B[j*k+f];sc+=uniform[f]*B[j*k+f]/std::sqrt(float(c));if(pc)ps+=past[f]*B[j*k+f]/std::sqrt(float(pc));}
  o[0]=bias[j]+dot;o[1]=bias[j]+sc;o[2]=dot/std::max(1e-8f,std::sqrt(xn*bn));o[3]=pc?bias[j]+ps:0;
 }
}
