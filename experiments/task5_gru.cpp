#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstdio>
static float sig(float x){return 1.f/(1.f+std::exp(-std::clamp(x,-30.f,30.f)));}
// GRU convention h=(1-z)*hprev+z*n; candidate dot product plus item bias.
extern "C" double gru_grad(int k,int T,int neg,const int* xids,const int* cand,const float* h0,const float* E,const float* Q,const float* b,const float* W,const float* gb,float* gx,float* gq,float* gbi,float* gw,float* ggb,float* gh0,float* last){
 int C=neg+1;std::vector<float> H((T+1)*k),Z(T*k),R(T*k),N(T*k),DH(T*k),rh(k);std::copy(h0,h0+k,H.begin());std::fill(gx,gx+T*k,0);std::fill(gq,gq+T*C*k,0);std::fill(gbi,gbi+T*C,0);std::fill(gw,gw+6*k*k,0);std::fill(ggb,ggb+3*k,0);double loss=0;float scale=1.f/(T*neg);
 for(int t=0;t<T;t++){
  const float* x=E+xids[t]*k;float* p=H.data()+t*k;float* h=p+k;
  for(int f=0;f<k;f++){float az=gb[f],ar=gb[k+f];for(int j=0;j<k;j++){az+=W[f*k+j]*x[j]+W[k*k+f*k+j]*p[j];ar+=W[2*k*k+f*k+j]*x[j]+W[3*k*k+f*k+j]*p[j];}Z[t*k+f]=sig(az);R[t*k+f]=sig(ar);rh[f]=R[t*k+f]*p[f];}
  for(int f=0;f<k;f++){float an=gb[2*k+f];for(int j=0;j<k;j++)an+=W[4*k*k+f*k+j]*x[j]+W[5*k*k+f*k+j]*rh[j];float n=std::tanh(an),z=Z[t*k+f];N[t*k+f]=n;h[f]=(1-z)*p[f]+z*n;}
  int pos=cand[t*C];
  for(int c=1;c<C;c++){int j=cand[t*C+c];float diff=b[pos]-b[j];for(int f=0;f<k;f++)diff+=h[f]*(Q[pos*k+f]-Q[j*k+f]);loss+=(std::max(-double(diff),0.)+std::log1p(std::exp(-std::abs(double(diff)))))*scale;float g=-sig(-diff)*scale;gbi[t*C]+=g;gbi[t*C+c]-=g;
   for(int f=0;f<k;f++){DH[t*k+f]+=g*(Q[pos*k+f]-Q[j*k+f]);gq[(t*C)*k+f]+=g*h[f];gq[(t*C+c)*k+f]-=g*h[f];}}
 }
 std::copy(H.end()-k,H.end(),last);std::vector<float> carry(k),dh(k),dz(k),dn(k),dr(k),dp(k),drh(k);
 for(int t=T-1;t>=0;t--){const float* p=H.data()+t*k;const float* x=E+xids[t]*k;
  for(int f=0;f<k;f++){float z=Z[t*k+f],n=N[t*k+f];dh[f]=DH[t*k+f]+carry[f];dn[f]=dh[f]*z*(1-n*n);dz[f]=dh[f]*(n-p[f])*z*(1-z);dp[f]=dh[f]*(1-z);drh[f]=0;}
  for(int f=0;f<k;f++)for(int j=0;j<k;j++)drh[j]+=W[5*k*k+f*k+j]*dn[f];
  for(int f=0;f<k;f++){float r=R[t*k+f];dr[f]=drh[f]*p[f]*r*(1-r);dp[f]+=drh[f]*r;}
  for(int f=0;f<k;f++){
   ggb[f]+=dz[f];ggb[k+f]+=dr[f];ggb[2*k+f]+=dn[f];
   for(int j=0;j<k;j++){
    gx[t*k+j]+=W[f*k+j]*dz[f]+W[2*k*k+f*k+j]*dr[f]+W[4*k*k+f*k+j]*dn[f];dp[j]+=W[k*k+f*k+j]*dz[f]+W[3*k*k+f*k+j]*dr[f];
    gw[f*k+j]+=dz[f]*x[j];gw[k*k+f*k+j]+=dz[f]*p[j];gw[2*k*k+f*k+j]+=dr[f]*x[j];gw[3*k*k+f*k+j]+=dr[f]*p[j];gw[4*k*k+f*k+j]+=dn[f]*x[j];gw[5*k*k+f*k+j]+=dn[f]*R[t*k+j]*p[j];
   }
  }carry=dp;
 }
 std::copy(carry.begin(),carry.end(),gh0);return loss;
}
static void ada(float* x,float* acc,const float* g,int n,float lr,float clip,float decay=0){for(int j=0;j<n;j++){float v=clip*g[j]+decay*x[j];acc[j]+=v*v;x[j]-=lr*v/(std::sqrt(acc[j])+1e-8f);}}
extern "C" double gru_train(int U,int I,int k,int epochs,int start,int seed,int trunc,int neg,const int* ptr,const int* seq,const int* sp,const int* seen,const int* pool,int npool,float* E,float* Q,float* b,float* W,float* gb,float* aE,float* aQ,float* ab,float* aW,float* agb,float lr){
 int C=neg+1;std::vector<int> order(U),cand(trunc*C);std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed);std::vector<float> h(k),last(k),gx(trunc*k),gq(trunc*C*k),gbi(trunc*C),gw(6*k*k),ggb(3*k),gh(k);double total=0;long count=0;
 for(int epoch=start;epoch<start+epochs;epoch++){rng.seed(seed+epoch);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);double eloss=0;long ec=0;
  for(int u:order){std::fill(h.begin(),h.end(),0);int lo=ptr[u],hi=ptr[u+1];for(int offset=lo;offset<hi-1;offset+=trunc){int T=std::min(trunc,hi-1-offset);
   for(int t=0;t<T;t++){cand[t*C]=seq[offset+t+1];for(int c=1;c<C;c++){int j;do{j=pool[rng()%npool];}while(std::binary_search(seen+sp[u],seen+sp[u+1],j));cand[t*C+c]=j;}}
   double loss=gru_grad(k,T,neg,seq+offset,cand.data(),h.data(),E,Q,b,W,gb,gx.data(),gq.data(),gbi.data(),gw.data(),ggb.data(),gh.data(),last.data());h=last;eloss+=loss*T;ec+=T;
   double norm=0;for(float v:gw)norm+=v*v;for(float v:ggb)norm+=v*v;for(int z=0;z<T*k;z++)norm+=gx[z]*gx[z];for(int z=0;z<T*C*k;z++)norm+=gq[z]*gq[z];float clip=norm>25?5/std::sqrt(norm):1;
   ada(W,aW,gw.data(),6*k*k,lr,clip,1e-5);ada(gb,agb,ggb.data(),3*k,lr,clip);
   for(int t=0;t<T;t++){int x=seq[offset+t];ada(E+x*k,aE+x*k,gx.data()+t*k,k,lr,clip);for(int c=0;c<C;c++){int j=cand[t*C+c];ada(Q+j*k,aQ+j*k,gq.data()+(t*C+c)*k,k,lr,clip);ada(b+j,ab+j,gbi.data()+t*C+c,1,lr,clip);}}
  }}total+=eloss;count+=ec;std::printf("GRU_EPOCH %d loss %.7f steps %ld\n",epoch+1,eloss/ec,ec);std::fflush(stdout);
 }return total/std::max(count,1L);
}
extern "C" void gru_encode(int U,int k,const int* ptr,const int* seq,const float* E,const float* W,const float* gb,float* result){
 std::vector<float> h(k),z(k),r(k),n(k),rh(k);
 for(int u=0;u<U;u++){std::fill(h.begin(),h.end(),0);for(int t=ptr[u];t<ptr[u+1];t++){const float* x=E+seq[t]*k;for(int f=0;f<k;f++){float az=gb[f],ar=gb[k+f];for(int j=0;j<k;j++){az+=W[f*k+j]*x[j]+W[k*k+f*k+j]*h[j];ar+=W[2*k*k+f*k+j]*x[j]+W[3*k*k+f*k+j]*h[j];}z[f]=sig(az);r[f]=sig(ar);rh[f]=r[f]*h[f];}
  for(int f=0;f<k;f++){float an=gb[2*k+f];for(int j=0;j<k;j++)an+=W[4*k*k+f*k+j]*x[j]+W[5*k*k+f*k+j]*rh[j];n[f]=std::tanh(an);}for(int f=0;f<k;f++)h[f]=(1-z[f])*h[f]+z[f]*n[f];}std::copy(h.begin(),h.end(),result+u*k);}
}
