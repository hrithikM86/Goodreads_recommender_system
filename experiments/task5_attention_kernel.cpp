#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
struct Work {std::vector<float> si,sj,ai,aj,di,dj,ge,gvi,gvj,gp;std::vector<int> hist;Work(int k):si(10),sj(10),ai(10),aj(10),di(10),dj(10),ge(10*k),gvi(k),gvj(k),gp(10),hist(10){} };
static void update_attention(int u,int lo,int pos,int i,int j,int k,const int* seq,const float* P,const float* Q,float* E,float* V,float* bias,float* position,float rate,Work& w){
 const int L=10;auto &si=w.si,&sj=w.sj,&ai=w.ai,&aj=w.aj,&di=w.di,&dj=w.dj,&ge=w.ge,&gvi=w.gvi,&gvj=w.gvj,&gp=w.gp;auto &hist=w.hist;
   int n=std::min(L,pos-lo);float scale=1/std::sqrt(float(k)),mi=-1e30f,mj=-1e30f;
   for(int h=0;h<n;h++){hist[h]=seq[pos-1-h];si[h]=sj[h]=0;for(int f=0;f<k;f++){float e=E[hist[h]*k+f];si[h]+=e*V[i*k+f];sj[h]+=e*V[j*k+f];}mi=std::max(mi,si[h]*scale+position[h]);mj=std::max(mj,sj[h]*scale+position[h]);}
   float zi=0,zj=0;for(int h=0;h<n;h++){ai[h]=std::exp(si[h]*scale+position[h]-mi);aj[h]=std::exp(sj[h]*scale+position[h]-mj);zi+=ai[h];zj+=aj[h];}
   float scorei=0,scorej=0;for(int h=0;h<n;h++){ai[h]/=zi;aj[h]/=zj;scorei+=ai[h]*si[h];scorej+=aj[h]*sj[h];}
   float diff=bias[i]-bias[j]+scorei-scorej;for(int f=0;f<k;f++)diff+=P[u*k+f]*(Q[i*k+f]-Q[j*k+f]);float g=1/(1+std::exp(std::clamp(diff,-30.f,30.f)));std::fill(gvi.begin(),gvi.end(),0);std::fill(gvj.begin(),gvj.end(),0);
   for(int h=0;h<n;h++){di[h]=ai[h]*(1+(si[h]-scorei)*scale);dj[h]=aj[h]*(1+(sj[h]-scorej)*scale);gp[h]=g*(ai[h]*(si[h]-scorei)-aj[h]*(sj[h]-scorej));for(int f=0;f<k;f++){float e=E[hist[h]*k+f];ge[h*k+f]=g*(di[h]*V[i*k+f]-dj[h]*V[j*k+f])-.02f*e;gvi[f]+=g*di[h]*e;gvj[f]-=g*dj[h]*e;}}
   for(int h=0;h<n;h++){for(int f=0;f<k;f++)E[hist[h]*k+f]+=rate*ge[h*k+f];position[h]+=rate*.1f*(gp[h]-.01f*position[h]);}
   for(int f=0;f<k;f++){V[i*k+f]+=rate*(gvi[f]-.02f*V[i*k+f]);V[j*k+f]+=rate*(gvj[f]-.02f*V[j*k+f]);}bias[i]+=rate*(g-.02f*bias[i]);bias[j]+=rate*(-g-.02f*bias[j]);
}
extern "C" void attention_update(int k,int u,int lo,int pos,int i,int j,const int* seq,const float* P,const float* Q,float* E,float* V,float* bias,float* position,float rate){Work w(k);update_attention(u,lo,pos,i,j,k,seq,P,Q,E,V,bias,position,rate,w);}
// Candidate-aware attention over recent books, with frozen long-term factors.
extern "C" void train_attention(int U,int k,const int* ptr,const int* seq,const int* sp,const int* seen,const int* active,int na,const float* P,const float* Q,float* E,float* V,float* bias,float* position,int start,int epochs,int seed,float lr){
 int total=ptr[U];std::mt19937 rng(seed);Work work(k);
 for(int epoch=start;epoch<start+epochs;epoch++){rng.seed(seed+epoch);float rate=lr*std::pow(.98f,epoch);
  for(int z=0;z<total-U;z++){int u=rng()%U,lo=ptr[u],hi=ptr[u+1];if(hi-lo<2)continue;int available=hi-lo-1;int span=(rng()%2)?std::min(available,10):available;int pos=hi-span+rng()%span,i=seq[pos],j,tries=0;
   do{j=(rng()%2)?seq[rng()%total]:active[rng()%na];tries++;}while((j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))&&tries<100);if(j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))continue;
   update_attention(u,lo,pos,i,j,k,seq,P,Q,E,V,bias,position,rate,work);
  }
 }
}
