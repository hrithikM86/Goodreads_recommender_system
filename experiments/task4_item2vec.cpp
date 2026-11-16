#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
extern "C" void train(int n,int k,int window,int negatives,const int*users,const int*items,const int*pool,int pooln,float*E,float*O,int start,int epochs,int seed){
 std::vector<int> order(n);std::vector<float> old(k);
 for(int ep=start;ep<start+epochs;ep++){
  std::mt19937 rng(seed+ep);for(int r=0;r<n;r++)order[r]=r;std::shuffle(order.begin(),order.end(),rng);float lr=.025f*std::pow(.95f,ep);
  for(int r:order){int a=items[r],radius=1+rng()%window;
   for(int c=std::max(0,r-radius);c<std::min(n,r+radius+1);c++){
    if(c==r||users[c]!=users[r])continue;int target=items[c];
    for(int z=0;z<=negatives;z++){int b=target;if(z){do{b=pool[rng()%pooln];}while(b==target);}float dot=0;
     for(int f=0;f<k;f++)dot+=E[a*k+f]*O[b*k+f];float g=((z==0?1.f:0.f)-1.f/(1.f+std::exp(-std::max(-20.f,std::min(20.f,dot)))))*lr;
     for(int f=0;f<k;f++){float e=E[a*k+f],o=O[b*k+f];E[a*k+f]+=g*o;O[b*k+f]+=g*e;}
    }
   }
  }
 }
}
extern "C" void score(int n,int k,const int*ptr,const int*hist,const double*times,const int*qu,const int*qi,const double*qt,const float*E,const float*O,const float*V,float*out){
 for(int q=0;q<n;q++){
  int u=qu[q],j=qi[q];float*result=out+q*12;std::fill(result,result+12,0.f);float norm=0;for(int f=0;f<k;f++)norm+=V[j*k+f]*V[j*k+f];if(norm==0)continue;
  double sum=0,rawsum=0,ws[3]={0,0,0},vs[3]={0,0,0},raw180=0;int count=0;float maxcos=-2;
  double before[3]={1e100,1e100,1e100},after[3]={1e100,1e100,1e100};float bc[3]={0,0,0},ac[3]={0,0,0},br[3]={0,0,0};
  bool goodtime=qt[q]>=0&&qt[q]<4102444800.;
  for(int r=ptr[u];r<ptr[u+1];r++){
   int h=hist[r];if(h==j)continue;float hn=0,c=0,raw=0;
   for(int f=0;f<k;f++){hn+=V[h*k+f]*V[h*k+f];c+=V[h*k+f]*V[j*k+f];raw+=E[h*k+f]*O[j*k+f];}if(hn==0)continue;
   sum+=c;rawsum+=raw;count++;maxcos=std::max(maxcos,c);if(!goodtime)continue;double dt=(qt[q]-times[r])/86400.;double ds=std::abs(dt);int hs[3]={30,180,730};
   for(int l=0;l<3;l++){double w=std::exp(-ds/hs[l]);ws[l]+=w;vs[l]+=w*c;if(l==1)raw180+=w*raw;}
   double*best=dt>0?before:after;float*values=dt>0?bc:ac;
   for(int l=0;l<3;l++)if(ds<best[l]){for(int m=2;m>l;m--){best[m]=best[m-1];values[m]=values[m-1];if(dt>0)br[m]=br[m-1];}best[l]=ds;values[l]=c;if(dt>0)br[l]=raw;break;}
  }
  if(!count)continue;result[0]=sum/count;result[1]=maxcos;result[2]=bc[0];result[3]=ac[0];int bn=0,an=0;for(int l=0;l<3;l++){if(before[l]<1e99){result[4]+=bc[l];bn++;}if(after[l]<1e99){result[5]+=ac[l];an++;}}
  if(bn)result[4]/=bn;if(an)result[5]/=an;for(int l=0;l<3;l++)result[6+l]=ws[l]>1e-8?vs[l]/ws[l]:0;result[9]=rawsum/count;result[10]=ws[1]>1e-8?raw180/ws[1]:0;result[11]=br[0];
 }
}
