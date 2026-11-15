#include <cmath>
#include <algorithm>
extern "C" void pair_pearson(int u,int n,const int* candidates,const int* ptr,
 const int* items,const float* ratings,const float* residual,float* raw,float* res,int* counts){
 for(int z=0;z<n;z++){
  int v=candidates[z],a=ptr[u],b=ptr[v],endA=ptr[u+1],endB=ptr[v+1],N=0;double sx=0,sy=0,sxx=0,syy=0,sxy=0,rx=0,ry=0,rxx=0,ryy=0,rxy=0;
  while(a<endA&&b<endB){
   if(items[a]<items[b]){a++;continue;}if(items[b]<items[a]){b++;continue;}
   double x=ratings[a],y=ratings[b],r=residual[a],s=residual[b];sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;rx+=r;ry+=s;rxx+=r*r;ryy+=s*s;rxy+=r*s;N++;a++;b++;
  }
  counts[z]=N;raw[z]=res[z]=0;
  if(N>=3){double vx=sxx-sx*sx/N,vy=syy-sy*sy/N,dx=rxx-rx*rx/N,dy=ryy-ry*ry/N;
   if(vx>1e-10&&vy>1e-10)raw[z]=std::max(-1.,std::min(1.,(sxy-sx*sy/N)/std::sqrt(vx*vy)));
   if(dx>1e-10&&dy>1e-10)res[z]=std::max(-1.,std::min(1.,(rxy-rx*ry/N)/std::sqrt(dx*dy)));
  }
 }
}
