#include <algorithm>
#include <cmath>
extern "C" void score(int n,int k,const int*ptr,const int*hist,const int*qi,const float*E,const float*O,const float*V,float*out){
 for(int r=0;r<n;r++){
  float*z=out+r*8;std::fill(z,z+8,0.f);int j=qi[r];if(j<0)continue;float norm=0;for(int f=0;f<k;f++)norm+=V[j*k+f]*V[j*k+f];if(norm==0)continue;
  int lo=ptr[r],hi=ptr[r+1],start=std::max(lo,hi-20);float cw[3]={0,0,0},cs[3]={0,0,0},rs[3]={0,0,0},maxcos=-2;int count=0;
  for(int h=start;h<hi;h++){
   int i=hist[h];if(i==j)continue;float valid=0,c=0,raw=0;for(int f=0;f<k;f++){valid+=V[i*k+f]*V[i*k+f];c+=V[i*k+f]*V[j*k+f];raw+=E[i*k+f]*O[j*k+f];}if(valid==0)continue;
   int distance=hi-1-h;float w=std::pow(.9f,distance);cs[2]+=w*c;rs[2]+=w*raw;cw[2]+=w;
   if(distance<10){cs[1]+=c;rs[1]+=raw;cw[1]++;}if(distance<3){cs[0]+=c;rs[0]+=raw;cw[0]++;}if(distance==0){z[0]=c;z[5]=raw;}maxcos=std::max(maxcos,c);count++;
  }
  z[1]=cw[0]?cs[0]/cw[0]:0;z[2]=cw[1]?cs[1]/cw[1]:0;z[3]=cw[2]?cs[2]/cw[2]:0;z[4]=count?maxcos:0;z[6]=cw[0]?rs[0]/cw[0]:0;z[7]=cw[2]?rs[2]/cw[2]:0;
 }
}
