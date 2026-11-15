extern "C" void paths(int n,const int* src,const int* dst,const int* rp,const int* ri,const float* rv,const int* cp,const int* ci,const float* cv,float* out){
 for(int z=0;z<n;z++){int s=src[z],t=dst[z];double total=0;if(s<0||t<0){out[z]=0;continue;}int a=rp[s],ae=rp[s+1],b=cp[t],be=cp[t+1];while(a<ae&&b<be){if(ri[a]<ci[b])a++;else if(ri[a]>ci[b])b++;else{total+=double(rv[a])*cv[b];a++;b++;}}out[z]=total;}
}
