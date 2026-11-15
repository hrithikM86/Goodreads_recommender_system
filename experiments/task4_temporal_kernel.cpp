#include <algorithm>
#include <cmath>
#include <vector>
// Exact two-sided exponential KDE on sorted event dates, in days.
extern "C" void temporal_kde(int n,const double* x,int m,const double* q,double h,double* out) {
 if(n==0){std::fill(out,out+m,0.);return;}
 std::vector<double> left(n),right(n);left[0]=1.;right[n-1]=1.;
 for(int k=1;k<n;k++)left[k]=1.+std::exp(-(x[k]-x[k-1])/h)*left[k-1];
 for(int k=n-2;k>=0;k--)right[k]=1.+std::exp(-(x[k+1]-x[k])/h)*right[k+1];
 for(int k=0;k<m;k++){
  int p=std::lower_bound(x,x+n,q[k])-x;double a=0.,b=0.;
  if(p)a=left[p-1]*std::exp(-(q[k]-x[p-1])/h);
  if(p<n)b=right[p]*std::exp(-(x[p]-q[k])/h);
  out[k]=(a+b)/(2.*h*n);
 }
}
