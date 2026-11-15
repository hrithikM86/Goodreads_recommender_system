#include <cmath>
#include <algorithm>
// Two nonnegative elastic-net fits on the same restricted neighborhood.
extern "C" double slim_fit(int n, const double* gram, const double* rhs, double* B) {
 double residual[64];
 for(int a=0;a<n;a++) for(int v=0;v<2;v++) residual[2*a+v]=rhs[a];
 double change=0;
 for(int sweep=0;sweep<100;sweep++) {
  change=0;
  for(int a=0;a<n;a++) for(int v=0;v<2;v++) {
   double old=B[2*a+v];
   double next=std::max(0.,(residual[2*a+v]+gram[a*n+a]*old-(v==0?.2:2.))/(gram[a*n+a]+10.));
   double delta=next-old; B[2*a+v]=next;
   for(int j=0;j<n;j++) residual[2*j+v]-=gram[j*n+a]*delta;
   change=std::max(change,std::abs(delta));
  }
  if(change<1e-5) break;
 }
 return change;
}
