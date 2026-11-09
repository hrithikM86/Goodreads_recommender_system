"""Training-only next-book time summaries with exact leave-one-event-out support."""
import numpy as np
NAMES=['book_log_count','book_log_std','book_first_delta','book_mean_delta','book_median_delta','book_last_delta','book_q10_delta','book_q90_delta','book_first_positive_delta','book_median_positive_delta','book_next_observed_gap','book_prev_observed_gap','book_future_fraction','book_log_future_count','book_unknown','last_date_valid']

def build_item_features(item,timestamps,allowed,candidate,last_time,target_row=None):
    count_items=int(item.max()+1);N=len(candidate);good=allowed&(timestamps>=0)&(timestamps<=4102444800)
    rows=np.flatnonzero(good);order=np.lexsort((timestamps[rows],item[rows]));rows=rows[order]
    sorted_item=item[rows];dates=timestamps[rows]/86400
    ptr=np.searchsorted(sorted_item,np.arange(count_items+1))
    sums=np.bincount(sorted_item,weights=dates,minlength=count_items);squares=np.bincount(sorted_item,weights=dates**2,minlength=count_items)
    rank=np.full(len(item),-1,dtype=np.int64);rank[rows]=np.arange(len(rows))
    target_row=np.full(N,-1,dtype=int) if target_row is None else np.asarray(target_row)
    result=np.full((N,len(NAMES)),np.nan);result[:,0]=0;result[:,14]=1;result[:,15]=((last_time>=0)&(last_time<=4102444800)).astype(float)
    groups={}
    for j,key in enumerate(candidate):
        if key>=0:groups.setdefault(key,[]).append(j)
    signed=lambda x:np.sign(x)*np.log1p(abs(x))
    for key,indices in groups.items():
        qs=np.asarray(indices);lo,hi=ptr[key:key+2];A=dates[lo:hi];n=len(A)
        if not n:continue
        removed=np.where(target_row[qs]>=0,rank[np.maximum(target_row[qs],0)]-lo,-1);remove=removed>=0;effective=n-remove
        ok=effective>0
        if not ok.any():continue
        qidx=qs[ok];r=removed[ok];rem=remove[ok];ne=effective[ok];q=last_time[qidx]/86400
        removed_value=A[np.clip(r,0,n-1)]*rem
        mean=(sums[key]-removed_value)/ne
        std=np.sqrt(np.maximum((squares[key]-A[np.clip(r,0,n-1)]**2*rem)/ne-mean**2,0))
        quantiles=[]
        for quant in [0,.1,.5,.9,1]:
            frac=quant*(ne-1);left=np.floor(frac).astype(int);right=np.ceil(frac).astype(int)
            il=left+(rem&(left>=r));ir=right+(rem&(right>=r))
            quantiles.append(A[il]*(1-(frac-left))+A[ir]*(frac-left))
        first,q10,median,q90,last=quantiles
        result[qidx,0]=np.log1p(ne);result[qidx,1]=np.log1p(std);result[qidx,14]=0
        for col,value in [(2,first),(3,mean),(4,median),(5,last),(6,q10),(7,q90)]:result[qidx,col]=signed(value-q)
        result[qidx,8]=np.log1p(np.maximum(first-q,0));result[qidx,9]=np.log1p(np.maximum(median-q,0))
        future=np.searchsorted(A,q,side='left');future+=rem&(future==r)
        past=np.searchsorted(A,q,side='right')-1;past-=rem&(past==r)
        result[qidx,10]=np.where(future<n,np.log1p(np.maximum(A[np.clip(future,0,n-1)]-q,0)),np.nan)
        result[qidx,11]=np.where(past>=0,np.log1p(np.maximum(q-A[np.clip(past,0,n-1)],0)),np.nan)
        after=n-np.searchsorted(A,q,side='right')-(rem&(A[np.clip(r,0,n-1)]>q))
        result[qidx,12]=after/ne;result[qidx,13]=np.log1p(after)
        bad=(last_time[qidx]<0)|(last_time[qidx]>4102444800)
        if bad.any():result[qidx[bad],2:14]=np.nan
    return result.astype(np.float32)
