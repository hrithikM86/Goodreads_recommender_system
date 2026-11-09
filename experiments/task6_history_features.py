import numpy as np

def history_features(t,r,k):
 g=np.diff(t[:k+1])/86400
 logs=np.log1p(g)
 last=list(logs[-10:][::-1]);last += [np.nan]*(10-len(last))
 out=last+[len(g),np.log1p(g.sum()),t[k]/86400, np.mean(r[:k+1]),r[k],np.mean(r[max(0,k-4):k+1]),np.mean(np.array(r[:k+1])==0)]
 for n in [0,3,5,10,20]:
  v=g if n==0 else g[-n:]
  q=np.quantile(v,[.1,.25,.5,.75,.9])
  out.extend(np.log1p(q));out.extend([np.log1p(v.mean()),np.log1p(v.std()),np.mean(v==0)])
 for a in [.1,.3,.7]:
  v=logs[0]
  for x in logs[1:]:v=a*x+(1-a)*v
  out.append(v)
 # Seasonal signals avoid converting anomalous dates through datetime.
 for period in [7,365.25]:
  angle=2*np.pi*(t[k]/86400 % period)/period
  out.extend([np.sin(angle),np.cos(angle)])
 return out

def basic_features(times,k):
    # Predict gap k (times[k+1]-times[k]); only times through k are known.
    gaps=np.diff(times[:k+1]);recent=gaps[-5:]
    last=list(gaps[-5:][::-1]);last += [np.nan]*(5-len(last))
    return last+[len(gaps),float(np.median(gaps)),float(np.mean(gaps)),float(np.std(gaps)),float(np.min(gaps)),float(np.max(gaps)),float(np.quantile(gaps,.25)),float(np.quantile(gaps,.75)),float(np.median(recent)),float(np.mean(recent)),float(np.std(recent)),float(times[k]),float(times[k]%604800),float(times[k]-times[0])]

