"""Generate Task 1 predictions with zero ratings excluded.
From repository root: python experiments/task1_submission.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from surprise import Dataset, Reader, SVD


def train_predict(data_dir, output=None):
    data_dir = Path(data_dir)
    output = Path(output) if output else data_dir / 'task_1_predictions.csv'
    ratings = pd.read_csv(data_dir / 'tasks_1_2_train.csv')
    ratings = ratings[ratings.rating > 0].reset_index(drop=True)
    test = pd.read_csv(data_dir / 'task_1_test.csv')
    trainset = Dataset.load_from_df(ratings[['user', 'item', 'rating']], Reader(rating_scale=(1, 5))).build_full_trainset()
    prediction = np.zeros(len(test))
    # Weights selected only on the local model-selection split.
    for factors, weight in [(16, .1541547817989482), (32, .16261770829034805), (64, .1196487441069676)]:
        print('Training SVD with', factors, 'factors', flush=True)
        model = SVD(n_factors=factors, n_epochs=50, reg_all=.1, random_state=42)
        model.fit(trainset)
        prediction += weight * np.array([model.predict(u, i).est for u, i in test[['user', 'item']].itertuples(index=False, name=None)])
    u, users = pd.factorize(ratings.user)
    i, items = pd.factorize(ratings.item)
    usermap = dict(zip(users, range(len(users))))
    itemmap = dict(zip(items, range(len(items))))
    r = ratings.rating.to_numpy(dtype=float)
    mu = r.mean(); bu = np.zeros(len(users)); bi = np.zeros(len(items))
    cu = np.bincount(u, minlength=len(users)); ci = np.bincount(i, minlength=len(items))
    for _ in range(20):
        bu = np.bincount(u, weights=r-mu-bi[i], minlength=len(users))/(cu+10)
        bi = np.bincount(i, weights=r-mu-bu[u], minlength=len(items))/(ci+10)
    X = coo_matrix((np.ones(len(r), dtype=np.float32), (u, i)), shape=(len(users), len(items))).tocsr()
    G = X.T.tocsr()
    R = coo_matrix((r-mu-bu[u]-bi[i], (u, i)), shape=X.shape).tocsr()
    qu = np.array([usermap.get(x, -1) for x in test.user])
    qi = np.array([itemmap.get(x, -1) for x in test.item])
    base = np.array([mu+(bu[x] if x>=0 else 0)+(bi[j] if j>=0 else 0) for x,j in zip(qu,qi)])
    settings = [(10, 1, .1062846995076872), (20, 2, .31124904671666187), (10, 2, .1460450195793872)]
    neighbors = [base.copy() for _ in settings]
    groups = {}
    for q, key in enumerate(qi):
        if key >= 0 and qu[q] >= 0:
            groups.setdefault(key, []).append(q)
    keys = np.array(list(groups))
    for offset in range(0, len(keys), 64):
        batch = keys[offset:offset+64]
        overlap = (G[batch] @ X).toarray()
        for row, key in enumerate(batch):
            similarity = overlap[row]/np.sqrt(np.maximum(ci[key]*ci, 1))
            similarity[key] = 0
            for q in groups[key]:
                lo, hi = R.indptr[qu[q]:qu[q]+2]
                w = similarity[R.indices[lo:hi]]
                order = np.argsort(w)[::-1][:20]
                w = w[order]; errors = R.data[lo:hi][order]
                for column, (k, power, weight) in enumerate(settings):
                    weights = w[:k]**power
                    neighbors[column][q] += (weights @ errors[:k])/(weights.sum()+1e-12)
    for values, (k, power, weight) in zip(neighbors, settings):
        prediction += weight*np.clip(values, 1, 5)
    prediction = np.clip(prediction, 1, 5)
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise ValueError('Invalid predictions')
    # The supplied test header already includes the empty rating target.
    test['rating'] = prediction
    test.to_csv(output, index=False)
    print('Saved', len(test), 'predictions to', output, flush=True)
    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('Assignment 1/assignment1_2026'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    train_predict(args.data_dir, args.output)
