#!/usr/bin/env python3
# Assignment 1 | Hrithik Mhatre | TSN: 200478215
# HOW TO RUN:
# 1. Put this file beside the supplied CSVs, or in the sibling Submission folder.
# 2. Install a C++17 compiler: macOS: xcode-select --install
#    macOS libraries: brew install libomp cmake
# 3. conda create -n goodreads_final python=3.12 -y
#    conda activate goodreads_final
# 4. python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 scikit-surprise==1.1.5 implicit==0.7.3 catboost==1.2.10 lightgbm==4.7.0 xgboost==3.4.1
# 5. python assignment1.py
# Optional: python assignment1.py --data-dir "/path/to/assignment1_2026" --tasks 2 4 5
# The fixed final ensembles train from raw CSVs. No parameter search is performed.
# Historical folds are needed to train the stacking models without target leakage.
# A private .assignment1_cache stores generated source, factors and checkpoints.
# First training is expensive; allow 20 GB disk and preferably 16 GB RAM.
# All supporting source/kernels are included below: no repository is required.
# CODE LOCATION (main):
# https://github.com/hrithikM86/Goodreads_recommender_system/blob/main/Assignment%201/Submission/assignment1.py
# EXPERIMENTS (Experiments branch):
# https://github.com/hrithikM86/Goodreads_recommender_system/tree/Experiments/experiments
# Submitted CSVs are canonical. Refit computations can vary across compiler/BLAS
# implementations; pinned reference packages and Linux/GCC give closest replay.
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import hashlib
import copy
import ctypes
import gzip
import io
import pickle
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import Ridge
from surprise import Dataset, Reader, SVD, SVDpp

def data_directory(explicit):
    here = Path(__file__).resolve().parent
    choices = [explicit] if explicit else [here, Path.cwd(), here.parent/'assignment1_2026', here/'assignment1_2026', here/'Assignment 1/assignment1_2026']
    for path in choices:
        path = Path(path).expanduser().resolve()
        if (path/'tasks_1_2_train.csv').is_file():
            return path
    raise FileNotFoundError('Use --data-dir to specify the folder containing the supplied CSVs.')

def prepare(cache, data):
    work, source, output = cache/'work', cache/'source', cache/'output'
    for path in [work, source, output]:
        path.mkdir(parents=True, exist_ok=True)
    for name, text in MODEL_SOURCE.items():
        for old, new in [('/workspace/work/Goodreads_recommender_system/Assignment 1/assignment1_2026', str(data)), ('/workspace/output', str(output)), ('/workspace/work', str(work))]:
            text = text.replace(old, new)
        (source/name).write_text(text, encoding='utf-8')
    links = work/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
    links.mkdir(parents=True, exist_ok=True)
    names = ['tasks_1_2_train.csv','tasks_3_4_train.csv','task_5_train.csv','task_6_train.csv']+[f'task_{t}_test.csv' for t in range(1,7)]
    for name in names:
        if (data/name).exists() and not (links/name).exists():
            try:
                (links/name).symlink_to(data/name)
            except OSError:
                shutil.copy2(data/name, links/name)
    env = os.environ.copy()
    env.update(ASSIGNMENT_WORK_DIR=str(work), ASSIGNMENT_DATA_DIR=str(data), ASSIGNMENT_INTERNAL_OUTPUT=str(output), TASK3_DATA_DIR=str(data), TASK3_ARTIFACTS_DIR=str(work/'task3'), SURPRISE_DATA_FOLDER=str(work/'surprise_data'))
    env['PYTHONPATH'] = str(source)+os.pathsep+env.get('PYTHONPATH','')
    env['CXX'] = env.get('CXX') or shutil.which('g++') or shutil.which('clang++') or shutil.which('c++') or ''
    if not env['CXX']:
        raise RuntimeError('A C++17 compiler is required; see the commands at the top.')
    manifest={n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in names if (data/n).exists()}
    guard=cache/'input_manifest.json'
    if guard.exists() and json.loads(guard.read_text())!=manifest:
        raise ValueError('Raw data differ from cache inputs; select a fresh --cache-dir.')
    guard.write_text(json.dumps(manifest,indent=2))
    return source, output, env

def save_output(task, generated, destination, data):
    q = pd.read_csv(data/f'task_{task}_test.csv')
    p = pd.read_csv(generated)
    assert p.columns.tolist() == q.columns.tolist()
    pd.testing.assert_frame_equal(p.iloc[:,:-1], q.iloc[:,:-1])
    assert not p.iloc[:,-1].isna().any()
    if task in [2,4,5]:
        assert ((p.which==p.item1)|(p.which==p.item2)).all()
    else:
        v = p.iloc[:,-1].to_numpy(float)
        assert np.isfinite(v).all()
        assert ((v>=1)&(v<=5)).all() if task in [1,3] else (v>=0).all()
    shutil.copy2(generated, destination)

def main():
    parser = argparse.ArgumentParser(description='Train the six fixed final Goodreads ensembles.')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--tasks', type=int, nargs='+', choices=range(1,7), default=list(range(1,7)))
    a = parser.parse_args()
    data = data_directory(a.data_dir)
    source, output, env = prepare((a.cache_dir or data/'.assignment1_cache').resolve(), data)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        print(f'Task {task}: training the selected final ensemble', flush=True)
        subprocess.run([sys.executable, str(source/'final_support.py'), '--task', str(task)], env=env, cwd=source, check=True)
        name = f'task_{task}_predictions.csv'
        save_output(task, output/'best'/name, a.output_dir/name, data)
        print('Saved', a.output_dir/name, flush=True)

# Fixed-model implementations (grouped by task) and required native kernels.
# Readable source is materialized automatically so this remains a single file.

MODEL_SOURCE = {
    # SHARED RUNTIME AND FIXED SETTINGS
    'final_imports.py': r'''import argparse
import ctypes
import fcntl
import gc
import hashlib
import importlib
import json
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import pickle
import subprocess
import sys
import time
from catboost import CatBoostClassifier, CatBoostRegressor
from concurrent.futures import ThreadPoolExecutor
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from lightgbm import LGBMRegressor
from pathlib import Path
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csr_matrix, diags, load_npz, save_npz
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from surprise import Dataset, Reader, SVD, SVDpp
from types import SimpleNamespace

import copy
import gzip
import io
import xgboost as xgb
''',
    'final_support.py': r'''from final_imports import *
"""Internal source bundled into assignment1.py; fixed accepted-model refits."""
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ['ASSIGNMENT_WORK_DIR'])
DATA = Path(os.environ['ASSIGNMENT_DATA_DIR'])
OUTPUT = Path(os.environ['ASSIGNMENT_INTERNAL_OUTPUT'])
SETTINGS = json.loads((HERE / 'selected_settings.json').read_text())
def mod(name):
    return importlib.import_module(name)
def folder(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
def run(script, *args):
    cmd = [sys.executable, str(HERE / (script + '.py')), *map(str, args)]
    print('RUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=os.environ.copy())
def cached_model(path, factory, X, y, **kwargs):
    if path.exists():
        model = factory()
        model.load_model(str(path))
    else:
        model = factory()
        model.fit(X, y, **kwargs)
        model.save_model(str(path))
    return model
def write_prediction(task, values, destination):
    q = pd.read_csv(DATA / f'task_{task}_test.csv')
    if task in (1, 3):
        q['rating'] = np.clip(values, 1, 5)
    elif task == 6:
        q['delta'] = np.maximum(values, 0)
    else:
        q['which'] = np.where(np.asarray(values) >= 0, q.item1, q.item2)
    assert len(q) == len(values)
    folder(destination.parent)
    q.to_csv(destination, index=False)
    return destination
def task1_components():
    legacy = WORK
    if not (legacy / 'task1_round1_predictions.npz').exists():
        run('task1_round1')
    original = folder(OUTPUT / 'task1_original') / 'task_1_predictions.csv'
    previous = folder(WORK / 'task1_improved')
    for full in (False, True):
        for kind in ('library', 'neighbors'):
            run('task1_improve', '--data-dir', DATA, '--legacy', legacy, '--out', previous, '--worker', kind, *(['--full'] if full else []))
    if not original.exists():
        mod('task1_submission').train_predict(DATA, original, previous / 'full/neighbors.npz')
    return (original, previous)
# TASK 1: final fit and prediction
def task1():
    original, previous = task1_components()
    improved = mod('task1_improve_stack')
    cfg = SETTINGS['task1']['improved']
    a = SimpleNamespace(data_dir=DATA, legacy=WORK, out=previous, accepted=original)
    X, names, ref, q, n = improved.features(a)
    y = q.rating.to_numpy()
    FX, fn, old_full, test, _ = improved.features(a, True)
    assert names == fn
    pp = 0
    for key, weight in cfg['weights'].items():
        r = cfg['runs'][key]
        path = previous / (key + '_full.cbm')
        factory = lambda r=r: CatBoostRegressor(**improved.cfg(r['depth'], r['iterations']))
        m = cached_model(path, factory, X, y)
        pp = pp + weight * np.clip(m.predict(FX), 1, 5)
    improved_csv = write_prediction(1, np.clip(pp, 1, 5), previous / 'new/task_1_predictions.csv')
    (previous / 'results.json').write_text(json.dumps(cfg, indent=2))
    del X, FX
    gc.collect()
    reference = folder(WORK / 'task1_next')
    for full in (False, True):
        for kind in ('svdpp', 'svdppstrong', 'itemspp', 'neighbors'):
            run('task1_next', '--data-dir', DATA, '--out', reference, '--worker', kind, *(['--full'] if full else []))
    nextmod = mod('task1_next_stack')
    a = SimpleNamespace(data_dir=DATA, legacy=WORK, previous=previous, out=reference, original=original)
    X, names, q, n = nextmod.matrices(a)
    FX, fn, test, _ = nextmod.matrices(a, True)
    assert names == fn
    y = q.rating.to_numpy()
    cfg = SETTINGS['task1']['next']
    raw = 0
    for key, weight in cfg['depth_weights'].items():
        if not weight:
            continue
        r = cfg['runs'][key]
        factory = lambda r=r: CatBoostRegressor(**improved.cfg(r['depth'], r['iterations']))
        m = cached_model(reference / (key + '_full.cbm'), factory, X, y)
        raw = raw + weight * np.clip(m.predict(FX), 1, 5)
    older = pd.read_csv(improved_csv).rating.to_numpy()
    pp = cfg['new_weight'] * raw + (1 - cfg['new_weight']) * older
    reference_csv = write_prediction(1, pp, reference / 'primary/task_1_predictions.csv')
    (reference / 'results.json').write_text(json.dumps(cfg, indent=2))
    del X, FX
    gc.collect()
    out = folder(WORK / 'task1_neural')
    for full in (False, True):
        run('task1_neural_cf', '--data-dir', DATA, '--previous', previous, '--out', out, *(['--full'] if full else []))
    neural = mod('task1_neural')
    a = SimpleNamespace(data_dir=DATA, legacy=WORK, previous=previous, reference=reference, original=original, out=out)
    X, names, q, n = neural.inputs(a)
    FX, fn, test, _ = neural.inputs(a, True)
    assert names == fn
    X = neural.transform(X, names)
    FX = neural.transform(FX, names)
    y = q.rating.to_numpy(dtype='f')
    reference_values = pd.read_csv(reference_csv).rating.to_numpy()
    predictions = {'accepted': reference_values}
    cfg = SETTINGS['task1']['neural']
    for name, r in cfg['runs'].items():
        path = out / (name + '_full.npz')
        if path.exists():
            m = neural.Neural.load(path)
        else:
            m = neural.Neural([X.shape[1]] + r['hidden'] + [5], r['seed'], r['lam'])
            m.fit(X, y, r['epochs'])
            m.save(path)
        predictions[name] = m.probabilities(FX) @ neural.RATINGS
    weights = SETTINGS['task1']['neural_weights']
    primary = sum((weights[k] * v for k, v in predictions.items()))
    conservative = 0.5 * primary + 0.5 * reference_values
    return write_prediction(1, conservative, OUTPUT / 'best/task_1_predictions.csv')
# TASK 2: final fit and prediction
def task2_original():
    m = mod('task2_optimize')
    root = folder(WORK / 'task2_optimized/with_zero')
    p = pd.read_csv(DATA / 'tasks_1_2_train.csv')
    u, users = pd.factorize(p.user)
    i, items = pd.factorize(p.item)
    order = np.random.RandomState(42).permutation(len(p))
    a, b = (int(0.9 * len(p)), int(0.95 * len(p)))
    fit, hold = (order[:a], order[a:])
    train = np.zeros(len(p), bool)
    train[fit] = True
    X = m.matrix(u, i, train, (len(users), len(items)))
    rng = np.random.RandomState(43)
    eligible = np.flatnonzero(train)
    negative = i[eligible[rng.randint(len(eligible), size=len(hold))]]
    equal = negative == i[hold]
    while equal.any():
        negative[equal] = i[eligible[rng.randint(len(eligible), size=equal.sum())]]
        equal = negative == i[hold]
    names, f = m.features(X, np.r_[u[hold], u[hold]], np.r_[i[hold], negative], folder(root / 'validation'))
    n = len(hold)
    d, s = (f[:n] - f[n:], f[:n] + f[n:])
    cut = b - a
    cfg = SETTINGS['task2']
    factory = lambda: CatBoostClassifier(iterations=cfg['iterations'], depth=cfg['depth'], learning_rate=0.035, l2_leaf_reg=10, loss_function='Logloss', random_seed=42, thread_count=2, verbose=100, allow_writing_files=False)
    model = cached_model(root / 'validation/stacker.cbm', factory, np.r_[m.design(d[:cut], s[:cut]), m.design(-d[:cut], s[:cut])], np.r_[np.ones(cut), np.zeros(cut)])
    del f, d, s, X
    gc.collect()
    q = pd.read_csv(DATA / 'task_2_test.csv')
    qu = pd.Index(users).get_indexer(q.user)
    qi = np.r_[pd.Index(items).get_indexer(q.item1), pd.Index(items).get_indexer(q.item2)]
    assert (qu >= 0).all() and (qi >= 0).all()
    X = m.matrix(u, i, np.ones(len(p), bool), (len(users), len(items)))
    fn, f = m.features(X, np.tile(qu, 2), qi, folder(root / 'full'))
    assert fn == names
    n = len(q)
    margin = m.predict(model, f[:n] - f[n:], f[:n] + f[n:])
    return write_prediction(2, margin, OUTPUT / 'best/task_2_predictions.csv')
def task3_components():
    legacy = folder(WORK / 'task3')
    env_art = os.environ.get('TASK3_ARTIFACTS_DIR')
    os.environ['TASK3_ARTIFACTS_DIR'] = str(legacy)
    os.environ['TASK3_DATA_DIR'] = str(DATA)
    for name, output in [('experiment', 'predictions.npz'), ('grid', 'grid_predictions.npz'), ('neighbors', 'neighbors_predictions.npz'), ('kernels', 'kernel_predictions.npz')]:
        if not (legacy / output).exists():
            run('task3_' + name)
    (legacy / 'meta_results.json').write_text(json.dumps({'base_columns': SETTINGS['task3']['base_columns']}))
    prior = folder(WORK / 'task3_optimized')
    round2 = folder(WORK / 'task3_round2')
    out = folder(WORK / 'task3_round3')
    opt = mod('task3_optimize')
    r3 = mod('task3_round3')
    for full in (False, True):
        part = folder(prior / ('full' if full else 'validation'))
        for kind in ('library', 'temporal'):
            run('task3_optimize', '--data-dir', DATA, '--output-dir', part, '--worker', kind, *(['--full'] if full else []))
        if full:
            run('task3_optimize', '--data-dir', DATA, '--output-dir', part, '--worker', 'baseline')
        for kind in ('joint', 'raw', 'learned', 'learnedwide', 'pearson'):
            run('task3_round2', '--data-dir', DATA, '--output-dir', folder(round2 / ('full' if full else 'validation')), '--prior-dir', prior, '--baseline-dir', legacy, '--worker', kind, *(['--full'] if full else []))
        r3.user_ridge(DATA, prior, folder(out / ('full' if full else 'validation')), full)
        r3.ordinal_ridge(DATA, prior, round2, folder(out / ('full' if full else 'validation')), full)
    return (legacy, prior, round2, out)
# TASK 3: final fit and prediction
def task3_original():
    legacy, prior, round2, out = task3_components()
    r3 = mod('task3_round3')
    X, anchor, names = r3.extended_design(DATA, legacy, prior, round2, out)
    FX, fa, fn = r3.extended_design(DATA, legacy, prior, round2, out, True)
    assert names == fn
    p, tr, q, *_ = r3.dataset(DATA, False)
    y = q.rating.to_numpy()
    pred = 0
    for r in SETTINGS['task3']['models']:
        factory = lambda r=r: CatBoostRegressor(iterations=r['iterations'], depth=r['depth'], learning_rate=0.025, l2_leaf_reg=20, random_seed=42, thread_count=2, loss_function='RMSE', allow_writing_files=False, verbose=100)
        model = cached_model(out / r['filename'], factory, X, y)
        pred = pred + r['weight'] * np.clip(model.predict(FX), 1, 5)
    return write_prediction(3, np.clip(pred, 1, 5), OUTPUT / 'best/task_3_predictions.csv')
def task4_components():
    root = folder(WORK / 'task4_two_approaches')
    for source, target in [('task4_pair_fm.cpp', 'pair_fm.so'), ('task4_drift.cpp', 'drift.so')]:
        if not (root / target).exists():
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O3', '-shared', '-fPIC', str(HERE / source), '-o', str(root / target)], check=True)
    m = mod('task4_two_approaches')
    for full in (False, True):
        m.fresh_stage(full)
        m.fm(full)
        path = WORK / 'task4_item2vec' / ('full_features.npz' if full else 'validation_features.npz')
        if not path.exists():
            run('task4_item2vec', *(['--full'] if full else []))
    return root
# TASK 4: final fit and prediction
def task4():
    task4_components()
    tree = mod('task4_tree_combiners')
    def matrices(full):
        d, s = tree.matrices(full)
        f = np.load(WORK / 'task4_item2vec' / ('full_features.npz' if full else 'validation_features.npz'))
        values = np.column_stack([f[k] for k in f.files])
        n = len(values) // 2
        assert n == len(d)
        return (np.c_[d, values[:n] - values[n:]].astype('f'), np.c_[s, values[:n] + values[n:]].astype('f'))
    d, s = matrices(False)
    cfg = SETTINGS['task4']
    n = 71462
    path = WORK / 'task4_item2vec/lightgbm.txt'
    lgb = mod('lightgbm')
    if path.exists():
        model = lgb.Booster(model_file=str(path), params={'num_threads': 2})
    else:
        model = tree.fit('lightgbm', cfg['params'], cfg['iterations'], np.r_[np.c_[d[:n], s[:n]], np.c_[-d[:n], s[:n]]], np.r_[np.ones(n), np.zeros(n)])
        model.booster_.save_model(str(path))
    del d, s
    gc.collect()
    fd, fs = matrices(True)
    if isinstance(model, lgb.Booster):
        margin = model.predict(np.c_[fd, fs]) - model.predict(np.c_[-fd, fs])
    else:
        margin = tree.predict(model, 'lightgbm', fd, fs)
    return write_prediction(4, margin, OUTPUT / 'best/task_4_predictions.csv')
def task5_components():
    baseline = folder(WORK / 'task5/results')
    if not (baseline / 'test_pair_features.npz').exists():
        run('task5_ensemble', '--data-dir', DATA, '--output-dir', baseline, '--predict')
    previous = folder(WORK / 'task5_improved')
    nextdir = folder(WORK / 'task5_next')
    current = folder(WORK / 'task5_round3')
    for full in (False, True):
        flags = ['--full'] if full else []
        for kind in ('fpmc', 'uniform', 'neighbors'):
            run('task5_improve', '--data-dir', DATA, '--baseline', baseline, '--out', previous, '--worker', kind, *flags)
        for name in ('task5_next', 'task5_ease_next', 'task5_fossil_next'):
            run(name, '--data-dir', DATA, '--baseline', baseline, '--out', nextdir, *flags)
        for name in ('task5_order', 'task5_era', 'task5_attention', 'task5_bpr_round3', 'task5_fold'):
            argv = ['--data-dir', DATA, '--baseline', baseline, '--out', current]
            if name in ('task5_attention', 'task5_fold'):
                argv += ['--previous', previous]
            run(name, *argv, *flags)
    for kind in ('base', 'fpmc', 'attention', 'uniform', 'fold', 'neighbors', 'metrics', 'ease', 'fossil', 'order', 'era', 'bpr'):
        run('task5_rolling', '--worker', kind)
    return (baseline, previous, nextdir, current)
# TASK 5: final fit and prediction
def task5():
    baseline, previous, nextdir, current = task5_components()
    rolling = mod('task5_rolling')
    pref = mod('task5_preference_round3')
    neural = mod('task5_round3')
    a = SimpleNamespace(data_dir=DATA, baseline=baseline, previous=previous, reference=nextdir, out=current)
    ed, es, names = rolling.refreshed()
    d, s, parts, extra = pref.load(a)
    n = len(d) // 2
    sd, ss = (d[:n].copy(), s[:n].copy())
    del d, s, parts
    gc.collect()
    draw = np.random.RandomState(418).randint(3, size=n) * n + np.arange(n)
    td = np.r_[sd, ed[draw]]
    ts = np.r_[ss, es[draw]]
    del ed, es, sd, ss
    gc.collect()
    out = folder(WORK / 'task5_rolling')
    models = {}
    for key, width, seed, epochs in [('sequence', 224, 71, 16), ('preference', 260, 73, 8)]:
        path = out / (key + '_production.pkl')
        if path.exists():
            model = pickle.loads(path.read_bytes())
        else:
            model, _ = neural.nn_fit(td[:, :width], ts[:, :width], epochs, seed=seed, hidden=(64, 32), alpha=0.1)
            path.write_bytes(pickle.dumps(model, protocol=5))
        models[key] = model
    del td, ts
    gc.collect()
    fd, fs, parts, extra = pref.load(a, True)
    margin = 0
    for key, width in [('sequence', 224), ('preference', 260)]:
        margin = margin + 0.5 * neural.nn_margin(*models[key], fd[:, :width], fs[:, :width])
    return write_prediction(5, margin, OUTPUT / 'best/task_5_predictions.csv')
def task6_legacy_features():
    pipe = mod('task6_pipeline')
    history = mod('task6_history_features')
    time_model = mod('task6_item_time')
    root = folder(WORK / 'task6_final')
    p, test, u, users, i, items, t, ratings, starts, early, v, e, rows = pipe.load_data(DATA)
    n, m = (len(early), len(v))
    richpath = WORK / 'task6_rich.npz'
    if not richpath.exists():
        core = pipe.core_matrix(t, ratings, u, starts, rows)
        y = (t[rows] - t[rows - 1]) / 86400
        np.savez_compressed(richpath, X=core[:n], V=core[n:n + m], T=core[n + m:], y=y[:n], vy=y[n:n + m], ty=y[n + m:])
    z = np.load(richpath)
    core = np.r_[z['X'], z['V'], z['T']]
    y = np.r_[z['y'], z['vy'], z['ty']]
    if not (root / 'book_features.npz').exists():
        allowed = np.ones(len(p), bool)
        allowed[v] = False
        allowed[e] = False
        book = time_model.build_item_features(i, t, allowed, i[rows], t[rows - 1], rows)
        np.savez_compressed(root / 'book_features.npz', X=book[:n], V=book[n:n + m], T=book[n + m:], train_rows=early, selection_rows=v, evaluation_rows=e, all_rows=rows)
    if not (root / 'full_features.npz').exists():
        book = time_model.build_item_features(i, t, np.ones(len(p), bool), i[rows], t[rows - 1], rows)
        query_core, query_basic, last = ([], [], [])
        for user in pd.Index(users).get_indexer(test.user):
            lo, hi = starts[user:user + 2]
            query_core.append(history.history_features(t[lo:hi], ratings[lo:hi], int(hi - lo - 1)))
            query_basic.append(history.basic_features(t[lo:hi], int(hi - lo - 1)))
            last.append(t[hi - 1])
        qi = pd.Index(items).get_indexer(test.item)
        query_book = time_model.build_item_features(i, t, np.ones(len(p), bool), qi, np.asarray(last))
        np.savez_compressed(root / 'full_features.npz', book=book, query_book=query_book, query_rich=np.asarray(query_core, 'f'), query_basic=np.asarray(query_basic, 'f'))
    return (p, test, rows, core, y, n, m)
# TASK 6: final fit and prediction
def task6():
    p, test, rows, core, y, n, m = task6_legacy_features()
    pipe = mod('task6_pipeline')
    imp = mod('task6_improve_models')
    root = WORK / 'task6_final'
    cfg = SETTINGS['task6']['base']
    books = np.load(root / 'book_features.npz')
    full = np.load(root / 'full_features.npz')
    foldbook = np.r_[books['X'], books['V'], books['T']]
    X = pipe.frame(core[:n], foldbook[:n], p, rows=rows[:n])
    V = pipe.frame(core[n:], foldbook[n:], p, rows=rows[n:])
    F = pipe.frame(core, full['book'], p, rows=rows)
    Q = pipe.frame(full['query_rich'], full['query_book'], p, query=test)
    old_fold = {}
    old_test = {}
    for name, spec in cfg['models'].items():
        def factory(spec=spec):
            return CatBoostRegressor(loss_function=spec['loss'], iterations=spec['iterations'], depth=spec['depth'], learning_rate=0.05, l2_leaf_reg=3, random_seed=spec['seed'], thread_count=2, verbose=200, allow_writing_files=False, cat_features=['user', 'next_item'], max_ctr_complexity=1)
        model = cached_model(root / (name + '.cbm'), factory, X, np.log1p(y[:n]) if spec['log'] else y[:n], eval_set=(V.iloc[:m], np.log1p(y[n:n + m]) if spec['log'] else y[n:n + m]), use_best_model=False)
        pred = model.predict(V)
        if spec['log']:
            pred = np.expm1(pred)
        old_fold[name] = np.maximum(0, pred)
        model = cached_model(folder(root / 'full_models') / (name + '.cbm'), factory, F, np.log1p(y) if spec['log'] else y)
        pred = model.predict(Q)
        if spec['log']:
            pred = np.expm1(pred)
        old_test[name] = np.maximum(0, pred)
    xr = np.expm1(core[:, [0, 19, 22]]).astype(float)
    ridge = Ridge(alpha=1.0).fit(xr[:n], y[:n])
    old_fold['ridge'] = np.maximum(0, ridge.predict(xr[n:]))
    ridge = Ridge(alpha=1.0).fit(xr, y)
    old_test['ridge'] = np.maximum(0, ridge.predict(np.expm1(full['query_rich'][:, [0, 19, 22]]).astype(float)))
    fold_anchor = sum((cfg['weights'][k] * old_fold[k] for k in cfg['weights']))
    test_anchor = sum((cfg['weights'][k] * old_test[k] for k in cfg['weights']))
    old_csv = write_prediction(6, test_anchor * 86400, root / 'task_6_predictions.csv')
    del X, V, F, Q, full, foldbook
    gc.collect()
    previous = folder(WORK / 'task6_improved')
    run('task6_improve', '--data-dir', DATA, '--legacy', WORK, '--out', previous, '--full')
    val = np.load(previous / 'validation.npz')
    complete = np.load(previous / 'full.npz')
    settings = SETTINGS['task6']['improved']
    predv, predt, predq = ({}, {}, {})
    for name in ('balanced', 'cat'):
        count = settings['runs'][name]['iterations']
        XX = val['X'][:n]
        QQ = val['X'][n:]
        weights = imp.user_weights(p, val['rows'][:n]) if name == 'balanced' else None
        if name == 'cat':
            XX = imp.framing(XX, p, val['rows'][:n])
            QQ = imp.framing(QQ, p, val['rows'][n:])
        path = previous / (name + '_validation.model')
        if path.exists():
            if name == 'cat':
                model = CatBoostRegressor()
                model.load_model(str(path))
            else:
                model = mod('lightgbm').Booster(model_file=str(path), params={'num_threads': 1})
        else:
            model = imp.model(name, count)
            if name == 'balanced':
                model.fit(XX, y[:n], sample_weight=weights)
            else:
                model.fit(XX, y[:n], eval_set=(QQ.iloc[:m], y[n:n + m]), use_best_model=False)
            if name == 'cat':
                model.save_model(str(path))
            else:
                model.booster_.save_model(str(path))
        forecast = np.maximum(0, model.predict(QQ))
        predv[name], predt[name] = (forecast[:m], forecast[m:])
        a = SimpleNamespace(out=previous)
        predq[name] = imp.full_fit(a, name, count, p, complete, val['progress'], test)
    all_fold = {'accepted': fold_anchor, 'lgb': np.zeros(len(fold_anchor)), 'recent': np.zeros(len(fold_anchor)), **{name: np.r_[predv[name], predt[name]] for name in ('balanced', 'cat')}}
    older_test = pd.read_csv(old_csv).delta.to_numpy() / 86400
    all_test = {'accepted': older_test, 'lgb': np.zeros(len(test)), 'recent': np.zeros(len(test)), **predq}
    improved_fold = sum((weight * all_fold[name] for name, weight in settings['blend_weights'].items()))
    improved_test = sum((weight * all_test[name] for name, weight in settings['blend_weights'].items()))
    csv = write_prediction(6, improved_test * 86400, previous / 'blend/task_6_predictions.csv')
    out = folder(WORK / 'task6_next')
    for fullflag in (False, True):
        run('task6_next', '--data-dir', DATA, '--previous', previous, '--out', out, *(['--full'] if fullflag else []))
    peer = np.load(out / 'validation_peer.npz')['X']
    full_peer = np.load(out / 'full_peer.npz')
    anchor = pd.read_csv(csv).delta.to_numpy() / 86400
    return task6_quantile_finish(val,complete,peer,full_peer,y,improved_fold,anchor,n)

# TASK 2 UPDATE: selected XGBoost/CatBoost interaction ensemble.
def pair_design(d,s):
    return np.c_[d,s,abs(d)].astype('f')
def pair_xgb(count):
    return xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_estimators=count,max_depth=5,min_child_weight=50,learning_rate=.035,reg_lambda=20.,reg_alpha=.1,max_bin=127,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=2)
def rating_xgb(count):
    return xgb.XGBRegressor(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',n_estimators=count,max_depth=5,learning_rate=.03,min_child_weight=100,reg_lambda=30.,reg_alpha=.1,subsample=.9,colsample_bytree=.9,max_bin=127,n_jobs=2,random_state=42)
def fixed_xgb(path,factory,X,y):
    model=factory()
    if path.exists():
        if path.suffix=='.gz':
            with gzip.open(path,'rb') as f:model.load_model(bytearray(f.read()))
        else:model.load_model(path)
    else:
        model.fit(X,y,verbose=False);folder(path.parent)
        if path.suffix=='.gz':
            with gzip.open(path,'wb') as f:f.write(model.get_booster().save_raw(raw_format='ubj'))
        else:model.save_model(path)
    return model
def fixed_lgb(path,factory,X,y):
    if path.exists():return lgb.Booster(model_file=str(path))
    model=factory();model.fit(X,y);folder(path.parent);model.booster_.save_model(str(path));return model

def task2():
    task2_original()
    root=WORK/'task2_optimized/with_zero'
    def pairs(stage):
        z=np.load(root/stage/'features.npz');F=np.column_stack([z[k] for k in z.files]).astype('f');n=len(F)//2
        return F[:n]-F[n:],F[:n]+F[n:]
    d,s=pairs('validation');X=np.r_[pair_design(d,s),pair_design(-d,s)];y=np.r_[np.ones(len(d)),np.zeros(len(d))]
    model=fixed_xgb(WORK/'task2_xgboost_refresh/production.ubj',lambda:pair_xgb(1041),X,y)
    d,s=pairs('full');new=(model.predict_proba(pair_design(d,s))[:,1]-model.predict_proba(pair_design(-d,s))[:,1])/2
    old=CatBoostClassifier();old.load_model(str(root/'validation/stacker.cbm'))
    previous=(old.predict_proba(np.c_[d,s])[:,1]-old.predict_proba(np.c_[-d,s])[:,1])/2
    weight=SETTINGS['task2_update']['weight']
    return write_prediction(2,weight*new+(1-weight)*previous,OUTPUT/'best/task_2_predictions.csv')

# TASK 3 UPDATE: temporal scaling,128-factor SVD++, local-linear smoothing.
def rating_checkpoint(label,values):
    path=write_prediction(3,values,OUTPUT/label/'task_3_predictions.csv')
    return pd.read_csv(path).rating.to_numpy()
def task3():
    prior_csv=task3_original();previous=pd.read_csv(prior_csv).rating.to_numpy()
    r3=mod('task3_round3');cf=mod('task3_refresh_features')
    legacy,prior,round2,out=(WORK/'task3',WORK/'task3_optimized',WORK/'task3_round2',WORK/'task3_round3')
    X,_,names=r3.extended_design(DATA,legacy,prior,round2,out)
    FX,_,fn=r3.extended_design(DATA,legacy,prior,round2,out,True)
    _,train,q,*_=r3.dataset(DATA,False);_,full,test,*_=r3.dataset(DATA,True);y=q.rating.to_numpy()
    X,names=cf.augment(X,list(map(str,names)),train,q);FX,fn=cf.augment(FX,list(map(str,fn)),full,test);assert names==fn
    model=fixed_lgb(WORK/'task3_lightgbm_refresh/production.txt',lambda:cf.estimator(278),X,y)
    weight=SETTINGS['task3_update']['lightgbm_weight'];previous=rating_checkpoint('task3_lightgbm_blend',weight*clip_rating(model.predict(FX))+(1-weight)*previous)
    model=fixed_xgb(WORK/'task3_xgboost_refresh/production.ubj',lambda:rating_xgb(720),X,y)
    weight=SETTINGS['task3_update']['xgboost_weight'];previous=rating_checkpoint('task3_xgboost_blend',weight*clip_rating(model.predict(FX))+(1-weight)*previous)
    scaled=mod('task3_scaled_bias');scaled_dir=folder(WORK/'task3_scaled_temporal')
    def scaling(fullflag):
        cache=scaled_dir/('test_features.npz' if fullflag else 'validation_features.npz')
        if cache.exists():return np.load(cache)['features']
        _,_,qq,u,i,t,yy,qu,qi,*_=r3.dataset(DATA,fullflag)
        with open(prior/('full' if fullflag else 'validation')/'temporal_bias.pkl','rb') as f:base=pickle.load(f)
        model=scaled.ScaledTemporalBias(base,20)
        params=scaled_dir/'production_scaling.npz'
        if fullflag and params.exists():model.restore(params)
        else:
            model.fit(u,i,t,yy)
            if fullflag:model.save(params)
        p,s,b=model.parts(qu,qi,qq.timestamp.to_numpy(float));features=np.column_stack([clip_rating(p),s,b]).astype('f')
        np.savez_compressed(cache,features=features);return features
    X=np.c_[X,scaling(False)];FX=np.c_[FX,scaling(True)]
    names+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect']
    model=fixed_xgb(scaled_dir/'production_xgboost.ubj.gz',lambda:rating_xgb(499),X,y)
    weight=SETTINGS['task3_update']['scaling_weight'];previous=rating_checkpoint('task3_scaled_temporal',weight*clip_rating(model.predict(FX))+(1-weight)*previous)
    capacity=mod('task3_capacity_final');capacity.compile_kernel()
    capacity.run_worker(128,'validation',200);capacity.run_worker(128,'full',200)
    def append_capacity(A,p):
        old=A[:,names.index('joint48strong_e50')]
        neighbor=.5*(A[:,names.index('learned_rating25_365')]+A[:,names.index('wide_learned_rating25_365')])
        return np.c_[A,p,p-old,p-neighbor].astype('f')
    cap_dir=WORK/'task3_capacity_extended/128'
    X=append_capacity(X,np.load(cap_dir/'validation_scores.npz')['prediction'])
    FX=append_capacity(FX,np.load(cap_dir/'full_scores.npz')['prediction'])
    model=fixed_xgb(cap_dir/'production_combiner.ubj.gz',lambda:rating_xgb(574),X,y)
    weight=SETTINGS['task3_update']['capacity_weight'];previous=rating_checkpoint('task3_timesvdpp128',weight*clip_rating(model.predict(FX))+(1-weight)*previous)
    local=mod('task3_local_linear_features');local_dir=folder(WORK/'task3_local_linear')
    L,ln,_=local.generate(local_dir,False);F,fn,_=local.generate(local_dir,True);assert ln==fn
    X=np.c_[X,L];FX=np.c_[FX,F]
    model=fixed_xgb(local_dir/'production_xgboost.ubj.gz',lambda:rating_xgb(642),X,y)
    weight=SETTINGS['task3_update']['local_linear_weight'];prediction=clip_rating(weight*clip_rating(model.predict(FX))+(1-weight)*previous)
    return write_prediction(3,prediction,OUTPUT/'best/task_3_predictions.csv')
def clip_rating(p):return np.clip(p,1,5)

# TASK 6 UPDATE: median experts with separately fitted quantile features.
def median_estimator(name,count):
    cfg={'raw':(31,200,50.,.035),'correction15':(15,150,30.,.025),'correction31':(31,200,50.,.025)}[name]
    return lgb.LGBMRegressor(objective='quantile',alpha=.5,n_estimators=count,num_leaves=cfg[0],min_child_samples=cfg[1],reg_lambda=cfg[2],learning_rate=cfg[3],max_bin=127,colsample_bytree=.9,subsample=.9,subsample_freq=1,reg_alpha=.1,path_smooth=10,random_state=3139,n_jobs=2,verbosity=-1)
def quantile_features(a,b,c,anchor):
    lo=np.minimum(np.minimum(a,b),c);hi=np.maximum(np.maximum(a,b),c);width=hi-lo
    return np.column_stack([np.log1p(a),np.log1p(b),np.log1p(c),np.log1p(width),np.log1p(b-lo),np.log1p(hi-b),np.log1p(width/(b+1)),(hi+lo-2*b)/(width+1),np.sign(anchor-b)*np.log1p(abs(anchor-b)),((a>b)|(b>c)).astype(float)]).astype('f')
def task6_quantile_finish(val,complete,peer,full_peer,y,ref,anchor,n):
    root=folder(WORK/'task6_quantile');interval=folder(WORK/'task6_interval_features')
    X=np.c_[val['X'],peer].astype('f');FX=np.c_[complete['X'],full_peer['X']].astype('f');Q=np.c_[complete['Q'],full_peer['Q']].astype('f')
    M=np.c_[X[n:],np.log1p(ref)].astype('f');MQ=np.c_[Q,np.log1p(anchor)].astype('f')
    raw=fixed_lgb(root/'median_raw_production.txt',lambda:median_estimator('raw',1171),FX,complete['y'])
    raw_test=np.maximum(0,raw.predict(Q));preds={'median_raw':raw_test}
    for key,count in [('correction15',1525),('correction31',1263)]:
        model=fixed_lgb(root/('median_'+key+'_production.txt'),lambda key=key,count=count:median_estimator(key,count),M,y[n:]-ref)
        preds['median_'+key]=np.maximum(0,anchor+model.predict(MQ))
    weights=SETTINGS['task6_update']['median_weights']
    old=sum(weights[k]*v for k,v in preds.items())
    median_csv=write_prediction(6,old*86400,OUTPUT/'task6_quantile_ensemble/task_6_predictions.csv')
    old=pd.read_csv(median_csv).delta.to_numpy()/86400
    earlier_median=fixed_lgb(root/'median_raw_selection.txt',lambda:median_estimator('raw',1171),X[:n],y[:n])
    earlier={50:np.maximum(0,earlier_median.predict(X[n:]))};test={50:raw_test}
    for q in [25,75]:
        def factory(q=q):
            model=median_estimator('raw',1171);model.set_params(alpha=q/100);return model
        model=fixed_lgb(interval/f'q{q}_earlier.txt',factory,X[:n],y[:n]);earlier[q]=np.maximum(0,model.predict(X[n:]))
        model=fixed_lgb(interval/f'q{q}_production.txt',factory,FX,complete['y']);test[q]=np.maximum(0,model.predict(Q))
    A=np.c_[M,quantile_features(earlier[25],earlier[50],earlier[75],ref)].astype('f')
    AQ=np.c_[MQ,quantile_features(test[25],test[50],test[75],anchor)].astype('f')
    predictions={'accepted':old}
    for key,count in [('correction15',1525),('correction31',1263)]:
        model=fixed_lgb(interval/('median_'+key+'_production.txt'),lambda key=key,count=count:median_estimator(key,count),A,y[n:]-ref)
        predictions['median_'+key]=np.maximum(0,anchor+model.predict(AQ))
    weights=SETTINGS['task6_update']['interval_weights'];value=sum(weights[k]*v for k,v in predictions.items())*86400
    return write_prediction(6,value,OUTPUT/'best/task_6_predictions.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=range(1, 7), required=True)
    arguments = parser.parse_args()
    result = globals()['task' + str(arguments.task)]()
    print('COMPLETED:', result, flush=True)
''',
    'selected_settings.json': r'''{"task1":{"improved":{"runs":{"depth4":{"depth":4,"iterations":1791},"depth6":{"depth":6,"iterations":1469}},"weights":{"depth4":0.5,"depth6":0.5}},"next":{"runs":{"depth4":{"depth":4,"iterations":1983},"depth6":{"depth":6,"iterations":1023}},"depth_weights":{"depth4":0.5,"depth6":0.5},"new_weight":0.75},"neural":{"runs":{"ce64":{"epochs":19,"hidden":[64,32],"lam":0.0,"seed":42},"joint64":{"epochs":18,"hidden":[64,32],"lam":2.0,"seed":73},"joint128":{"epochs":10,"hidden":[128,64],"lam":2.0,"seed":117}}},"neural_weights":{"accepted":0.39062690771432695,"ce64":0.1642789009208677,"joint64":0.353405287415689,"joint128":0.09168890394911629}},"task2":{"depth":7,"iterations":275},"task3":{"base_columns":["bias_static","bias_drift_item","bias_temporal180","bias_temporal90","bias_temporal90_strong","bias_svd16","bias_svd_global_month","bias_svd32","grid_drift_u5_d2","grid_drift_u5_d10","grid_drift_u5_d40","grid_drift_u15_d2","grid_drift_u15_d10","grid_drift_u15_d40","grid_asymmetric_2","grid_asymmetric_10","grid_asymmetric_40","grid_period_reg10","grid_period_reg50","grid_period_reg100","neighbor_static_20_1_0","neighbor_static_20_2_0","neighbor_static_20_1_365","neighbor_static_20_2_365","neighbor_static_20_1_90","neighbor_static_20_2_90","neighbor_temporal_20_1_0","neighbor_temporal_20_2_0","neighbor_temporal_20_1_365","neighbor_temporal_20_2_365","neighbor_temporal_20_1_90","neighbor_temporal_20_2_90","kernel_static_7_1","kernel_static_7_5","kernel_static_30_1","kernel_static_30_5","kernel_static_90_1","kernel_static_90_5","kernel_static_365_1","kernel_static_365_5","kernel_temporal_7_1","kernel_temporal_7_5","kernel_temporal_30_1","kernel_temporal_30_5","kernel_temporal_90_1","kernel_temporal_90_5","kernel_temporal_365_1","kernel_temporal_365_5"],"models":[{"depth":6,"iterations":1550,"weight":0.5,"filename":"cat_6_res0_full.cbm"},{"depth":8,"iterations":908,"weight":0.5,"filename":"cat_8_res0_full.cbm"}]},"task4":{"params":{"num_leaves":15,"min_child_samples":100},"iterations":1100},"task6":{"base":{"models":{"book_mae":{"mode":"book","iterations":1196,"loss":"MAE","log":false,"depth":6,"seed":42,"categories":true},"book_log":{"mode":"book","iterations":1171,"loss":"MAE","log":true,"depth":6,"seed":42,"categories":true}},"weights":{"book_mae":0.8854989833070736,"book_log":0.0958632591333806,"ridge":0.018637757559545862}},"improved":{"runs":{"balanced":{"iterations":561},"cat":{"iterations":1800}},"blend_weights":{"accepted":0.13833201728720498,"lgb":0.0,"recent":0.0,"balanced":0.44086204832498915,"cat":0.4208059343878058}}},"task2_update":{"weight":0.6000000000000001},"task3_update":{"lightgbm_weight":0.5009940772982997,"xgboost_weight":0.4215098328071107,"scaling_weight":0.3259124287933958,"capacity_weight":0.43438806455897916,"local_linear_weight":0.42738492022261776},"task6_update":{"median_weights":{"median_raw":0.008621097809580928,"median_correction15":0.44638134950997876,"median_correction31":0.5449975526804404},"interval_weights":{"accepted":0.2041991638970812,"median_correction15":0.3028727728425532,"median_correction31":0.4929280632603656}}}''',
    # TASK 1 MODEL SOURCE
    'task1_improve.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
WEIGHTS=np.array([.1541547817989482,.16261770829034805,.1196487441069676,.1062846995076872,.31124904671666187,.1460450195793872])
def clip(p):return np.clip(p,1,5)
def dataset(data,full):
    p=pd.read_csv(data/'tasks_1_2_train.csv');p=p[p.rating>0].reset_index(drop=True);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);order=np.random.RandomState(42).permutation(len(p));a=int(.9*len(p));b=int(.95*len(p));ix=np.arange(len(p)) if full else order[:a];q=pd.read_csv(data/'task_1_test.csv') if full else p.iloc[order[a:]].reset_index(drop=True);qu=pd.Index(users).get_indexer(q.user);qi=pd.Index(items).get_indexer(q.item)
    return p,p.iloc[ix],q,u[ix],i[ix],qu,qi,users,items,b-a
def guard(a):
    part=a.out/('full' if a.full else 'validation');part.mkdir(parents=True,exist_ok=True);m={'full':a.full,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_1_2_train.csv','task_1_test.csv']}};path=part/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==m
    temp=part/f'manifest_{os.getpid()}.json';temp.write_text(json.dumps(m,indent=2));temp.replace(path);return part
def library(a):
    out=guard(a);os.environ.setdefault('SURPRISE_DATA_FOLDER',str(a.out/'surprise'))
    if (out/'library.npz').exists():return
    p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);U,I=len(users),len(items);order=np.argsort(u,kind='stable');uu=u[order];ii=np.ascontiguousarray(i[order],np.int32);y=np.ascontiguousarray(tr.rating.to_numpy(float)[order],np.float32);count=np.bincount(uu,minlength=U);ic=np.ascontiguousarray(np.bincount(ii,minlength=I),np.int32);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(count)],np.int32);H=coo_matrix((np.ones(len(y)),(uu,ii)),shape=(U,I)).tocsr();assert np.max(H.data)==1;mu=float(y.mean());start=time.time();result={};so=out/'svdpp.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_svdpp_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.svdpp_train.argtypes=[ctypes.c_int]*2+[ip,ip,fp,ip,ctypes.c_float]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*2+[ctypes.c_int]
    for k,reg,seed in [(24,.1,42),(48,.15,17)]:
        rng=np.random.RandomState(seed);P=rng.normal(0,.05,(U,k)).astype(np.float32);Q=rng.normal(0,.05,(I,k)).astype(np.float32);Y=np.zeros_like(Q);bu=np.zeros(U,np.float32);bi=np.zeros(I,np.float32);last=0
        for epoch in [20,50,100]:
            lib.svdpp_train(U,k,ptr,ii,y,ic,mu,P,Q,Y,bu,bi,last,epoch-last,seed,.007,reg,1);last=epoch;effective=P+(H@Y)/np.sqrt(np.maximum(count,1))[:,None];effective[count==0]=0;Q[ic==0]=0;pred=np.full(len(q),mu);knownu=qu>=0;knowni=qi>=0;both=knownu&knowni;pred[knownu]+=bu[qu[knownu]];pred[knowni]+=bi[qi[knowni]];pred[both]+=np.einsum('ij,ij->i',effective[qu[both]],Q[qi[both]]);name=f'svdpp{k}_e{epoch}';result[name]=clip(pred);assert np.isfinite(pred).all();np.savez_compressed(out/(name+'_model.npz'),effective=effective,Q=Q,bu=bu,bi=bi,mu=mu,cu=count,ci=ic);print('Fast SVD++',k,epoch,'seconds',round(time.time()-start),flush=True)
    if not a.full:
        z=np.load(a.legacy/'task1_round1_predictions.npz')
        for k in [16,32,64]:result[f'svd{k}']=np.r_[z[f'v_svd{k}_reg.1'],z[f't_svd{k}_reg.1']]
        split=np.load(a.legacy/'task1_split.npz');assert np.array_equal(split['order'],np.random.RandomState(42).permutation(len(p)));assert np.array_equal(q.rating.to_numpy(),np.r_[split['validation_y'],split['evaluation_y']])
    np.savez_compressed(out/'library.npz',**result)
def neighbors(a):
    out=guard(a)
    if (out/'neighbors.npz').exists():return
    p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);U,I=len(users),len(items);y=tr.rating.to_numpy(float);mu=y.mean();cu=np.bincount(u,minlength=U);ci=np.bincount(i,minlength=I);bu=np.zeros(U);bi=np.zeros(I)
    for _ in range(20):bu=np.bincount(u,weights=y-mu-bi[i],minlength=U)/(cu+10);bi=np.bincount(i,weights=y-mu-bu[u],minlength=I)/(ci+10)
    residual=y-mu-bu[u]-bi[i];base=np.full(len(q),mu);knownu=qu>=0;knowni=qi>=0;base[knownu]+=bu[qu[knownu]];base[knowni]+=bi[qi[knowni]];X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix((residual,(u,i)),shape=(U,I)).tocsr();G=X.T.tocsr();RT=R.T.tocsr();RC=R.tocsc();norm=np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel());settings=[(ratio,reg) for ratio in [.25,1,4] for reg in [10,50]];result={f'joint_r{r}_p{pen}':base.copy() for r,pen in settings};result.update({f'cos{k}_{power}':base.copy() for k,power in [(10,1),(20,2),(10,2)]});groups={}
    for z,j in enumerate(qi):
        if j>=0 and qu[z]>=0:groups.setdefault(j,[]).append(z)
    rows=[[] for _ in settings];cols=[[] for _ in settings];values=[[] for _ in settings];start=time.time();keys=np.array(list(groups))
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).toarray();dot=(RT[batch]@R).toarray()
        for z,j in enumerate(batch):
            sim=co[z]/np.sqrt(np.maximum(ci[j]*ci,1));sim[j]=0;corr=np.abs(dot[z])/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+5);corr[j]=0;nb1=np.argpartition(sim,-32)[-32:];nb2=np.argpartition(corr,-32)[-32:];nb=np.unique(np.r_[nb1[sim[nb1]>0],nb2[corr[nb2]>0]]);nb=nb[nb!=j];queries=np.array(groups[j]);lo,hi=RC.indptr[j:j+2];us=RC.indices[lo:hi];yy=RC.data[lo:hi]
            for row in queries:
                lo,hi=R.indptr[qu[row]:qu[row]+2];w=sim[R.indices[lo:hi]];order=np.argsort(w)[::-1][:20];w=w[order];err=R.data[lo:hi][order]
                for k,power in [(10,1),(20,2),(10,2)]:ww=w[:k]**power;result[f'cos{k}_{power}'][row]+=(ww@err[:k])/(ww.sum()+1e-12)
            if not len(nb):continue
            A=np.c_[X[us][:,nb].toarray(),R[us][:,nb].toarray()];B=np.c_[X[qu[queries]][:,nb].toarray(),R[qu[queries]][:,nb].toarray()]
            for ratio in [.25,1,4]:
                scale=np.r_[np.ones(len(nb)),np.full(len(nb),1/np.sqrt(ratio))];scaled=A*scale;dual=len(us)<scaled.shape[1];eig,V=np.linalg.eigh(scaled@scaled.T if dual else scaled.T@scaled);left=scaled.T@V if dual else V;rhs=V.T@(yy if dual else scaled.T@yy)
                for index,(r,pen) in enumerate(settings):
                    if r!=ratio:continue
                    w=((left@(rhs/(np.maximum(eig,0)+pen)))*scale).astype(np.float32);result[f'joint_r{r}_p{pen}'][queries]+=B@w
                    if a.full:rows[index].extend(np.r_[nb,nb+I]);cols[index].extend([j]*len(w));values[index].extend(w)
        if offset%5000<64:print('Task1 neighborhoods',offset,len(keys),'seconds',round(time.time()-start),flush=True)
    for j,(r,pen) in enumerate(settings):
        if a.full:save_npz(out/f'joint_r{r}_p{pen}.npz',coo_matrix((values[j],(rows[j],cols[j])),shape=(2*I,I)).tocsr())
    np.savez_compressed(out/'neighbors.npz',**{k:clip(v) for k,v in result.items()});np.savez_compressed(out/'bias.npz',mu=mu,bu=bu,bi=bi,cu=cu,ci=ci)
def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--full',action='store_true');ap.add_argument('--worker',choices=['library','neighbors'],required=True);a=ap.parse_args();part=a.out/('full' if a.full else 'validation');part.mkdir(parents=True,exist_ok=True)
    with open(part/(a.worker+'.lock'),'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);(library if a.worker=='library' else neighbors)(a)
if __name__=='__main__':main()
''',
    'task1_improve_stack.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from task1_improve import dataset,WEIGHTS,clip
def features(a,full=False):
    part=a.out/('full' if full else 'validation');lib=dict(np.load(part/'library.npz'));nb=dict(np.load(part/'neighbors.npz'));p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,full);cos=np.column_stack([nb[f'cos{k}_{power}'] for k,power in [(10,1),(20,2),(10,2)]]);nw=WEIGHTS[3:]
    if full:
        ref=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(ref.drop(columns='rating'),q.drop(columns='rating'));baseline=ref.rating.to_numpy();svd=clip((baseline-cos@nw)/WEIGHTS[:3].sum())
    else:
        svds=np.column_stack([lib[f'svd{k}'] for k in [16,32,64]]);svd=(svds@WEIGHTS[:3])/WEIGHTS[:3].sum();baseline=svds@WEIGHTS[:3]+cos@nw
    columns={'accepted':baseline,'svd_aggregate':svd,**{f'cos{k}_{power}':cos[:,j] for j,(k,power) in enumerate([(10,1),(20,2),(10,2)])},**{k:lib[k] for k in lib if k.startswith('svdpp')},**{k:nb[k] for k in nb if k.startswith('joint')}}
    for key in ['user','item']:
        stats=tr.groupby(key).rating.agg(['count','mean','std'])
        for stat in ['count','mean','std']:columns[key+'_'+stat]=q[key].map(stats[stat]).fillna(float(tr.rating.mean()) if stat=='mean' else 0).to_numpy()
    return np.column_stack(list(columns.values())).astype(np.float32),list(columns),baseline,q,n
def cfg(depth,iters=1800):return dict(iterations=iters,depth=depth,learning_rate=.03,l2_leaf_reg=15,random_seed=42,thread_count=1,loss_function='RMSE',verbose=300,allow_writing_files=False)
''',
    'task1_neural.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task1_next_stack import matrices
from task1_improve import dataset
RATINGS=np.arange(1,6,dtype=np.float32)
def softmax(z):
 z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def head_gradient(p,y,lam):
 mean=p@RATINGS;g=p.copy();g[np.arange(len(y)),y.astype(int)-1]-=1;g+=2*lam*(mean-y)[:,None]*p*(RATINGS[None,:]-mean[:,None]);return g/len(y)
def inputs(a,full=False):
 arg=SimpleNamespace(data_dir=a.data_dir,legacy=a.legacy,previous=a.previous,out=a.reference,original=a.original);X,names,q,n=matrices(arg,full);p,tr,_,u,i,qu,qi,users,items,_=dataset(a.data_dir,full);prior=np.bincount(tr.rating.astype(int)-1,minlength=5)/len(tr)
 for key,ids in [('user',q.user),('item',q.item)]:
  table=pd.crosstab(tr[key],tr.rating).reindex(columns=[1,2,3,4,5],fill_value=0);counts=table.sum(1).to_numpy();freq=(table.to_numpy()+20*prior)/(counts[:,None]+20);mapped=pd.Index(table.index).get_indexer(ids);v=np.tile(prior,(len(q),1));ok=mapped>=0;v[ok]=freq[mapped[ok]];X=np.c_[X,v];names += [f'{key}_freq{r}' for r in range(1,6)]
 z=np.load(a.out/('full' if full else 'validation')/'cf_probabilities.npz')
 for key in z.files:X=np.c_[X,z[key],z[key]@RATINGS];names += [key+'_p'+str(r) for r in range(1,6)]+[key+'_mean']
 assert np.isfinite(X).all();return X.astype('f'),names,q,n
def transform(X,names):
 X=X.copy()
 for j,name in enumerate(names):
  if name.endswith('_count'):X[:,j]=np.log1p(X[:,j])
 return X
class Neural:
 def __init__(self,dims,seed,lam,drop=.1):
  self.seed=seed;self.lam=lam;self.drop=drop;rng=np.random.RandomState(seed);self.W=[(rng.normal(size=(l,r))*np.sqrt(2/l)).astype('f') for l,r in zip(dims[:-1],dims[1:])];self.W[-1]*=.05;self.b=[np.zeros(r,'f') for r in dims[1:]];self.mean=None;self.std=None
 def probabilities(self,X):
  out=[]
  for rows in np.array_split(np.arange(len(X)),max(1,(len(X)+4095)//4096)):
   a=np.clip((X[rows]-self.mean)/self.std,-10,10)
   for W,b in zip(self.W[:-1],self.b[:-1]):a=np.maximum(a@W+b,0)
   logits=a@self.W[-1]+self.b[-1]-(RATINGS[None,:]-X[rows,0,None])**2/(2*.7);out.append(softmax(logits))
  return np.concatenate(out)
 def save(self,path):
  z={**{f'W{j}':w for j,w in enumerate(self.W)},**{f'b{j}':b for j,b in enumerate(self.b)},'mean':self.mean,'std':self.std,'seed':self.seed,'lam':self.lam,'drop':self.drop};temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**z);temp.replace(path)
 @classmethod
 def load(cls,path):
  z=dict(np.load(path));W=[z[k] for k in z if k.startswith('W')];m=cls([W[0].shape[0]]+[w.shape[1] for w in W],int(z['seed']),float(z['lam']),float(z['drop']));m.W=W;m.b=[z[f'b{j}'] for j in range(len(W))];m.mean=z['mean'];m.std=z['std'];return m
 def fit(self,X,y,epochs,validation=None,path=None):
  self.mean=X.mean(0);self.std=np.maximum(X.std(0),.05);Z=np.clip((X-self.mean)/self.std,-10,10);anchor=-(RATINGS[None,:]-X[:,0,None])**2/(2*.7);rng=np.random.RandomState(self.seed);params=self.W+self.b;moment=[np.zeros_like(p) for p in params];variance=[np.zeros_like(p) for p in params];step=0
  for epoch in range(1,epochs+1):
   order=rng.permutation(len(X));lr=.001*(.985**(epoch-1))
   for ids in np.array_split(order,max(1,(len(order)+511)//512)):
    acts=[Z[ids]];masks=[];pre=[]
    for W,b in zip(self.W[:-1],self.b[:-1]):
     v=acts[-1]@W+b;pre.append(v);mask=(rng.random_sample(v.shape)>=self.drop).astype('f')/(1-self.drop);masks.append(mask);acts.append(np.maximum(v,0)*mask)
    prob=softmax(acts[-1]@self.W[-1]+self.b[-1]+anchor[ids]);g=head_gradient(prob,y[ids],self.lam);gw=[None]*len(self.W);gb=[None]*len(self.b)
    for layer in range(len(self.W)-1,-1,-1):
     gw[layer]=acts[layer].T@g+.0001*self.W[layer];gb[layer]=g.sum(0)
     if layer:g=(g@self.W[layer].T)*(pre[layer-1]>0)*masks[layer-1]
    step+=1
    for j,grad in enumerate(gw+gb):
     moment[j]*=.9;moment[j]+=.1*grad;variance[j]*=.999;variance[j]+=.001*grad*grad;params[j]-=lr*(moment[j]/(1-.9**step))/(np.sqrt(variance[j]/(1-.999**step))+1e-8)
  return None
''',
    'task1_neural_cf.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task1_improve import dataset
def probabilities(model,qu,qi):
 P,Q,bu,bi,W,b=[model[x] for x in ['P','Q','bu','bi','W','b']];out=np.empty((len(qu),5),'f')
 for ix in np.array_split(np.arange(len(qu)),max(1,(len(qu)+8191)//8192)):
  ku=qu[ix]>=0;ki=qi[ix]>=0;h=np.zeros((len(ix),P.shape[1]),'f');ok=ku&ki;h[ok]=np.tanh(P[qu[ix[ok]]]*Q[qi[ix[ok]]]);logits=h@W.T+b;logits[ku]+=bu[qu[ix[ku]]];logits[ki]+=bi[qi[ix[ki]]];logits-=logits.max(1,keepdims=True);prob=np.exp(logits);out[ix]=prob/prob.sum(1,keepdims=True)
 assert np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-6);return out
def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(exist_ok=True,parents=True);target=out/'cf_probabilities.npz'
 if target.exists():return
 p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);u=np.ascontiguousarray(u,np.int32);i=np.ascontiguousarray(i,np.int32);y=np.ascontiguousarray(tr.rating.to_numpy()-1,np.int32);cu=np.ascontiguousarray(np.bincount(u,minlength=len(users)),np.int32);ci=np.ascontiguousarray(np.bincount(i,minlength=len(items)),np.int32);warm=dict(np.load(a.previous/('full' if a.full else 'validation')/'svdpp24_e50_model.npz'));k=24;globalp=np.bincount(y,minlength=5)/len(y);userfreq=np.zeros((len(users),5));itemfreq=np.zeros((len(items),5));np.add.at(userfreq,(u,y),1);np.add.at(itemfreq,(i,y),1)
 so=out/'neural.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_neural_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.neural_train.argtypes=[ctypes.c_int]*2+[ip]*5+[fp]*12+[ctypes.c_int]*3+[ctypes.c_float];result={};start=time.time()
 for lam in [0.,2.]:
  P=np.array(warm['effective'],dtype='f',order='C',copy=True);Q=np.array(warm['Q'],dtype='f',order='C',copy=True);bu=(.5*np.log(((userfreq+20*globalp)/(cu[:,None]+20))/globalp)).astype('f');bi=(.5*np.log(((itemfreq+20*globalp)/(ci[:,None]+20))/globalp)).astype('f');W=np.tile((np.arange(5)-2)[:,None]*.3,(1,k)).astype('f');b=np.log(globalp).astype('f');parameters=[P,Q,bu,bi,W,b];history=[np.zeros_like(z) for z in parameters];last=0;name='ce' if lam==0 else 'ce_mse'
  for epoch in [3,8,15,25]:
   lib.neural_train(len(u),k,u,i,y,cu,ci,*parameters,*history,last,epoch-last,133,lam);last=epoch;assert all(np.isfinite(z).all() for z in parameters);result[name+'_e'+str(epoch)]=probabilities(dict(zip(['P','Q','bu','bi','W','b'],parameters)),qu,qi);print('NEURAL_CF',a.full,name,epoch,'seconds',round(time.time()-start),flush=True)
  np.savez_compressed(out/(name+'_model.npz'),**dict(zip(['P','Q','bu','bi','W','b'],parameters)))
 temp=out/'cf_probabilities.tmp.npz';np.savez_compressed(temp,**result);temp.replace(target);(out/'manifest.json').write_text(json.dumps({'full':a.full,'positive_train_rows':len(u),'selection_rows':n,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_1_2_train.csv','task_1_test.csv']}},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','previous','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task1_neural_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
static void step(int u,int i,int y,int k,const int* cu,const int* ci,float* P,float* Q,float* bu,float* bi,float* W,float* b,float* HP,float* HQ,float* Hbu,float* Hbi,float* HW,float* Hb,float lambda){
 std::vector<float> h(k),dh(k);float p[5],d[5],mx=-1e30f,total=0,mean=0;
 for(int f=0;f<k;f++)h[f]=std::tanh(P[u*k+f]*Q[i*k+f]);
 for(int c=0;c<5;c++){float v=b[c]+bu[u*5+c]+bi[i*5+c];for(int f=0;f<k;f++)v+=W[c*k+f]*h[f];p[c]=v;mx=std::max(mx,v);}
 for(int c=0;c<5;c++){p[c]=std::exp(p[c]-mx);total+=p[c];}for(int c=0;c<5;c++){p[c]/=total;mean+=p[c]*(c+1);}
 for(int c=0;c<5;c++)d[c]=p[c]-(y==c?1.f:0.f)+2*lambda*(mean-(y+1))*p[c]*((c+1)-mean);
 for(int f=0;f<k;f++){float g=0;for(int c=0;c<5;c++)g+=d[c]*W[c*k+f];dh[f]=g*(1-h[f]*h[f]);}
 auto update=[](float& value,float& hist,float grad){hist+=grad*grad;value-=.05f*grad/std::sqrt(1.f+hist);};
 for(int f=0;f<k;f++){int a=u*k+f,z=i*k+f;float pu=P[a],qi=Q[z];update(P[a],HP[a],dh[f]*qi+.01f*pu);update(Q[z],HQ[z],dh[f]*pu+.01f*qi);}
 for(int c=0;c<5;c++){int a=u*5+c,z=i*5+c;update(bu[a],Hbu[a],d[c]+5.f/std::max(cu[u],1)*bu[a]);update(bi[z],Hbi[z],d[c]+5.f/std::max(ci[i],1)*bi[z]);update(b[c],Hb[c],d[c]);for(int f=0;f<k;f++){int q=c*k+f;update(W[q],HW[q],d[c]*h[f]+.00001f*W[q]);}}
}
extern "C" void neural_train(int n,int k,const int* u,const int* i,const int* y,const int* cu,const int* ci,float* P,float* Q,float* bu,float* bi,float* W,float* b,float* HP,float* HQ,float* Hbu,float* Hbi,float* HW,float* Hb,int start,int epochs,int seed,float lambda){
 std::vector<int> order(n);for(int epoch=start;epoch<start+epochs;epoch++){std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+epoch);std::shuffle(order.begin(),order.end(),rng);for(int row:order)step(u[row],i[row],y[row],k,cu,ci,P,Q,bu,bi,W,b,HP,HQ,Hbu,Hbi,HW,Hb,lambda);}
}
''',
    'task1_next.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task1_improve import dataset,clip
def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(parents=True,exist_ok=True)
 p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);U,I=len(users),len(items);y=tr.rating.to_numpy(float);mu=y.mean();cu=np.bincount(u,minlength=U);ci=np.bincount(i,minlength=I);bu=np.zeros(U);bi=np.zeros(I)
 for _ in range(20):bu=np.bincount(u,weights=y-mu-bi[i],minlength=U)/(cu+10);bi=np.bincount(i,weights=y-mu-bu[u],minlength=I)/(ci+10)
 X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix((y-mu-bu[u]-bi[i],(u,i)),shape=(U,I)).tocsr();base=np.full(len(q),mu);ku=qu>=0;ki=qi>=0;base[ku]+=bu[qu[ku]];base[ki]+=bi[qi[ki]]
 if a.worker in ['svdpp','svdppstrong','itemspp']:
  if a.worker=='itemspp':u,i,qu,qi,U,I,cu,ci=i,u,qi,qu,I,U,ci,cu;X=X.T.tocsr();ku=qu>=0;ki=qi>=0
  filename=a.worker+'.npz'
  if (out/filename).exists():return
  order=np.argsort(u,kind='stable');ii=np.ascontiguousarray(i[order],np.int32);yy=np.ascontiguousarray(y[order],np.float32);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(cu)],np.int32);ic=np.ascontiguousarray(ci,np.int32);so=out/'svd.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_svdpp_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.svdpp_train.argtypes=[ctypes.c_int]*2+[ip,ip,fp,ip,ctypes.c_float]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*2+[ctypes.c_int]
  k=96 if a.worker=='svdppstrong' else (32 if a.worker=='itemspp' else 64);reg=.3 if a.worker=='svdppstrong' else (.1 if a.worker=='itemspp' else .04);rng=np.random.RandomState(71);P=rng.normal(0,.05,(U,k)).astype('f');Q=rng.normal(0,.05,(I,k)).astype('f');Y=np.zeros_like(Q);b=np.zeros(U,'f');c=np.zeros(I,'f');last=0;result={};start=time.time()
  for epoch in ([30,70,140] if a.worker=='svdppstrong' else ([20,50,100] if a.worker=='itemspp' else [50,100,180])):
   lib.svdpp_train(U,k,ptr,ii,yy,ic,mu,P,Q,Y,b,c,last,epoch-last,71,.007,reg,1);last=epoch;effective=P+(X@Y)/np.sqrt(np.maximum(cu,1))[:,None];effective[cu==0]=0;Q[ci==0]=0;pred=np.full(len(q),mu);pred[ku]+=b[qu[ku]];pred[ki]+=c[qi[ki]];ok=ku&ki;pred[ok]+=np.einsum('ij,ij->i',effective[qu[ok]],Q[qi[ok]]);result[('svdppstrong96_e' if a.worker=='svdppstrong' else ('itemspp32_e' if a.worker=='itemspp' else 'svdpp64low_e'))+str(epoch)]=clip(pred);print('svdpp',epoch,round(time.time()-start),flush=True)
  np.savez_compressed(out/filename,**result);np.savez_compressed(out/(a.worker+'_final.npz'),effective=effective,Q=Q,bu=b,bi=c,mu=mu);return
 if (out/'neighbors.npz').exists():return
 result={};start=time.time()
 for axis in ['item','user']:
  A=X if axis=='item' else X.T.tocsr();E=R if axis=='item' else R.T.tocsr();target=qi if axis=='item' else qu;context=qu if axis=='item' else qi;G=A.T.tocsr();ET=E.T.tocsr();EC=E.tocsc();count=np.asarray(A.sum(0)).ravel();norm=np.sqrt(np.asarray(E.multiply(E).sum(0)).ravel());groups={}
  for z,j in enumerate(target):
   if j>=0 and context[z]>=0:groups.setdefault(j,[]).append(z)
  keys=np.array(list(groups));settings=[(r,pen) for r in [1,4] for pen in [20,100]]
  for r,pen in settings:result[f'{axis}_ridge_r{r}_p{pen}']=base.copy()
  for power in [1,2]:result[f'{axis}_pearson{power}']=base.copy()
  for off in range(0,len(keys),32):
   batch=keys[off:off+32];co=(G[batch]@A).toarray();dot=(ET[batch]@E).toarray()
   for z,j in enumerate(batch):
    sim=co[z]/np.sqrt(np.maximum(count[j]*count,1));sim[j]=0;corr=dot[z]/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+10);corr[j]=0;top=lambda arr:np.argpartition(arr,-min(64,len(arr)))[-min(64,len(arr)):];nb=np.unique(np.r_[top(sim),top(abs(corr))]);nb=nb[(nb!=j)&(sim[nb]>0)];queries=np.array(groups[j]);lo,hi=EC.indptr[j:j+2];trainrows=EC.indices[lo:hi];yy=EC.data[lo:hi]
    for row in queries:
     lo,hi=E.indptr[context[row]:context[row]+2];ids=E.indices[lo:hi];w=corr[ids];order=np.argsort(abs(w))[::-1][:40];w=w[order];err=E.data[lo:hi][order]
     for power in [1,2]:ww=np.sign(w)*abs(w)**power;result[f'{axis}_pearson{power}'][row]+=(ww@err)/(abs(ww).sum()+1e-12)
    if not len(nb):continue
    T=np.c_[A[trainrows][:,nb].toarray(),E[trainrows][:,nb].toarray()];B=np.c_[A[context[queries]][:,nb].toarray(),E[context[queries]][:,nb].toarray()]
    for ratio in [1,4]:
     scale=np.r_[np.ones(len(nb)),np.full(len(nb),1/np.sqrt(ratio))];S=T*scale;dual=len(trainrows)<S.shape[1];eig,V=np.linalg.eigh(S@S.T if dual else S.T@S);left=S.T@V if dual else V;rhs=V.T@(yy if dual else S.T@yy)
     for pen in [20,100]:w=(left@(rhs/(np.maximum(eig,0)+pen)))*scale;result[f'{axis}_ridge_r{ratio}_p{pen}'][queries]+=B@w
   if off%5000<32:print(axis,off,len(keys),'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(out/'neighbors.npz',**{k:clip(v) for k,v in result.items()})
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--worker',choices=['svdpp','svdppstrong','itemspp','neighbors'],required=True);ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task1_next_stack.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task1_improve_stack import features
def matrices(a,full=False):
 args=SimpleNamespace(data_dir=a.data_dir,legacy=a.legacy,out=a.previous,accepted=a.original);X,names,old,q,n=features(args,full);part=a.out/('full' if full else 'validation')
 for kind in ['svdpp','svdppstrong','itemspp','neighbors']:
  z=np.load(part/(kind+'.npz'));X=np.c_[X,np.column_stack([z[k] for k in z.files])];names+=z.files
 return X,names,q,n
''',
    'task1_round1.py': r'''from final_imports import *
p=pd.read_csv(Path(os.environ['ASSIGNMENT_DATA_DIR'])/'tasks_1_2_train.csv',dtype={'user':str,'item':str});p=p[p.rating>0].reset_index(drop=True)
order=np.random.RandomState(42).permutation(len(p));a,b=int(.9*len(p)),int(.95*len(p));tr=p.iloc[order[:a]];q=p.iloc[order[a:]];root=Path(os.environ['ASSIGNMENT_WORK_DIR']);np.savez_compressed(root/'task1_split.npz',order=order,validation_y=q.rating.to_numpy()[:b-a],evaluation_y=q.rating.to_numpy()[b-a:])
training=Dataset.load_from_df(tr[['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset();result={}
for k in [16,32,64]:
 model=SVD(n_factors=k,n_epochs=50,reg_all=.1,random_state=42);model.fit(training);v=np.array([model.predict(u,i).est for u,i in q[['user','item']].itertuples(index=False,name=None)]);result[f'v_svd{k}_reg.1']=v[:b-a];result[f't_svd{k}_reg.1']=v[b-a:]
np.savez_compressed(root/'task1_round1_predictions.npz',**result)
''',
    'task1_submission.py': r'''from final_imports import *
from task1_improve import WEIGHTS
def train_predict(data_dir,output,neighbor_cache):
 p=pd.read_csv(Path(data_dir)/'tasks_1_2_train.csv');p=p[p.rating>0].reset_index(drop=True);q=pd.read_csv(Path(data_dir)/'task_1_test.csv');training=Dataset.load_from_df(p[['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset();prediction=np.zeros(len(q))
 for k,w in zip([16,32,64],WEIGHTS[:3]):
  model=SVD(n_factors=k,n_epochs=50,reg_all=.1,random_state=42);model.fit(training);prediction+=w*np.array([model.predict(u,i).est for u,i in q[['user','item']].itertuples(index=False,name=None)])
 z=np.load(neighbor_cache)
 for name,w in zip(['cos10_1','cos20_2','cos10_2'],WEIGHTS[3:]):prediction+=w*z[name]
 q['rating']=np.clip(prediction,1,5);Path(output).parent.mkdir(parents=True,exist_ok=True);q.to_csv(output,index=False)
 return prediction
''',
    'task1_svdpp_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
extern "C" void svdpp_train(int U,int k,const int* ptr,const int* item,const float* rating,
 const int* ic,float mu,float* P,float* Q,float* Y,float* bu,float* bi,
 int start,int epochs,int seed,float lr,float reg,int shuffle){
 std::vector<int> users(U),order;std::vector<float> z(k),initial(k);
 for(int epoch=start;epoch<start+epochs;epoch++){
  std::iota(users.begin(),users.end(),0);std::mt19937 rng(seed+epoch);if(shuffle)std::shuffle(users.begin(),users.end(),rng);float eta=lr*std::pow(.985f,epoch);
  for(int u:users){int n=ptr[u+1]-ptr[u];if(!n)continue;float norm=1.f/std::sqrt(float(n));std::fill(z.begin(),z.end(),0);
   for(int row=ptr[u];row<ptr[u+1];row++)for(int f=0;f<k;f++)z[f]+=Y[item[row]*k+f]*norm;initial=z;order.resize(n);std::iota(order.begin(),order.end(),ptr[u]);if(shuffle)std::shuffle(order.begin(),order.end(),rng);
   for(int row:order){int j=item[row];float pred=mu+bu[u]+bi[j];for(int f=0;f<k;f++)pred+=(P[u*k+f]+z[f])*Q[j*k+f];float e=rating[row]-pred;bu[u]+=eta*(e-10.f/n*bu[u]);bi[j]+=eta*(e-10.f/ic[j]*bi[j]);
    for(int f=0;f<k;f++){int p=u*k+f,q=j*k+f;float oldq=Q[q],effective=P[p]+z[f];P[p]+=eta*(e*oldq-reg*P[p]);Q[q]+=eta*(e*effective-reg*oldq);z[f]+=eta*(e*oldq-reg*z[f]);}
   }
   float decay=std::pow(1.f-eta*reg,n);for(int row=ptr[u];row<ptr[u+1];row++)for(int f=0;f<k;f++){int j=item[row]*k+f;Y[j]=decay*Y[j]+(z[f]-decay*initial[f])*norm;}
  }
 }
}
''',
    # TASK 2 MODEL SOURCE
    'task2_optimize.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
def log(*args): print(*args,flush=True)
def matrix(u,i,mask,shape):
    x=coo_matrix((np.ones(np.sum(mask),np.float32),(u[mask],i[mask])),shape=shape).tocsr();x.data[:]=1
    return x
def factor_scores(X,qu,qi,out):
    specs=[('bpr64',BayesianPersonalizedRanking,dict(factors=64,iterations=200,regularization=.01,learning_rate=.03,random_state=17,num_threads=1)),
           ('bpr128',BayesianPersonalizedRanking,dict(factors=128,iterations=350,regularization=.03,learning_rate=.03,random_state=42,num_threads=1)),
           ('bpr96',BayesianPersonalizedRanking,dict(factors=96,iterations=300,regularization=.003,learning_rate=.05,random_state=23,num_threads=1)),
           ('als128',AlternatingLeastSquares,dict(factors=128,iterations=30,regularization=.1,alpha=40,random_state=42,num_threads=2))]
    scores={}
    for name,cls,kw in specs:
        path=out/(name+'.npz')
        if path.exists():
            z=np.load(path);uf,iv=z['user'],z['item']
        else:
            start=time.time();model=cls(**kw);model.fit(X,show_progress=False);uf,iv=model.user_factors,model.item_factors
            np.savez_compressed(path,user=uf,item=iv);log(name,'seconds',round(time.time()-start,1))
        scores[name]=np.einsum('ij,ij->i',uf[qu],iv[qi]); valid=(np.asarray(X.sum(1)).ravel()[qu]>0)&(np.asarray(X.sum(0)).ravel()[qi]>0);scores[name][~valid]=0;del uf,iv
    return scores
def ease_scores(X,qu,qi,out):
    pop=np.asarray(X.sum(0)).ravel();top=np.argsort(pop)[-min(2000,X.shape[1]):];Z=X[:,top].astype(np.float64)
    G=(Z.T@Z).toarray();mapping=np.full(X.shape[1],-1);mapping[top]=np.arange(len(top));q=mapping[qi]
    scores={}
    for reg in [50.,300.,1000.]:
        cache=out/('ease_'+str(int(reg))+'.npy')
        if cache.exists(): B=np.load(cache)
        else:
            A=G.copy();A.flat[::len(top)+1]+=reg
            P=cho_solve(cho_factor(A),np.eye(len(top)));B=-P/np.diag(P)[None,:];np.fill_diagonal(B,0)
            np.save(cache,B.astype(np.float32));del A,P
        s=np.zeros(len(qu),np.float32)
        valid=np.flatnonzero(q>=0)
        for ix in np.array_split(valid,max(1,(len(valid)+1999)//2000)):
            if len(ix):s[ix]=np.asarray(Z[qu[ix]].multiply(B[:,q[ix]].T).sum(1)).ravel()
        scores['ease'+str(int(reg))]=s
    scores['ease_covered']=(q>=0).astype(np.float32)
    return scores
def linear_neighbors(X,qu,qi,out):
    cache=out/'linear_scores.npz'
    if cache.exists():return dict(np.load(cache))
    source=Path(__file__).with_name('task2_slim_kernel.cpp');libpath=out/'slim_kernel.so'
    if not libpath.exists():subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(libpath)],check=True)
    lib=ctypes.CDLL(str(libpath.resolve()));kernel=lib.slim_fit
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');kernel.argtypes=[ctypes.c_int,ptr,ptr,ptr];kernel.restype=ctypes.c_double
    pop=np.asarray(X.sum(0)).ravel();Xt=X.T.tocsr()
    names=['cosine_max','cosine_top10','jaccard_max','slim02','slim2']
    scores={n:np.zeros(len(qu),np.float32) for n in names}
    order=np.argsort(qi,kind='stable');items,starts=np.unique(qi[order],return_index=True);ends=np.r_[starts[1:],len(order)]
    slim_rows=[[],[]];slim_cols=[[],[]];slim_data=[[],[]]
    max_change=0.;nonconverged=0
    for offset in range(0,len(items),64):
        keys=items[offset:offset+64];co=(Xt[keys]@X).toarray()
        for k,j in enumerate(keys):
            p=offset+k;ix=order[starts[p]:ends[p]];c=co[k];c[j]=0
            cos=c/np.sqrt(np.maximum(pop[j]*pop,1));jac=c/np.maximum(pop[j]+pop-c,1)
            H=X[qu[ix]]
            for r,query in enumerate(ix):
                hist=H.indices[H.indptr[r]:H.indptr[r+1]];v=cos[hist]
                if len(v):
                    scores['cosine_max'][query]=v.max();scores['cosine_top10'][query]=np.sort(v)[-10:].sum();scores['jaccard_max'][query]=jac[hist].max()
            n=min(32,np.count_nonzero(c))
            if not n:continue
            nb=np.argpartition(cos,-n)[-n:];nb=nb[np.argsort(cos[nb])[::-1]]
            gram=np.ascontiguousarray((Xt[nb]@Xt[nb].T).toarray(),dtype=np.float64)
            rhs=np.ascontiguousarray(c[nb],dtype=np.float64);B=np.zeros((n,2))
            change=kernel(n,gram,rhs,B)
            max_change=max(max_change,change);nonconverged+=change>=1e-5
            hs=H[:,nb]
            for v,name in enumerate(['slim02','slim2']):
                scores[name][ix]=np.asarray(hs@B[:,v]).ravel();nz=B[:,v]>0
                slim_rows[v].extend(nb[nz]);slim_cols[v].extend([j]*int(nz.sum()));slim_data[v].extend(B[nz,v])
        if offset%2048==0:log('Neighborhood/SLIM',offset,'/',len(items))
    for v,name in enumerate(['slim02','slim2']):
        W=coo_matrix((slim_data[v],(slim_rows[v],slim_cols[v])),shape=(X.shape[1],X.shape[1])).tocsr()
        assert np.count_nonzero(W.diagonal())==0;save_npz(out/(name+'.npz'),W)
    log('SLIM max final coordinate change',max_change,'not converged',nonconverged)
    (out/'slim_convergence.json').write_text(json.dumps(dict(max_final_change=max_change,nonconverged=int(nonconverged),items=len(items))))
    np.savez_compressed(cache,**scores)
    return scores
def features(X,qu,qi,out):
    cache=out/'features.npz'
    if cache.exists():
        z=np.load(cache);names=list(z.files);f=np.column_stack([z[n] for n in names]);valid=(np.asarray(X.sum(1)).ravel()[qu]>0)&(np.asarray(X.sum(0)).ravel()[qi]>0)
        for k,n in enumerate(names):
            if n.startswith(('bpr','als')):f[~valid,k]=0
        if not valid.all():np.savez_compressed(cache,**{n:f[:,k] for k,n in enumerate(names)})
        return names,f
    pop=np.asarray(X.sum(0)).ravel();up=np.asarray(X.sum(1)).ravel()
    scores=dict(popularity=np.log1p(pop[qi]),already_seen=np.asarray(X[qu,qi]).ravel().astype(np.float32),user_count=np.log1p(up[qu]))
    scores.update(factor_scores(X,qu,qi,out));scores.update(ease_scores(X,qu,qi,out));scores.update(linear_neighbors(X,qu,qi,out))
    assert all(np.isfinite(x).all() for x in scores.values());np.savez_compressed(cache,**scores)
    return list(scores),np.column_stack(list(scores.values()))
def design(d,s):return np.column_stack([d,s])
def predict(m,d,s):
    return m.predict_proba(design(d,s))[:,1]-m.predict_proba(design(-d,s))[:,1]
''',
    'task2_slim_kernel.cpp': r'''#include <cmath>
#include <algorithm>
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
''',
    # TASK 3 MODEL SOURCE
    'task3_capacity_final.py': r'''from final_imports import *
from task3_round2 import time_arrays
from task3_optimize import dataset,clip,mse
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';OUT=W/'task3_capacity_extended';OUT.mkdir(exist_ok=True)
def compile_kernel():
    path = OUT / 'joint.so'
    if not path.exists():
        subprocess.run([os.environ.get('CXX','c++'), '-O3', '-shared', '-fPIC', str(Path(__file__).with_name('task3_joint_kernel.cpp')), '-o', str(path)], check=True)
    return path
def make_data(stage):
    p = pd.read_csv(DATA / 'tasks_3_4_train.csv')
    p = p[p.rating > 0].reset_index(drop=True)
    u, users = pd.factorize(p.user)
    i, items = pd.factorize(p.item)
    N = len(p)
    oldorder = np.random.RandomState(42).permutation(N)
    a = int(0.9 * N)
    pool = oldorder[:a]
    if stage == 'validation':
        rows = pool
        query = p.iloc[oldorder[a:]].reset_index(drop=True)
        tune = 0
    else:
        rows = np.arange(N)
        query = pd.read_csv(DATA / 'task_3_test.csv')
        tune = 0
    qu = pd.Index(users).get_indexer(query.user).astype('int32')
    qi = pd.Index(items).get_indexer(query.item).astype('int32')
    U, I = (len(users), len(items))
    uu = u[rows].astype('int32')
    ii = i[rows].astype('int32')
    tt = p.timestamp.to_numpy(float)[rows]
    yy = p.rating.to_numpy(float)[rows]
    assert (yy > 0).all()
    sort = np.argsort(uu, kind='stable')
    uu = uu[sort]
    ii = np.ascontiguousarray(ii[sort])
    tt = tt[sort]
    yy = np.ascontiguousarray(yy[sort], dtype='f')
    count = np.bincount(uu, minlength=U)
    ic = np.ascontiguousarray(np.bincount(ii, minlength=I), dtype='int32')
    ptr = np.ascontiguousarray(np.r_[0, np.cumsum(count)], dtype='int32')
    fields, dev, qdev, center = time_arrays(uu, ii, tt, qu, qi, query.timestamp.to_numpy(float), U)
    codes = [c for c, _, _ in fields]
    counts = [np.ascontiguousarray(np.bincount(c), dtype='int32') for c in codes]
    H = coo_matrix((np.ones(len(uu), dtype='f'), (uu, ii)), shape=(U, I)).tocsr()
    if stage != 'full':
        assert np.asarray(H[qu, qi]).max() == 0, 'Held-out positive found in training history'
    return dict(U=U, I=I, u=uu, i=ii, t=tt, y=yy, ptr=ptr, ic=ic, uc=count, fields=fields, codes=codes, counts=counts, dev=dev, center=center, H=H, qu=qu, qi=qi, qt=query.timestamp.to_numpy(float), target=query.rating.to_numpy(float) if stage != 'full' else None, tune=tune, rows=len(rows), query=query)
def predict_state(s, u, i, t):
    u = np.asarray(u, dtype='int64')
    i = np.asarray(i, dtype='int64')
    good = (t >= 0) & (t <= 4102444800)
    day = np.where(good, np.floor(np.clip(t, 0, 4102444800) / 86400), -1).astype('int64')
    d = (t - s['center'][np.maximum(u, 0)]) / (86400 * 365.25)
    dev = np.where(good & (u >= 0), np.clip(np.sign(d) * abs(d) ** 0.4, -5, 5), 0).astype('f')
    out = np.full(len(u), float(s['mu']))
    vu = (u >= 0) & (s['uc'][np.maximum(u, 0)] > 0)
    vi = (i >= 0) & (s['ic'][np.maximum(i, 0)] > 0)
    ok = vu & vi
    out[vu] += s['bu'][u[vu]] + s['alpha'][u[vu]] * dev[vu]
    out[vi] += s['bi'][i[vi]]
    out[ok] += np.einsum('ij,ij->i', s['effective'][u[ok]] + dev[ok, None] * s['A'][u[ok]], s['Q'][i[ok]])
    keys = [(u + 1) * 1000000000 + day, (i + 1) * 1000000000 + np.where(good, day // 90, -1), (u + 1) * 1000000000 + np.where(good, day // 90, -1), np.where(good, day // 30, -1)]
    for j, labels in enumerate(keys):
        ix = pd.Index(s['keys' + str(j)]).get_indexer(labels)
        valid = ix >= 0
        out[valid] += s['term' + str(j)][ix[valid]]
    return clip(out)
def run_worker(k, stage, epochs=None):
    folder = OUT / str(k)
    folder.mkdir(exist_ok=True)
    cache = folder / (stage + '_scores.npz')
    report_path = folder / (stage + '_results.json')
    if cache.exists() and report_path.exists():
        return json.loads(report_path.read_text())
    d = make_data(stage)
    U, I = (d['U'], d['I'])
    rng = np.random.RandomState(47)
    std = 0.1 * (48 / k) ** 0.25
    P = rng.normal(0, std, (U, k)).astype('f')
    Q = rng.normal(0, std, (I, k)).astype('f')
    Y = np.zeros((I, k), dtype='f')
    A = np.zeros_like(P)
    bu = np.zeros(U, dtype='f')
    bi = np.zeros(I, dtype='f')
    terms = [np.zeros(len(c), dtype='f') for c in d['counts']]
    alpha = np.zeros(U, dtype='f')
    mu = float(d['y'].mean())
    lib = ctypes.CDLL(str((OUT / 'joint.so').resolve()))
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    fn = lib.joint_train
    fn.argtypes = [ctypes.c_int, ctypes.c_int] + [ip] * 6 + [fp] * 2 + [ip] * 5 + [ctypes.c_float] + [fp] * 11 + [ctypes.c_int] * 3 + [ctypes.c_float] * 5
    fn.restype = None
    checkpoints = list(range(10, epochs, 10)) + [epochs]
    last = 0
    started = time.time()
    state = None
    for stop in checkpoints:
        fn(U, k, d['ptr'], d['i'], *d['codes'], d['dev'], d['y'], d['ic'], *d['counts'], mu, P, Q, Y, A, bu, bi, *terms, alpha, last, stop - last, 47, 0.005, 0.08, 30.0, 40.0, 10.0)
        last = stop
        assert all((np.isfinite(v).all() for v in [P, Q, Y, A, bu, bi, alpha, *terms]))
        print('CAPACITY', k, stage, 'epoch', stop, 'elapsed', round(time.time() - started), flush=True)
        if stop == epochs:
            effective = np.asarray(P + d['H'] @ Y / np.sqrt(np.maximum(d['uc'], 1))[:, None], dtype='f')
            state = dict(effective=effective, Q=Q, A=A, bu=bu, bi=bi, alpha=alpha, mu=np.array(mu), center=d['center'], uc=d['uc'], ic=d['ic'], **{'keys' + str(j): f[2] for j, f in enumerate(d['fields'])}, **{'term' + str(j): v for j, v in enumerate(terms)})
            pred = predict_state(state, d['qu'], d['qi'], d['qt'])
    np.savez_compressed(cache, prediction=pred)
    r = dict(factors=k, stage=stage, epochs=epochs, train_rows=d['rows'], scored_rows=len(pred))
    r.update(parameters=dict(seed=47, learning_rate=0.005, factor_regularization=0.08, drift_factor_regularization=0.15, day_penalty=30, item_period_penalty=40, user_period_penalty=10, initial_std=std), seconds=round(time.time() - started, 2))
    report_path.write_text(json.dumps(r, indent=2))
    return r
''',
    'task3_components.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('SURPRISE_DATA_FOLDER',str(Path(__file__).resolve().parent/'artifacts'/'surprise_data'))
from task3_temporal_model import TemporalBias
def extra_features(train,query):
    cols=[query.timestamp.to_numpy()/86400/365.25]
    for key in ['user','item']:
        for field in ['rating','timestamp']:
            agg=train.groupby(key)[field].agg(['count','mean','std','min','max'])
            for stat in agg:
                value=query[key].map(agg[stat]).fillna(0).to_numpy()
                if field=='timestamp':value=value/86400/365.25
                cols.append(value)
            if field=='timestamp':
                mean=query[key].map(agg['mean']).fillna(train.timestamp.mean()).to_numpy()
                cols.append((query.timestamp.to_numpy()-mean)/86400/365.25)
    return np.nan_to_num(np.column_stack(cols),nan=0,posinf=0,neginf=0)
def generate(train,query,configs,user_index=None,item_index=None):
    users=pd.Index(pd.unique(train.user)) if user_index is None else pd.Index(user_index)
    items=pd.Index(pd.unique(train.item)) if item_index is None else pd.Index(item_index)
    u=users.get_indexer(train.user);i=items.get_indexer(train.item)
    qu=pd.Index(users).get_indexer(query.user);qi=pd.Index(items).get_indexer(query.item)
    t=train.timestamp.to_numpy(dtype=float);qt=query.timestamp.to_numpy(dtype=float);y=train.rating.to_numpy(dtype=float)
    U=len(users);I=len(items);out={};saved={}
    for name,config in configs.items():
        m=TemporalBias(config).fit(u,i,t,y,U);out[name]=np.clip(m.predict(qu,qi,qt),1,5)
        if name=='bias_static':saved['static']=m
        if name=='bias_temporal90_strong':saved['temporal']=m
    data=Dataset.load_from_df(train[['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset()
    for factors in [16,32]:
        print('SVD',factors,flush=True)
        m=SVD(n_factors=factors,n_epochs=50,reg_all=.1,random_state=42);m.fit(data)
        pred=np.array([m.predict(uu,ii).est for uu,ii in query[['user','item']].itertuples(index=False,name=None)])
        out['bias_svd'+str(factors)]=pred
        if factors==16:
            trainpred=np.array([m.predict(uu,ii).est for uu,ii in train[['user','item']].itertuples(index=False,name=None)])
            def monthkeys(series):
                d=pd.to_datetime(series,unit='s',errors='coerce',utc=True)
                return d.dt.year.fillna(-1)*12+d.dt.month.fillna(0)
            frame=pd.DataFrame({'key':monthkeys(train.timestamp).to_numpy(),'res':y-trainpred});agg=frame.groupby('key').res.agg(['sum','count']);correction=agg['sum']/(agg['count']+100)
            out['bias_svd_global_month']=np.clip(pred+monthkeys(query.timestamp).map(correction).fillna(0).to_numpy(),1,5)
    X=coo_matrix((np.ones(len(y),dtype=np.float32),(u,i)),shape=(U,I)).tocsr();G=X.T.tocsr();pop=np.asarray(X.sum(axis=0)).ravel()
    Times=coo_matrix((t,(u,i)),shape=(U,I)).tocsr()
    R={name:coo_matrix((y-m.training_prediction,(u,i)),shape=(U,I)).tocsr() for name,m in saved.items()}
    bases={name:m.predict(qu,qi,qt) for name,m in saved.items()}
    settings=[(20,1,0),(20,2,0),(20,1,365),(20,2,365),(20,1,90),(20,2,90)]
    for name,base in bases.items():
        for k,power,tau in settings:out['neighbor_'+name+'_'+str(k)+'_'+str(power)+'_'+str(tau)]=base.copy()
    groups={}
    for q,key in enumerate(qi):
        if key>=0 and qu[q]>=0:groups.setdefault(key,[]).append(q)
    keys=np.array(list(groups))
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).toarray()
        for row,key in enumerate(batch):
            cosine=co[row]/np.sqrt(np.maximum(pop[key]*pop,1));cosine[key]=0
            for q in groups[key]:
                lo,hi=X.indptr[qu[q]:qu[q]+2];sim=cosine[X.indices[lo:hi]];distance=abs(Times.data[lo:hi]-qt[q])/86400
                for k,power,tau in settings:
                    w=sim**power
                    if tau:w=w*np.exp(-distance/tau)
                    idx=np.argpartition(w,-k)[-k:] if len(w)>k else np.arange(len(w));weights=w[idx];den=weights.sum()+1e-12
                    for name in R:out['neighbor_'+name+'_'+str(k)+'_'+str(power)+'_'+str(tau)][q]+=(weights@R[name].data[lo:hi][idx])/den
    settings=[(tau,reg) for tau in [7,30,90,365] for reg in [1,5]]
    for name,base in bases.items():
        for tau,reg in settings:out['kernel_'+name+'_'+str(tau)+'_'+str(reg)]=base.copy()
    trainorder=np.argsort(u);tu=u[trainorder];starts=np.searchsorted(tu,np.arange(U+1));groups={}
    for q,user in enumerate(qu):
        if user>=0:groups.setdefault(user,[]).append(q)
    for user,qs in groups.items():
        rows=trainorder[starts[user]:starts[user+1]]
        if not len(rows):continue
        distance=abs(qt[qs,None]-t[rows][None,:])/86400
        for tau,reg in settings:
            w=np.exp(-distance/tau);den=w.sum(axis=1)+reg
            for name,m in saved.items():out['kernel_'+name+'_'+str(tau)+'_'+str(reg)][qs]+=w@(y[rows]-m.training_prediction[rows])/den
    return {name:np.clip(pred,1,5) for name,pred in out.items()}
''',
    'task3_configs.json': r'''{"bias_static":{"user":5,"item":10},"bias_drift_item":{"user":5,"item":10,"month":100,"drift":20,"item90":30},"bias_temporal180":{"user":5,"item":10,"month":100,"drift":20,"user180":10,"item180":30,"userday":10},"bias_temporal90":{"user":5,"item":10,"month":100,"drift":20,"user90":10,"item90":30,"userday":5},"bias_temporal90_strong":{"user":5,"item":10,"month":100,"drift":40,"user90":30,"item90":60,"userday":20},"grid_drift_u5_d2":{"user":5,"item":10,"month":100,"drift":2,"userday":20},"grid_drift_u5_d10":{"user":5,"item":10,"month":100,"drift":10,"userday":20},"grid_drift_u5_d40":{"user":5,"item":10,"month":100,"drift":40,"userday":20},"grid_drift_u15_d2":{"user":15,"item":10,"month":100,"drift":2,"userday":20},"grid_drift_u15_d10":{"user":15,"item":10,"month":100,"drift":10,"userday":20},"grid_drift_u15_d40":{"user":15,"item":10,"month":100,"drift":40,"userday":20},"grid_asymmetric_2":{"user":10,"item":10,"month":100,"drift_pos":2,"drift_neg":2,"userday":20},"grid_asymmetric_10":{"user":10,"item":10,"month":100,"drift_pos":10,"drift_neg":10,"userday":20},"grid_asymmetric_40":{"user":10,"item":10,"month":100,"drift_pos":40,"drift_neg":40,"userday":20},"grid_period_reg10":{"user":10,"item":10,"month":100,"drift":20,"user180":10,"item180":10,"userday":10},"grid_period_reg50":{"user":10,"item":10,"month":100,"drift":20,"user180":50,"item180":50,"userday":50},"grid_period_reg100":{"user":10,"item":10,"month":100,"drift":20,"user180":100,"item180":100,"userday":100}}''',
    'task3_experiment.py': r'''from final_imports import *
from task3_paths import DATA,ART
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('SURPRISE_DATA_FOLDER',str(ART/'surprise_data'))
from task3_temporal_model import TemporalBias
root=str(DATA)+'/'
p=pd.read_csv(root+'tasks_3_4_train.csv');p=p[p.rating>0].reset_index(drop=True)
u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);t=p.timestamp.to_numpy(dtype=float);y=p.rating.to_numpy(dtype=float)
order=np.random.RandomState(42).permutation(len(p));a=int(.9*len(p));b=int(.95*len(p));tr=order[:a];va=order[a:b];te=order[b:]
np.savez_compressed(str(ART/'split.npz'),order=order,users=u,items=i,time=t,y=y)
results={};out={};best=float('inf')
configs=[('static',{'user':5,'item':10}),('drift_item',{'user':5,'item':10,'month':100,'drift':20,'item90':30}),('temporal180',{'user':5,'item':10,'month':100,'drift':20,'user180':10,'item180':30,'userday':10}),('temporal90',{'user':5,'item':10,'month':100,'drift':20,'user90':10,'item90':30,'userday':5}),('temporal90_strong',{'user':5,'item':10,'month':100,'drift':40,'user90':30,'item90':60,'userday':20})]
for name,config in configs:
 start=time.time();m=TemporalBias(config).fit(u[tr],i[tr],t[tr],y[tr],len(users));v=np.clip(m.predict(u[va],i[va],t[va]),1,5);q=np.clip(m.predict(u[te],i[te],t[te]),1,5)
 results[name]={'selection_MSE':float(np.mean((v-y[va])**2)),'evaluation_MSE':float(np.mean((q-y[te])**2)),'config':config,'seconds':time.time()-start};out['v_'+name]=v;out['t_'+name]=q
 print(name,results[name],flush=True)
 if name=='temporal90_strong':
  with open(str(ART/'best_bias.pkl'),'wb') as f:pickle.dump(m,f)
 with open(str(ART/'results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ART/'predictions.npz'),**out)
data=Dataset.load_from_df(p.iloc[tr][['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset()
for factors in [16,32]:
 name='svd'+str(factors);m=SVD(n_factors=factors,n_epochs=50,reg_all=.1,random_state=42);m.fit(data)
 v=np.array([m.predict(uu,ii).est for uu,ii in p.iloc[va][['user','item']].itertuples(index=False,name=None)]);q=np.array([m.predict(uu,ii).est for uu,ii in p.iloc[te][['user','item']].itertuples(index=False,name=None)])
 results[name]={'selection_MSE':float(np.mean((v-y[va])**2)),'evaluation_MSE':float(np.mean((q-y[te])**2))};out['v_'+name]=v;out['t_'+name]=q
 print(name,results[name],flush=True)
 if factors==16:
  trainpred=np.array([m.predict(uu,ii).est for uu,ii in p.iloc[tr][['user','item']].itertuples(index=False,name=None)])
  dates=pd.to_datetime(p.timestamp,unit='s',errors='coerce',utc=True);months=dates.dt.year.fillna(-1)*12+dates.dt.month.fillna(0)
  frame=pd.DataFrame({'key':months.iloc[tr].to_numpy(),'res':y[tr]-trainpred});agg=frame.groupby('key').res.agg(['sum','count']);corr=agg['sum']/(agg['count']+100)
  vv=np.clip(v+months.iloc[va].map(corr).fillna(0).to_numpy(),1,5);qq=np.clip(q+months.iloc[te].map(corr).fillna(0).to_numpy(),1,5)
  results['svd_global_month']={'selection_MSE':float(np.mean((vv-y[va])**2)),'evaluation_MSE':float(np.mean((qq-y[te])**2))};out['v_svd_global_month']=vv;out['t_svd_global_month']=qq
  print('svd_global_month',results['svd_global_month'],flush=True)
 with open(str(ART/'results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ART/'predictions.npz'),**out)
''',
    'task3_grid.py': r'''from final_imports import *
from task3_paths import ART
os.environ['OPENBLAS_NUM_THREADS']='1'
from task3_temporal_model import TemporalBias
z=np.load(str(ART/'split.npz'));u,i,t,y,order=[z[k] for k in ['users','items','time','y','order']];a=int(.9*len(u));b=int(.95*len(u));tr=order[:a];va=order[a:b];te=order[b:];U=int(u.max()+1)
results={};out={}
configs=[]
for regu in [5,15]:
 for drift in [2,10,40]:
  configs.append(('drift_u'+str(regu)+'_d'+str(drift),{'user':regu,'item':10,'month':100,'drift':drift,'userday':20}))
for drift in [2,10,40]:
 configs.append(('asymmetric_'+str(drift),{'user':10,'item':10,'month':100,'drift_pos':drift,'drift_neg':drift,'userday':20}))
for reg in [10,50,100]:
 configs.append(('period_reg'+str(reg),{'user':10,'item':10,'month':100,'drift':20,'user180':reg,'item180':reg,'userday':reg}))
for name,config in configs:
 m=TemporalBias(config).fit(u[tr],i[tr],t[tr],y[tr],U);v=np.clip(m.predict(u[va],i[va],t[va]),1,5);q=np.clip(m.predict(u[te],i[te],t[te]),1,5)
 results[name]={'selection_MSE':float(np.mean((v-y[va])**2)),'evaluation_MSE':float(np.mean((q-y[te])**2)),'config':config};out['v_'+name]=v;out['t_'+name]=q
 print(name,results[name],flush=True)
 with open(str(ART/'grid_results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ART/'grid_predictions.npz'),**out)
''',
    'task3_joint_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
extern "C" void joint_train(int U,int k,const int* ptr,const int* items,const int* day,
 const int* iper,const int* uper,const int* month,const float* dev,const float* y,
 const int* ic,const int* dc,const int* ipc,const int* upc,const int* mc,float mu,
 float* P,float* Q,float* Y,float* A,float* bu,float* bi,float* du,float* bt,float* ut,float* mt,float* alpha,
 int start,int epochs,int seed,float lr,float reg,float rday,float rperiod,float ruserperiod){
 std::vector<int> users(U);std::vector<int> order;std::vector<float> z(k),initial(k);
 for(int ep=start;ep<start+epochs;ep++){
  std::iota(users.begin(),users.end(),0);std::mt19937 rng(seed+ep);std::shuffle(users.begin(),users.end(),rng);float eta=lr*std::pow(.985f,ep);
  for(int u:users){int n=ptr[u+1]-ptr[u];if(!n)continue;float norm=1.f/std::sqrt(float(n));std::fill(z.begin(),z.end(),0.f);
   for(int a=ptr[u];a<ptr[u+1];a++)for(int f=0;f<k;f++)z[f]+=Y[items[a]*k+f]*norm;initial=z;
   order.resize(n);std::iota(order.begin(),order.end(),ptr[u]);std::shuffle(order.begin(),order.end(),rng);
   for(int a:order){int j=items[a],d=day[a],b=iper[a],v=uper[a],m=month[a];float t=dev[a];float pred=mu+bu[u]+bi[j]+du[d]+bt[b]+ut[v]+mt[m]+alpha[u]*t;
    for(int f=0;f<k;f++)pred+=(P[u*k+f]+z[f]+t*A[u*k+f])*Q[j*k+f];float e=y[a]-pred;
    bu[u]+=eta*(e-3.f/n*bu[u]);bi[j]+=eta*(e-5.f/ic[j]*bi[j]);du[d]+=eta*(e-rday/dc[d]*du[d]);bt[b]+=eta*(e-rperiod/ipc[b]*bt[b]);ut[v]+=eta*(e-ruserperiod/upc[v]*ut[v]);mt[m]+=eta*(e-100.f/mc[m]*mt[m]);alpha[u]+=eta*(e*t-10.f/n*alpha[u]);
    for(int f=0;f<k;f++){int p=u*k+f,q=j*k+f;float oldq=Q[q],eff=P[p]+z[f]+t*A[p];P[p]+=eta*(e*oldq-reg*P[p]);A[p]+=eta*(e*oldq*t-.15f*A[p]);Q[q]+=eta*(e*eff-reg*oldq);z[f]+=eta*(e*oldq-reg*z[f]);}
   }
   float decay=std::pow(1.f-eta*reg,n);
   for(int a=ptr[u];a<ptr[u+1];a++)for(int f=0;f<k;f++){int j=items[a]*k+f;Y[j]=decay*Y[j]+(z[f]-decay*initial[f])*norm;}
  }
 }
}
''',
    'task3_kernels.py': r'''from final_imports import *
from task3_paths import ART
os.environ['OPENBLAS_NUM_THREADS']='1'
from task3_temporal_model import TemporalBias
z=np.load(str(ART/'split.npz'));u,i,t,y,order=[z[k] for k in ['users','items','time','y','order']];a=int(.9*len(u));b=int(.95*len(u));tr=order[:a];query=order[a:];U=int(u.max()+1)
models={'static':TemporalBias({'user':5,'item':10}).fit(u[tr],i[tr],t[tr],y[tr],U)}
with open(str(ART/'best_bias.pkl'),'rb') as f:models['temporal']=pickle.load(f)
config=[(tau,reg) for tau in [7,30,90,365] for reg in [1,5]]
pred={name+'_'+str(tau)+'_'+str(reg):m.predict(u[query],i[query],t[query]) for name,m in models.items() for tau,reg in config}
trainorder=np.argsort(u[tr]);tu=u[tr][trainorder];starts=np.searchsorted(tu,np.arange(U+1));groups={}
for q,user in enumerate(u[query]):groups.setdefault(user,[]).append(q)
for user,qs in groups.items():
 rows=trainorder[starts[user]:starts[user+1]]
 if not len(rows):continue
 dist=abs(t[query[qs]][:,None]-t[tr[rows]][None,:])/86400
 for tau,reg in config:
  w=np.exp(-dist/tau);den=w.sum(axis=1)+reg
  for name,m in models.items():
   residual=y[tr[rows]]-m.training_prediction[rows]
   pred[name+'_'+str(tau)+'_'+str(reg)][qs]+=w@residual/den
results={};out={}
for name,q in pred.items():
 q=np.clip(q,1,5);v=q[:b-a];e=q[b-a:]
 results[name]={'selection_MSE':float(np.mean((v-y[order[a:b]])**2)),'evaluation_MSE':float(np.mean((e-y[order[b:]])**2))};out['v_'+name]=v;out['t_'+name]=e
print(json.dumps(sorted(results.items(),key=lambda x:x[1]['selection_MSE'])[:8],indent=2),flush=True)
with open(str(ART/'kernel_results.json'),'w') as f:json.dump(results,f,indent=2)
np.savez_compressed(str(ART/'kernel_predictions.npz'),**out)
''',
    'task3_local_linear_features.py': r'''from final_imports import *
from task3_optimize import dataset
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
SETTINGS=[('user',30.,5.,20.),('user',90.,5.,20.),('user',365.,5.,20.),('item',90.,20.,80.),('item',365.,20.,80.)]
def moments_fit(dx, r, penalty, slope_penalty):
    w = np.exp(-abs(dx))
    s0 = w.sum(1)
    s1 = (w * dx).sum(1)
    s2 = (w * dx * dx).sum(1)
    r0 = w @ r
    r1 = w * dx @ r
    det = (s0 + penalty) * (s2 + slope_penalty) - s1 * s1
    intercept = (r0 * (s2 + slope_penalty) - r1 * s1) / det
    slope = (r1 * (s0 + penalty) - r0 * s1) / det
    return (r0 / (s0 + penalty), intercept, slope, s0)
def generate(out, full=False):
    file = out / ('full_features.npz' if full else 'validation_features.npz')
    if file.exists():
        z = np.load(file)
        return (z['features'], list(z['names']), list(map(int, z['constant_indices'])))
    p, tr, q, u, i, t, y, qu, qi, *_ = dataset(DATA, full)
    assert (y > 0).all()
    with open(W / 'task3_optimized' / ('full' if full else 'validation') / 'temporal_bias.pkl', 'rb') as f:
        base = pickle.load(f)
    assert len(base.training_prediction) == len(y)
    assert np.allclose(base.training_prediction, base.predict(u.astype('int64'), i.astype('int64'), t), atol=1e-07)
    residual = y - base.training_prediction
    bq = base.predict(qu.astype('int64'), qi.astype('int64'), q.timestamp.to_numpy(float))
    qt = q.timestamp.to_numpy(float) / 86400
    tt = t / 86400
    valid = (t >= 0) & (t <= 4102444800)
    qvalid = (q.timestamp.to_numpy(float) >= 0) & (q.timestamp.to_numpy(float) <= 4102444800)
    features = []
    names = []
    constant = []
    for key, tau, pen, spen in SETTINGS:
        vals = np.zeros((len(q), 5))
        train_groups = tr.groupby(key, sort=False).indices
        for k, ix in q.groupby(key, sort=False).indices.items():
            rows = train_groups.get(k)
            if rows is None:
                continue
            rows = rows[valid[rows]]
            qs = ix[qvalid[ix]]
            if not len(rows) or not len(qs):
                continue
            for start in range(0, len(qs), 64):
                take = qs[start:start + 64]
                dx = (tt[rows][None, :] - qt[take, None]) / tau
                c, b, s, support = moments_fit(dx, residual[rows], pen, spen)
                vals[take] = np.column_stack([c, b, b - c, s, np.log1p(support)])
        start = len(features)
        features.extend([np.clip(bq + vals[:, 0], 1, 5), np.clip(bq + vals[:, 1], 1, 5), vals[:, 2], vals[:, 3], vals[:, 4]])
        constant.extend([start, start + 4])
        names.extend([f'{kind}_{key}_{int(tau)}' for kind in ['local_constant', 'local_linear', 'linear_minus_constant', 'local_slope', 'local_support']])
        print('LOCAL_FEATURES', 'full' if full else 'validation', key, tau, flush=True)
    X = np.column_stack(features).astype('f')
    assert X.shape == (len(q), 25) and np.isfinite(X).all()
    np.savez_compressed(file, features=X, names=names, constant_indices=constant)
    return (X, names, constant)
''',
    'task3_mf_kernel.cpp': r'''#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <cmath>
extern "C" void train_mf(int n,int k,const int* u,const int* i,const int* period,
 const float* dev,const float* scale,const float* y,float* P,float* A,float* S,float* Q,
 int mode,int start,int epochs,int seed,float lr,float rp,float rq,float ra,float rs) {
 std::vector<int> order(n);
 for(int epoch=start;epoch<start+epochs;epoch++){
  std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+epoch);std::shuffle(order.begin(),order.end(),rng);
  float rate=lr*std::pow(.98f,epoch);
  for(int row:order){
   int uu=u[row],ii=i[row],bb=period[row];float d=mode?dev[row]:0.f;float w=mode?scale[row]:0.f;
   float* p=P+uu*k;float* a=A+uu*k;float* s=S+bb*k;float* q=Q+ii*k;float pred=0;
   for(int f=0;f<k;f++)pred+=(p[f]+d*a[f]+w*s[f])*q[f];
   float error=y[row]-pred;
   for(int f=0;f<k;f++){
    float oldq=q[f],eff=p[f]+d*a[f]+w*s[f];
    p[f]+=rate*(error*oldq-rp*p[f]);
    if(mode){a[f]+=rate*(error*oldq*d-ra*a[f]);s[f]+=rate*(error*oldq*w-rs*s[f]);}
    q[f]+=rate*(error*eff-rq*oldq);
   }
  }
 }
}
extern "C" void predict_mf(int n,int k,int nu,int ni,const int* u,const int* i,
 const int* period,const float* dev,const float* scale,const int* uc,const int* ic,
 const float* P,const float* A,const float* S,const float* Q,int mode,float* out){
 for(int z=0;z<n;z++){
  int uu=u[z],ii=i[z],b=period[z];float v=0;
  if(uu>=0&&uu<nu&&ii>=0&&ii<ni&&uc[uu]>0&&ic[ii]>0){
   for(int f=0;f<k;f++){
    float eff=P[uu*k+f];if(mode)eff+=dev[z]*A[uu*k+f]+(b>=0?scale[z]*S[b*k+f]:0.f);
    v+=eff*Q[ii*k+f];
   }
  }
  out[z]=v;
 }
}
''',
    'task3_neighbors.py': r'''from final_imports import *
from task3_paths import ART
os.environ['OPENBLAS_NUM_THREADS']='1'
from task3_temporal_model import TemporalBias
z=np.load(str(ART/'split.npz'));u,i,t,y,order=[z[k] for k in ['users','items','time','y','order']];U=int(u.max()+1);I=int(i.max()+1);a=int(.9*len(u));b=int(.95*len(u));tr=order[:a];query=order[a:]
static=TemporalBias({'user':5,'item':10}).fit(u[tr],i[tr],t[tr],y[tr],U)
with open(str(ART/'best_bias.pkl'),'rb') as f:temporal=pickle.load(f)
bases={'static':static.predict(u[query],i[query],t[query]),'temporal':temporal.predict(u[query],i[query],t[query])}
R={name:coo_matrix((y[tr]-m.training_prediction,(u[tr],i[tr])),shape=(U,I)).tocsr() for name,m in [('static',static),('temporal',temporal)]}
X=coo_matrix((np.ones(len(tr),dtype=np.float32),(u[tr],i[tr])),shape=(U,I)).tocsr();G=X.T.tocsr();pop=np.asarray(X.sum(axis=0)).ravel()
Times=coo_matrix((t[tr],(u[tr],i[tr])),shape=(U,I)).tocsr()
assert all(np.array_equal(m.indices,Times.indices) and np.array_equal(m.indptr,Times.indptr) for m in R.values())
configs=[(20,1,0),(20,2,0),(20,1,365),(20,2,365),(20,1,90),(20,2,90)]
preds={name+'_'+str(k)+'_'+str(power)+'_'+str(tau):base.copy() for name,base in bases.items() for k,power,tau in configs}
groups={}
for q,key in enumerate(i[query]):groups.setdefault(key,[]).append(q)
keys=np.array(list(groups));start=time.time()
for offset in range(0,len(keys),64):
 batch=keys[offset:offset+64];co=(G[batch]@X).toarray()
 for row,key in enumerate(batch):
  cosine=co[row]/np.sqrt(np.maximum(pop[key]*pop,1));cosine[key]=0
  for q in groups[key]:
   user=u[query[q]];lo,hi=X.indptr[user:user+2];hist=X.indices[lo:hi];sim=cosine[hist];distance=abs(Times.data[lo:hi]-t[query[q]])/86400
   for k,power,tau in configs:
    w=sim**power
    if tau:w=w*np.exp(-distance/tau)
    if len(w)>k:idx=np.argpartition(w,-k)[-k:]
    else:idx=np.arange(len(w))
    weights=w[idx];den=weights.sum()+1e-12
    for name in R:preds[name+'_'+str(k)+'_'+str(power)+'_'+str(tau)][q]+=(weights@R[name].data[lo:hi][idx])/den
 if offset%5000<64:print('Items',offset,'/',len(keys),'seconds',round(time.time()-start),flush=True)
results={};out={}
for name,p in preds.items():
 p=np.clip(p,1,5);v=p[:b-a];q=p[b-a:]
 results[name]={'selection_MSE':float(np.mean((v-y[order[a:b]])**2)),'evaluation_MSE':float(np.mean((q-y[order[b:]])**2))};out['v_'+name]=v;out['t_'+name]=q
print(json.dumps(sorted(results.items(),key=lambda x:x[1]['selection_MSE']),indent=2),flush=True)
with open(str(ART/'neighbors_results.json'),'w') as f:json.dump(results,f,indent=2)
np.savez_compressed(str(ART/'neighbors_predictions.npz'),**out)
''',
    'task3_optimize.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from task3_temporal_model import TemporalBias
CONFIG={'user':5,'item':10,'month':100,'drift':40,'user90':30,'item90':60,'userday':20}
def log(*a):print(*a,flush=True)
def mse(y,p):return float(np.mean((y-p)**2))
def clip(p):return np.clip(p,1,5)
def sparse(v,u,i,shape):return coo_matrix((v,(u,i)),shape=shape).tocsr()
def dataset(data,full):
    p=pd.read_csv(data/'tasks_3_4_train.csv');p=p[p.rating>0].reset_index(drop=True)
    u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);t=p.timestamp.to_numpy(float);y=p.rating.to_numpy(float)
    order=np.random.RandomState(42).permutation(len(p));a=int(.9*len(p));b=int(.95*len(p));idx=np.arange(len(p)) if full else order[:a]
    query=pd.read_csv(data/'task_3_test.csv') if full else p.iloc[order[a:]].reset_index(drop=True)
    qu=pd.Index(users).get_indexer(query.user);qi=pd.Index(items).get_indexer(query.item)
    return p,p.iloc[idx],query,u[idx].astype(np.int32),i[idx].astype(np.int32),t[idx],y[idx],qu.astype(np.int32),qi.astype(np.int32),users,items,a,b
def smooth(train,query,residual,key,tau,penalty):
    groups={k:ix for k,ix in train.groupby(key,sort=False).indices.items()};out=np.zeros(len(query))
    t=train.timestamp.to_numpy(float)/86400;qt=query.timestamp.to_numpy(float)/86400
    for k,qs in query.groupby(key,sort=False).indices.items():
        rows=groups.get(k)
        if rows is None:continue
        rows=rows[(t[rows]>=0)&(t[rows]<=4102444800/86400)]
        if not len(rows):continue
        for ix in np.array_split(qs,max(1,(len(qs)+127)//128)):
            w=np.exp(-np.abs(qt[ix,None]-t[rows][None,:])/tau);out[ix]=w@residual[rows]/(w.sum(1)+penalty)
    out[(qt<0)|(qt>4102444800/86400)]=0;return out
def library_worker(data,out,full):
    target=out/'library_features.npz'
    if target.exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);start=time.time()
    training=Dataset.load_from_df(tr[['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset()
    m=SVDpp(n_factors=24,n_epochs=25,lr_all=.007,reg_all=.08,random_state=42,cache_ratings=True);m.fit(training)
    U=pd.Index([training.to_raw_uid(k) for k in range(training.n_users)]);I=pd.Index([training.to_raw_iid(k) for k in range(training.n_items)])
    cu=U.get_indexer(tr.user);ci=I.get_indexer(tr.item);H=sparse(np.ones(len(tr)),cu,ci,(len(U),len(I)));count=np.asarray(H.sum(1)).ravel()
    effective=m.pu+(H@m.yj)/np.sqrt(np.maximum(count,1))[:,None]
    def predict(frame):
        uu=U.get_indexer(frame.user);ii=I.get_indexer(frame.item);v=np.full(len(frame),training.global_mean);vu=uu>=0;vi=ii>=0;ok=vu&vi
        v[vu]+=m.bu[uu[vu]];v[vi]+=m.bi[ii[vi]];v[ok]+=np.einsum('ij,ij->i',effective[uu[ok]],m.qi[ii[ok]]);return clip(v)
    pred=predict(q);tp=predict(tr)
    sample=np.random.RandomState(42).choice(len(q),min(128,len(q)),replace=False);reference=np.array([m.predict(q.user.iloc[k],q.item.iloc[k]).est for k in sample]);assert np.allclose(pred[sample],reference,atol=1e-10)
    r=y-tp;du=smooth(tr,q,r,'user',30,3);di=smooth(tr,q,r,'item',90,20)
    np.savez_compressed(target,svdpp24=pred,svdpp_user30=clip(pred+du),svdpp_temporal=clip(pred+du+di))
    np.savez_compressed(out/'svdpp_model.npz',users=U.to_numpy(dtype=str),items=I.to_numpy(dtype=str),effective=effective,qi=m.qi,bu=m.bu,bi=m.bi,mu=training.global_mean)
    log('Library SVD++ + temporal corrections complete; seconds',round(time.time()-start,1))
def mf_library(out):
    so=out/'mf_kernel.so'
    if not so.exists():subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_mf_kernel.cpp')),'-o',str(so)],check=True)
    lib=ctypes.CDLL(str(so.resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.train_mf.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,fp,fp,fp,fp,fp,fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float];lib.train_mf.restype=None
    lib.predict_mf.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,fp,ip,ip,fp,fp,fp,fp,ctypes.c_int,fp];lib.predict_mf.restype=None
    return lib
def context_features(train,query,base_train,base_query):
    result={};qt=query.timestamp.to_numpy(float)/86400;tt=train.timestamp.to_numpy(float)/86400;y=train.rating.to_numpy(float);good=(tt>=0)&(tt<=4102444800/86400);res=y-base_train
    result['valid_query_date']=((qt>=0)&(qt<=4102444800/86400)).astype(float);result['query_year']=np.clip(qt/365.25,0,130);result['year_sin']=np.sin(2*np.pi*qt/365.25);result['year_cos']=np.cos(2*np.pi*qt/365.25)
    for key in ['user','item']:
        groups={k:ix for k,ix in train.groupby(key,sort=False).indices.items()}
        names=['prev_rating','next_rating','prev_gap','next_gap','has_prev','has_next','rating_entropy']
        for day in [0,1,7,30,90,365]:names += ['mean'+str(day),'std'+str(day),'count'+str(day),'residual'+str(day)]
        for value in range(1,6):names.append('fraction'+str(value))
        for n in names:result[key+'_'+n]=np.zeros(len(query))
        for name,qs in query.groupby(key,sort=False).indices.items():
            rows=groups.get(name)
            if rows is None:continue
            vals=y[rows];freq=np.bincount(vals.astype(int),minlength=6)[1:6]/len(vals)
            for value in range(1,6):result[key+'_fraction'+str(value)][qs]=freq[value-1]
            result[key+'_rating_entropy'][qs]=-np.sum(freq*np.log(np.maximum(freq,1e-12)))
            rows=rows[good[rows]];rows=rows[np.argsort(tt[rows],kind='stable')]
            if not len(rows):continue
            dates=tt[rows];v=y[rows];rr=res[rows];c=np.r_[0,np.cumsum(v)];c2=np.r_[0,np.cumsum(v*v)];cr=np.r_[0,np.cumsum(rr)];q=qt[qs];pos=np.searchsorted(dates,q)
            for side,ids,ok in [('prev',np.maximum(pos-1,0),pos>0),('next',np.minimum(pos,len(rows)-1),pos<len(rows))]:
                result[key+'_'+side+'_rating'][qs]=np.where(ok,v[ids],0);result[key+'_'+side+'_gap'][qs]=np.log1p(np.where(ok,np.abs(q-dates[ids]),100000));result[key+'_has_'+side][qs]=ok
            for h in [0,1,7,30,90,365]:
                if h==0:
                    lo=np.searchsorted(dates,np.floor(q),side='left');hi=np.searchsorted(dates,np.floor(q)+1,side='left')
                else:lo=np.searchsorted(dates,q-h);hi=np.searchsorted(dates,q+h,side='right')
                n=hi-lo;total=c[hi]-c[lo];mean=np.divide(total,n,out=np.zeros(len(n)),where=n>0);var=np.maximum(0,np.divide(c2[hi]-c2[lo],n,out=np.zeros(len(n)),where=n>0)-mean**2)
                result[key+'_mean'+str(h)][qs]=mean;result[key+'_std'+str(h)][qs]=np.sqrt(var);result[key+'_count'+str(h)][qs]=np.log1p(n)
                result[key+'_residual'+str(h)][qs]=(cr[hi]-cr[lo])/(n+(3 if key=='user' else 20))
    return result
def user_neighbors(u,i,t,y,qu,qi,qt,base_train,U,I):
    shape=(U,I);X=sparse(np.ones(len(u),np.float32),u,i,shape);up=np.asarray(X.sum(1)).ravel();mean=np.bincount(u,weights=y,minlength=U)/np.maximum(up,1)
    C=sparse(y-mean[u],u,i,shape);R=sparse(y-base_train,u,i,shape).tocsc();Times=sparse(t,u,i,shape).tocsc();norm=np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel());result={}
    for sim in ['binary','pearson']:
        for tau in [0,365]:
            for k in [20,50]:result[f'usernn_{sim}_{tau}_{k}']=np.zeros(len(qu))
        result['usernn_'+sim+'_support']=np.zeros(len(qu))
    groups={}
    for a,user in enumerate(qu):
        if user>=0 and qi[a]>=0:groups.setdefault(user,[]).append(a)
    keys=np.array(list(groups));XT=X.T.tocsr();CT=C.T.tocsr()
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(X[batch]@XT).toarray();dot=(C[batch]@CT).toarray()
        for row,user in enumerate(batch):
            binary=co[row]/np.sqrt(np.maximum(up[user]*up,1));pearson=np.maximum(0,dot[row]/np.maximum(norm[user]*norm,1e-8));shrink=co[row]/(co[row]+10);binary*=shrink;pearson*=shrink;binary[user]=0;pearson[user]=0
            for a in groups[user]:
                j=qi[a];lo,hi=R.indptr[j:j+2];other=R.indices[lo:hi];res=R.data[lo:hi];distance=np.abs(Times.data[lo:hi]-qt[a])/86400;valid=(Times.data[lo:hi]>=0)&(Times.data[lo:hi]<=4102444800)
                for name,sim in [('binary',binary),('pearson',pearson)]:
                    w0=sim[other];result['usernn_'+name+'_support'][a]=w0.sum()
                    for tau in [0,365]:
                        w=w0 if tau==0 else w0*np.exp(-distance/tau)*valid
                        for k in [20,50]:
                            ix=np.argpartition(w,-k)[-k:] if len(w)>k else np.arange(len(w));result[f'usernn_{name}_{tau}_{k}'][a]=(w[ix]@res[ix])/(w[ix].sum()+.25)
        if offset%4096==0:log('User-neighborhood rows',offset,'/',len(keys))
    return result
def temporal_worker(data,out,full):
    target=out/'temporal_features.npz'
    if target.exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items);start=time.time();base=TemporalBias(CONFIG).fit(u.astype(np.int64),i.astype(np.int64),t,y,U);bq=base.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));features={'temporal_base':clip(bq)}
    def dev(uu,tt):
        valid=(tt>=0)&(tt<=4102444800)&(uu>=0);d=(tt-base.mean_time[np.maximum(uu,0)])/(86400*365.25);return np.ascontiguousarray(np.where(valid,np.clip(np.sign(d)*np.abs(d)**.4,-5,5),0),dtype=np.float32)
    def keys(uu,tt):return (uu.astype(np.int64)+1)*1000000000+np.where((tt>=0)&(tt<=4102444800),np.floor(np.clip(tt,0,4102444800)/86400/180),-1).astype(np.int64)
    period,index=pd.factorize(keys(u,t));period=np.ascontiguousarray(period,np.int32);qp=np.ascontiguousarray(pd.Index(index).get_indexer(keys(qu,q.timestamp.to_numpy(float))),np.int32);count=np.bincount(period);scale=np.ascontiguousarray(np.sqrt(count[period]/(count[period]+10)),np.float32);qs=np.zeros(len(q),np.float32);ok=qp>=0;qs[ok]=np.sqrt(count[qp[ok]]/(count[qp[ok]]+10));dv=dev(u,t);qd=dev(qu,q.timestamp.to_numpy(float));res=np.ascontiguousarray(y-base.training_prediction,np.float32);uc=np.ascontiguousarray(np.bincount(u,minlength=U),np.int32);ic=np.ascontiguousarray(np.bincount(i,minlength=I),np.int32);lib=mf_library(out)
    for name,k,mode,epochs,lr,reg,seed in [('residual_mf32',32,0,[30,60],.01,.06,42),('time_mf32',32,1,[30,60],.008,.08,17),('time_mf64',64,1,[30,50],.007,.1,23)]:
        rng=np.random.RandomState(seed);P=(rng.normal(0,.05,(U,k))).astype(np.float32);Q=(rng.normal(0,.05,(I,k))).astype(np.float32);A=np.zeros_like(P);S=np.zeros((len(index),k),np.float32);last=0
        for epoch in epochs:
            lib.train_mf(len(u),k,u,i,period,dv,scale,res,P,A,S,Q,mode,last,epoch-last,seed,lr,reg,reg,.3,.2);last=epoch
            assert all(np.isfinite(v).all() for v in [P,Q,A,S]);pred=np.zeros(len(q),np.float32);lib.predict_mf(len(q),k,U,I,qu,qi,qp,qd,qs,uc,ic,P,A,S,Q,mode,pred);features[name+'_e'+str(epoch)]=clip(bq+pred)
            np.savez_compressed(out/(name+'_e'+str(epoch)+'.npz'),P=P,Q=Q,A=A,S=S,period_keys=np.asarray(index),mean_time=base.mean_time,uc=uc,ic=ic,mode=mode)
            log(name,'checkpoint',epoch,'seconds elapsed',round(time.time()-start,1))
    features.update(context_features(tr,q,base.training_prediction,bq));features.update(user_neighbors(u,i,t,y,qu,qi,q.timestamp.to_numpy(float),base.training_prediction,U,I))
    assert all(np.isfinite(v).all() for v in features.values());np.savez_compressed(target,**features)
    with open(out/'temporal_bias.pkl','wb') as f:pickle.dump(base,f)
    log('Temporal MF, rating context and user neighborhoods complete')
def baseline_validation(data,legacy):
    from task3_components import extra_features
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,False);split=np.load(legacy/'split.npz');assert np.array_equal(split['y'],p.rating.to_numpy()) and np.array_equal(split['time'],p.timestamp.to_numpy(float));assert np.array_equal(split['order'],np.random.RandomState(42).permutation(len(p)))
    parts={}
    for prefix,file in [('bias','predictions'),('grid','grid_predictions'),('neighbor','neighbors_predictions'),('kernel','kernel_predictions')]:
        z=np.load(legacy/(file+'.npz'))
        for key in z.files:
            if key.startswith('v_'):parts[prefix+'_'+key[2:]]=np.r_[z[key],z['t_'+key[2:]]]
    names=json.loads((legacy/'meta_results.json').read_text())['base_columns'];F=np.column_stack([parts[k] for k in names]);F=np.c_[F,extra_features(tr,q)];return F,q.rating.to_numpy(),b-a
def baseline_full(data,out):
    target=out/'baseline_features.npz'
    if target.exists():return
    from task3_components import generate,extra_features
    p,tr,q,*rest=dataset(data,True);configs=json.loads(Path(__file__).with_name('task3_configs.json').read_text());components=generate(tr,q,configs)
    names=['bias_static','bias_drift_item','bias_temporal180','bias_temporal90','bias_temporal90_strong','bias_svd16','bias_svd_global_month','bias_svd32']+[k for k in configs if k.startswith('grid_')]
    names += [f'neighbor_{kind}_20_{power}_{tau}' for kind in ['static','temporal'] for tau in [0,365,90] for power in [1,2]]
    names += [f'kernel_{kind}_{tau}_{reg}' for kind in ['static','temporal'] for tau in [7,30,90,365] for reg in [1,5]]
    F=np.column_stack([components[k] for k in names]);F=np.c_[F,extra_features(tr,q)];np.savez_compressed(target,features=F)
    log('Full-data baseline component cache complete')
if __name__ == '__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--worker',choices=['library','temporal','baseline'],required=True);ap.add_argument('--full',action='store_true');a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.worker=='baseline':baseline_full(a.data_dir,a.output_dir)
 else:{'library':library_worker,'temporal':temporal_worker}[a.worker](a.data_dir,a.output_dir,a.full)
''',
    'task3_paths.py': r'''from final_imports import *
HERE=Path(__file__).resolve().parent
DATA=Path(os.environ.get('TASK3_DATA_DIR',str(HERE.parent/'Assignment 1/assignment1_2026')))
ART=Path(os.environ.get('TASK3_ARTIFACTS_DIR',str(HERE/'artifacts'/'task3')))
ART.mkdir(parents=True,exist_ok=True)
''',
    'task3_pearson_kernel.cpp': r'''#include <cmath>
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
''',
    'task3_refresh_features.py': r'''from final_imports import *
def context(train, q):
    qt = q.timestamp.to_numpy(float) / 86400
    valid = (qt >= 0) & (qt <= 47482)
    day = np.where(valid, qt, 0)
    cols = [valid.astype(float), np.sin(day * 2 * np.pi / 365.25), np.cos(day * 2 * np.pi / 365.25), np.sin(day * 2 * np.pi / 7), np.cos(day * 2 * np.pi / 7)]
    names = ['valid_timestamp', 'year_sin', 'year_cos', 'week_sin', 'week_cos']
    for key in ['user', 'item']:
        a = train.groupby(key).rating.agg(['count', 'std'])
        count = q[key].map(a['count']).fillna(0).to_numpy()
        cols.extend([np.log1p(count), q[key].map(a['std']).fillna(0).to_numpy()])
        names.extend([key + '_log_count', key + '_rating_std'])
        good = train[(train.timestamp >= 0) & (train.timestamp <= 4102444800)]
        times = {k: np.sort(v.timestamp.to_numpy(float) / 86400) for k, v in good.groupby(key, sort=False)}
        vals = np.zeros((len(q), 5), dtype='f')
        for k, ix in q.groupby(key, sort=False).indices.items():
            ts = times.get(k)
            if ts is None:
                continue
            t = qt[ix]
            j = np.searchsorted(ts, t)
            left = ts[np.maximum(0, j - 1)]
            right = ts[np.minimum(len(ts) - 1, j)]
            before = np.where(j > 0, t - left, 36525)
            after = np.where(j < len(ts), right - t, 36525)
            vals[ix] = np.column_stack([np.log1p(np.clip(before, 0, 36525)), np.log1p(np.clip(after, 0, 36525)), np.log1p(np.clip(np.minimum(before, after), 0, 36525)), j / len(ts), ((t < ts[0]) | (t > ts[-1])).astype(float)])
        vals[~valid] = 0
        cols.extend(vals.T)
        names.extend([key + '_' + x for x in ['prev_gap', 'next_gap', 'nearest_gap', 'time_percentile', 'outside_history']])
    return (np.column_stack(cols).astype('f'), names)
def augment(X, names, train, q):
    c, cn = context(train, q)
    indices = [j for j, k in enumerate(names) if (j < 48 or 71 <= j < 81) or (not k.startswith('prior_') and 'support' not in k)]
    P = X[:, indices]
    d = np.column_stack([P.mean(1), P.std(1), P.max(1) - P.min(1), np.quantile(P, 0.25, axis=1), np.quantile(P, 0.75, axis=1)]).astype('f')
    return (np.c_[X, c, d].astype('f'), list(names) + cn + ['model_mean', 'model_std', 'model_range', 'model_q25', 'model_q75'])
def estimator(count):
    return lgb.LGBMRegressor(objective='regression', n_estimators=count, num_leaves=63, learning_rate=0.025, min_child_samples=200, reg_lambda=30, reg_alpha=0.1, max_bin=127, colsample_bytree=0.9, subsample=0.9, subsample_freq=1, n_jobs=2, random_state=42, verbosity=-1)
''',
    'task3_round2.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from task3_optimize import dataset, clip, baseline_validation
def log(*a):print(*a,flush=True)
def codes(train,query):
 c,index=pd.factorize(train);return np.ascontiguousarray(c,np.int32),pd.Index(index).get_indexer(query),np.asarray(index)
def time_arrays(u,i,t,qu,qi,qt,U):
 good=(t>=0)&(t<=4102444800);qgood=(qt>=0)&(qt<=4102444800);day=np.where(good,np.floor(np.clip(t,0,4102444800)/86400),-1).astype(np.int64);qd=np.where(qgood,np.floor(np.clip(qt,0,4102444800)/86400),-1).astype(np.int64)
 center=np.bincount(u[good],weights=t[good],minlength=U)/np.maximum(np.bincount(u[good],minlength=U),1);fallback=np.mean(t[good]);center[np.bincount(u[good],minlength=U)==0]=fallback
 def dev(uu,tt,gg):
  d=(tt-center[np.maximum(uu,0)])/(86400*365.25);return np.ascontiguousarray(np.where(gg&(uu>=0),np.clip(np.sign(d)*np.abs(d)**.4,-5,5),0),np.float32)
 fields=[((u.astype(np.int64)+1)*1000000000+day,(qu.astype(np.int64)+1)*1000000000+qd),((i.astype(np.int64)+1)*1000000000+np.where(good,day//90,-1),(qi.astype(np.int64)+1)*1000000000+np.where(qgood,qd//90,-1)),((u.astype(np.int64)+1)*1000000000+np.where(good,day//90,-1),(qu.astype(np.int64)+1)*1000000000+np.where(qgood,qd//90,-1)),(np.where(good,day//30,-1),np.where(qgood,qd//30,-1))]
 return [codes(a,b) for a,b in fields],dev(u,t,good),dev(qu,qt,qgood),center
def joint_worker(data,out,full):
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items);qt=q.timestamp.to_numpy(float);order=np.argsort(u,kind='stable');u=u[order];i=np.ascontiguousarray(i[order],np.int32);t=t[order];y=np.ascontiguousarray(y[order],np.float32);count=np.bincount(u,minlength=U);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(count)],np.int32);ic=np.ascontiguousarray(np.bincount(i,minlength=I),np.int32)
 fields,dv,qd,center=time_arrays(u,i,t,qu,qi,qt,U);cs=[c for c,_,_ in fields];qc=[c for _,c,_ in fields];counts=[np.ascontiguousarray(np.bincount(c),np.int32) for c in cs];mu=float(np.mean(y));H=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();so=out/'joint.so'
 if not so.exists():subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_joint_kernel.cpp')),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so.resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.joint_train.argtypes=[ctypes.c_int,ctypes.c_int]+[ip]*6+[fp]*2+[ip]*5+[ctypes.c_float]+[fp]*11+[ctypes.c_int]*3+[ctypes.c_float]*5;lib.joint_train.restype=None;target=out/'joint_features.npz';features=dict(np.load(target)) if target.exists() else {}
 for name,k,reg,rday,rperiod,lr,seed,epochs in [('joint24',24,.025,10.,20.,.007,42,[40,80]),('joint48',48,.035,10.,30.,.006,17,[40,80]),('joint24weak',24,.01,30.,10.,.005,23,[40,80]),('joint24strong',24,.12,20.,40.,.007,31,[15,30,50]),('joint48strong',48,.08,30.,40.,.005,47,[15,30,50]),('joint16strong',16,.2,30.,60.,.008,11,[15,30,50])]:
  if all(name+'_e'+str(epoch) in features for epoch in epochs):continue
  rng=np.random.RandomState(seed);P=rng.normal(0,.1,(U,k)).astype(np.float32);Q=rng.normal(0,.1,(I,k)).astype(np.float32);Y=np.zeros((I,k),np.float32);A=np.zeros((U,k),np.float32);bu=np.zeros(U,np.float32);bi=np.zeros(I,np.float32);du,bt,ut,mt=[np.zeros(len(c),np.float32) for c in counts];alpha=np.zeros(U,np.float32);last=0
  for epoch in epochs:
   lib.joint_train(U,k,ptr,i,*cs,dv,y,ic,*counts,mu,P,Q,Y,A,bu,bi,du,bt,ut,mt,alpha,last,epoch-last,seed,lr,reg,rday,rperiod,10.);last=epoch
   assert all(np.isfinite(v).all() for v in [P,Q,Y,A,bu,bi,du,bt,ut,mt,alpha])
   effective=P+(H@Y)/np.sqrt(np.maximum(count,1))[:,None];pred=np.full(len(q),mu);vu=(qu>=0)&(count[np.maximum(qu,0)]>0);vi=(qi>=0)&(ic[np.maximum(qi,0)]>0);ok=vu&vi;pred[vu]+=bu[qu[vu]]+alpha[qu[vu]]*qd[vu];pred[vi]+=bi[qi[vi]]
   pred[ok]+=np.einsum('ij,ij->i',effective[qu[ok]]+qd[ok,None]*A[qu[ok]],Q[qi[ok]])
   for c,coeff in zip(qc,[du,bt,ut,mt]):valid=c>=0;pred[valid]+=coeff[c[valid]]
   features[name+'_e'+str(epoch)]=clip(pred);np.savez_compressed(out/(name+'_e'+str(epoch)+'.npz'),P=P,Q=Q,Y=Y,A=A,effective=effective,bu=bu,bi=bi,du=du,bt=bt,ut=ut,mt=mt,alpha=alpha,mu=mu,center=center,uc=count,ic=ic,**{'keys'+str(j):f[2] for j,f in enumerate(fields)})
   log(name,epoch,'complete')
 temp=out/'joint_features.partial.npz';np.savez_compressed(temp,**features);temp.replace(out/'joint_features.npz')
def encodings(codes,queries,y,rng):
 n=len(y);fold=rng.randint(5,size=n);train=[];test=[]
 for c,qc,penalty in zip(codes,queries,[5.,10.,5.,10.]):
  size=int(max(c.max(),qc.max(initial=-1))+1);total=np.bincount(c,minlength=size);s=np.bincount(c,weights=y,minlength=size);ss=np.bincount(c,weights=y*y,minlength=size);low=np.bincount(c,weights=y<=2,minlength=size);high=np.bincount(c,weights=y>=4,minlength=size);A=np.zeros((n,5),np.float32)
  def values(count,sums,squares,lo,hi,prior):
   mean=(sums+penalty*prior)/(count+penalty);sd=np.sqrt(np.maximum(0,(squares+penalty*(prior*prior+1))/(count+penalty)-mean*mean));return np.c_[mean,sd,np.log1p(count),(lo+1)/(count+5),(hi+3)/(count+5)]
  for k in range(5):
   rows=np.flatnonzero(fold==k);count=total-np.bincount(c[rows],minlength=size);sums=s-np.bincount(c[rows],weights=y[rows],minlength=size);squares=ss-np.bincount(c[rows],weights=y[rows]*y[rows],minlength=size);lo=low-np.bincount(c[rows],weights=y[rows]<=2,minlength=size);hi=high-np.bincount(c[rows],weights=y[rows]>=4,minlength=size);prior=np.mean(y[fold!=k]);A[rows]=values(count,sums,squares,lo,hi,prior)[c[rows]]
  V=values(total,s,ss,low,high,np.mean(y));B=np.tile(np.array([np.mean(y),1,0,.2,.6],np.float32),(len(qc),1));ok=qc>=0;B[ok]=V[qc[ok]];train.append(A);test.append(B)
 return np.c_[tuple(train)],np.c_[tuple(test)]
def raw_worker(data,out,full):
 if (out/'raw_features.npz').exists():return
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items);X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();up=np.asarray(X.sum(1)).ravel();pop=np.asarray(X.sum(0)).ravel();idf=np.log1p(U/np.maximum(pop,1)).astype(np.float32);weighted=X.multiply(idf).multiply((1/np.sqrt(np.maximum(up,1)))[:,None]).tocsr();svd=TruncatedSVD(n_components=24,n_iter=5,random_state=42);svd.fit(weighted);Q=svd.components_.T.astype(np.float32);user_sum=X.multiply(idf)@Q
 qt=q.timestamp.to_numpy(float);good=(t>=0)&(t<=4102444800);qgood=(qt>=0)&(qt<=4102444800);bin=np.where(good,np.floor(np.clip(t,0,4102444800)/(365*86400)),-1).astype(np.int64);qb=np.where(qgood,np.floor(np.clip(qt,0,4102444800)/(365*86400)),-1).astype(np.int64)
 uc,quc,ukey=codes((u.astype(np.int64)+1)*1000000000+bin,(qu.astype(np.int64)+1)*1000000000+qb);ic,qic,ikey=codes((i.astype(np.int64)+1)*1000000000+bin,(qi.astype(np.int64)+1)*1000000000+qb)
 te,ve=encodings([u,i,uc,ic],[qu,qi,quc,qic],y,np.random.RandomState(19))
 def design(uu,ii,tt,bins,enc,training):
  knownu=uu>=0;knowni=ii>=0;profile=np.zeros((len(uu),24),np.float32);book=np.zeros_like(profile);profile[knownu]=user_sum[uu[knownu]];book[knowni]=Q[ii[knowni]]
  present=np.zeros(len(uu),bool);ok=knownu&knowni
  if training:present[:]=True
  else:present[ok]=np.asarray(X[uu[ok],ii[ok]]).ravel()>0
  ix=np.flatnonzero(present);profile[ix]-=idf[ii[ix],None]*Q[ii[ix]];profile/=np.maximum(np.linalg.norm(profile,axis=1,keepdims=True),1e-8);book/=np.maximum(np.linalg.norm(book,axis=1,keepdims=True),1e-8)
  days=np.clip(tt,0,4102444800)/86400;numeric=np.c_[enc,profile,book,profile*book,days/365.25,np.sin(days/365.25*2*np.pi),np.cos(days/365.25*2*np.pi)].astype(np.float32);frame=pd.DataFrame(numeric,columns=['x'+str(k) for k in range(numeric.shape[1])]);frame['user']=[str(v) for v in uu];frame['item']=[str(v) for v in ii];frame['user_year']=[str(v)+':'+str(z) for v,z in zip(uu,bins)];frame['item_year']=[str(v)+':'+str(z) for v,z in zip(ii,bins)];return frame
 A=design(u,i,t,bin,te,True);B=design(qu,qi,qt,qb,ve,False)
 cats=['user','item','user_year','item_year']
 for key in cats:
  A[key]=A[key].astype('category');B[key]=pd.Categorical(B[key],categories=A[key].cat.categories)
 log('LightGBM raw fit',A.shape)
 m=lgb.LGBMRegressor(n_estimators=600,num_leaves=31,max_bin=63,learning_rate=.06,min_child_samples=150,reg_lambda=15,colsample_bytree=.85,subsample=.85,subsample_freq=1,cat_smooth=20,max_cat_threshold=32,path_smooth=10,n_jobs=2,random_state=42,verbosity=-1)
 A=A.drop(columns=cats);B=B.drop(columns=cats)
 m.fit(A,y,eval_set=[(A,y)],eval_metric='l2',callbacks=[lgb.log_evaluation(100)])
 m.booster_.save_model(str(out/'raw_ratings.txt'));features={f'raw_lgbm_e{epoch}':clip(m.predict(B,num_iteration=epoch)) for epoch in [300,600]};np.savez_compressed(out/'raw_features.npz',**features)
 np.savez_compressed(out/'raw_query_design.npz',numeric=B.to_numpy(np.float32));np.savez_compressed(out/'unsupervised_embeddings.npz',Q=Q,idf=idf);log('Direct rating learner complete')
def learned_worker(data,out,full,prior,wide=False):
 prefix='wide_' if wide else '';target=out/('learnedwide_features.npz' if wide else 'learned_features.npz')
 if target.exists():return
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items)
 with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as handle:bias=pickle.load(handle)
 residual=y-bias.training_prediction;bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix((residual,(u,i)),shape=(U,I)).tocsr();T=coo_matrix((t,(u,i)),shape=(U,I)).tocsr();C=R.tocsc();XT=X.T.tocsr();RT=R.T.tocsr();pop=np.asarray(X.sum(0)).ravel();norm=np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel())
 settings=[('rating',5),('rating',25),('rating',100),('presence',25),('presence',100)];result={f'{prefix}learned_{kind}{reg}_{tau}':bq.copy() for kind,reg in settings for tau in [0,365]};result[prefix+'learned_support']=np.zeros(len(q));groups={}
 for z,j in enumerate(qi):
  if j>=0 and qu[z]>=0:groups.setdefault(j,[]).append(z)
 keys=np.array(list(groups));rows=[[] for _ in settings];cols=[[] for _ in settings];vals=[[] for _ in settings]
 for offset in range(0,len(keys),64):
  batch=keys[offset:offset+64];co=(XT[batch]@X).toarray();dot=(RT[batch]@R).toarray()
  for z,j in enumerate(batch):
   co[z,j]=0;dot[z,j]=0;sim=co[z]/np.sqrt(np.maximum(pop[j]*pop,1));corr=np.abs(dot[z])/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+5)
   take=min(128 if wide else 32,len(sim));nb1=np.argpartition(sim,-take)[-take:];nb2=np.argpartition(corr,-take)[-take:];nb=np.unique(np.r_[nb1[sim[nb1]>0],nb2[corr[nb2]>0]]);nb=nb[nb!=j]
   if not len(nb):continue
   lo,hi=C.indptr[j:j+2];us=C.indices[lo:hi];yy=C.data[lo:hi];queries=np.array(groups[j]);rr=R[qu[queries]][:,nb].toarray();xx=X[qu[queries]][:,nb].toarray();times=T[qu[queries]][:,nb].toarray();tw=np.exp(-np.abs(times-q.timestamp.to_numpy(float)[queries,None])/86400/365)*((times>=0)&(times<=4102444800));result[prefix+'learned_support'][queries]=np.log1p(xx.sum(1))
   for kind in ['rating','presence']:
    A=(R if kind=='rating' else X)[us][:,nb].toarray();dual=wide and A.shape[0]<A.shape[1];gram=A@A.T if dual else A.T@A;eig,V=np.linalg.eigh(gram);rhs=V.T@yy if dual else V.T@(A.T@yy)
    for k,(label,reg) in enumerate(settings):
     if label!=kind:continue
     w=((A.T@V if dual else V)@(rhs/(np.maximum(eig,0)+reg))).astype(np.float32);design=rr if kind=='rating' else xx;result[f'{prefix}learned_{kind}{reg}_0'][queries]+=design@w;result[f'{prefix}learned_{kind}{reg}_365'][queries]+=(design*tw)@w;rows[k].extend(nb);cols[k].extend([j]*len(nb));vals[k].extend(w)
  if offset%2048==0:log('Learned rating neighborhoods',offset,'/',len(keys))
 for k,(kind,reg) in enumerate(settings):
  W=coo_matrix((np.array(vals[k],np.float32),(rows[k],cols[k])),shape=(I,I)).tocsr();assert np.count_nonzero(W.diagonal())==0;save_npz(out/f'{prefix}learned_{kind}{reg}.npz',W)
 for key in result:
  if key!=prefix+'learned_support':result[key]=clip(result[key])
 np.savez_compressed(target,**result);log('Learned rating/presence neighborhoods complete',prefix)
def pearson_worker(data,out,full,prior):
 target=out/'pearson_features.npz'
 if target.exists():return
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items)
 with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as h:bias=pickle.load(h)
 bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();Y=coo_matrix((y.astype(np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix(((y-bias.training_prediction).astype(np.float32),(u,i)),shape=(U,I)).tocsr();T=coo_matrix((t,(u,i)),shape=(U,I)).tocsc();RC=R.tocsc();up=np.asarray(X.sum(1)).ravel();XT=X.T.tocsr();assert np.array_equal(Y.indices,R.indices)
 so=out/'pearson.so'
 if not so.exists():subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_pearson_kernel.cpp')),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so.resolve()));fn=lib.pair_pearson;ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,fp,fp,fp,ip];fn.restype=None
 variants=['raw_positive','raw_signed','residual_positive','residual_signed'];result={f'pearson_{name}_{k}_{tau}':bq.copy() for name in variants for k in [20,50] for tau in [0,365]};result.update({f'pearson_{name}_support':np.zeros(len(q)) for name in variants});groups={}
 for z,user in enumerate(qu):
  if user>=0 and qi[z]>=0:groups.setdefault(user,[]).append(z)
 keys=np.array(list(groups));qt=q.timestamp.to_numpy(float)
 for offset in range(0,len(keys),64):
  batch=keys[offset:offset+64];co=(X[batch]@XT).toarray()
  for z,user in enumerate(batch):
   c=co[z];c[user]=0;sim=c/np.sqrt(np.maximum(up[user]*up,1));l1=min(1000,U);l2=min(500,U);n1=np.argpartition(sim,-l1)[-l1:];n2=np.argpartition(c,-l2)[-l2:];nb=np.unique(np.r_[n1[c[n1]>=3],n2[c[n2]>=5]]).astype(np.int32)
   wanted=np.zeros(U,bool)
   for queryrow in groups[user]:
    lo,hi=RC.indptr[qi[queryrow]:qi[queryrow]+2];wanted[RC.indices[lo:hi]]=True
   nb=nb[wanted[nb]]
   if not len(nb):continue
   raw=np.zeros(len(nb),np.float32);res=raw.copy();counts=np.zeros(len(nb),np.int32);fn(int(user),len(nb),nb,Y.indptr,Y.indices,Y.data,R.data,raw,res,counts);shrink=counts/(counts+20.);rw=raw*shrink;rr=res*shrink;weights={}
   for name,v in [('raw_positive',np.maximum(rw,0)),('raw_signed',rw),('residual_positive',np.maximum(rr,0)),('residual_signed',rr)]:
    w=np.zeros(U);w[nb]=v;weights[name]=w
   for row in groups[user]:
    j=qi[row];lo,hi=RC.indptr[j:j+2];other=RC.indices[lo:hi];values=RC.data[lo:hi];dates=T.data[lo:hi];distance=np.abs(dates-qt[row])/86400;time_weight=np.exp(-distance/365)*((dates>=0)&(dates<=4102444800))
    for name,weight in weights.items():
     w0=weight[other];active=np.flatnonzero(w0);result[f'pearson_{name}_support'][row]=np.log1p(len(active))
     if not len(active):continue
     vals=values[active];original=w0[active];tw=time_weight[active]
     for tau in [0,365]:
      w=original if tau==0 else original*tw;ranking=np.argsort(-np.abs(w),kind='stable')[:50]
      for k in [20,50]:
       chosen=ranking[:k];result[f'pearson_{name}_{k}_{tau}'][row]+=(w[chosen]@vals[chosen])/(np.abs(w[chosen]).sum()+.5)
  if offset%2048==0:log('Exact common-book Pearson users',offset,'/',len(keys))
 for name in result:
  if not name.endswith('_support'):result[name]=clip(result[name])
 np.savez_compressed(target,**result);log('Exact common-book Pearson complete')
def current_features(data,legacy,prior,full):
 if full:base=np.load(prior/'full/baseline_features.npz')['features']
 else:base,y,n=baseline_validation(data,legacy)
 zs=[np.load(prior/('full' if full else 'validation')/(k+'_features.npz')) for k in ['library','temporal']];F=np.c_[base,np.column_stack([z[k] for z in zs for k in z.files])];return F
if __name__ == '__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','output-dir','prior-dir','baseline-dir']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--worker',choices=['joint','raw','learned','learnedwide','pearson'],required=True);ap.add_argument('--full',action='store_true');a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.worker=='pearson':pearson_worker(a.data_dir,a.output_dir,a.full,a.prior_dir)
 elif a.worker in ['learned','learnedwide']:learned_worker(a.data_dir,a.output_dir,a.full,a.prior_dir,wide=a.worker=='learnedwide')
 else:{'joint':joint_worker,'raw':raw_worker}[a.worker](a.data_dir,a.output_dir,a.full)
''',
    'task3_round3.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from task3_optimize import dataset, clip
from task3_round2 import current_features
KINDS=['joint','raw','learned','learnedwide','pearson']
def log(*a): print(*a,flush=True)
def user_ridge(data, prior, out, full=False):
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(full=full, **{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']})
    guard=out/'manifest.json'
    if guard.exists(): assert json.loads(guard.read_text())==manifest
    guard.write_text(json.dumps(manifest,indent=2))
    if (out/'user_features.npz').exists(): return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full)
    with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as h: bias=pickle.load(h)
    r=y-bias.training_prediction; bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float))
    shape=(len(users),len(items));X=coo_matrix((np.ones(len(y),np.float32),(u,i)),shape=shape).tocsr(); R=coo_matrix((r,(u,i)),shape=shape).tocsr(); T=coo_matrix((t,(u,i)),shape=shape).tocsr()
    XT=X.T.tocsr(); RT=R.T.tocsr(); TC=T.T.tocsr(); pop=np.asarray(X.sum(1)).ravel(); norm=np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel())
    settings=[('rating',5),('rating',25),('rating',100),('presence',25),('presence',100)]
    result={f'user_ridge_{k}{reg}_{tau}':bq.copy() for k,reg in settings for tau in [0,365]};result['user_ridge_support']=np.zeros(len(q));groups={}
    for z,j in enumerate(qu):
        if j>=0 and qi[z]>=0: groups.setdefault(j,[]).append(z)
    keys=np.array(list(groups));rows=[[] for _ in settings];cols=[[] for _ in settings];vals=[[] for _ in settings];qt=q.timestamp.to_numpy(float);start=time.time()
    for offset in range(0,len(keys),32):
        batch=keys[offset:offset+32];co=(X[batch]@XT).toarray();dot=(R[batch]@RT).toarray()
        for z,j in enumerate(batch):
            co[z,j]=0;dot[z,j]=0;sim=co[z]/np.sqrt(np.maximum(pop[j]*pop,1));corr=np.abs(dot[z])/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+5)
            take=min(32,len(sim));nb1=np.argpartition(sim,-take)[-take:];nb2=np.argpartition(corr,-take)[-take:];nb=np.unique(np.r_[nb1[sim[nb1]>0],nb2[corr[nb2]>0]]);nb=nb[nb!=j]
            if not len(nb):continue
            lo,hi=R.indptr[j:j+2];books=R.indices[lo:hi];yy=R.data[lo:hi];queries=np.array(groups[j]);rr=RT[qi[queries]][:,nb].toarray();xx=XT[qi[queries]][:,nb].toarray();times=TC[qi[queries]][:,nb].toarray();tw=np.exp(-np.abs(times-qt[queries,None])/86400/365)*((times>=0)&(times<=4102444800));result['user_ridge_support'][queries]=np.log1p(xx.sum(1))
            for kind in ['rating','presence']:
                A=(RT if kind=='rating' else XT)[books][:,nb].toarray();dual=A.shape[0]<A.shape[1];gram=A@A.T if dual else A.T@A;eig,V=np.linalg.eigh(gram);rhs=V.T@yy if dual else V.T@(A.T@yy)
                for k,(label,reg) in enumerate(settings):
                    if label!=kind:continue
                    w=((A.T@V if dual else V)@(rhs/(np.maximum(eig,0)+reg))).astype(np.float32);design=rr if kind=='rating' else xx
                    result[f'user_ridge_{kind}{reg}_0'][queries]+=design@w;result[f'user_ridge_{kind}{reg}_365'][queries]+=(design*tw)@w;rows[k].extend(nb);cols[k].extend([j]*len(nb));vals[k].extend(w)
        if offset%2048==0:log('User ridge',offset,len(keys),'seconds',round(time.time()-start))
    for k,(kind,reg) in enumerate(settings):
        W=coo_matrix((np.array(vals[k],np.float32),(rows[k],cols[k])),shape=(len(users),len(users))).tocsr();assert np.count_nonzero(W.diagonal())==0;save_npz(out/f'user_{kind}{reg}.npz',W)
    temp=out/'user_features.partial.npz';np.savez_compressed(temp,**{k:clip(v) if k!='user_ridge_support' else v for k,v in result.items()});temp.replace(out/'user_features.npz')
def design(data,legacy,prior,round2,out,full=False,include_user=True):
    F=current_features(data,legacy,prior,full);values={}
    for kind in KINDS:
        z=np.load(round2/('full' if full else 'validation')/(kind+'_features.npz'));values.update({k:z[k] for k in z.files})
    if include_user:
        z=np.load(out/('full' if full else 'validation')/'user_features.npz');values.update({k:z[k] for k in z.files})
    names=['prior_'+str(k) for k in range(F.shape[1])]+list(values)
    X=np.c_[F,np.column_stack(list(values.values()))].astype(np.float32)
    anchor=np.mean([values['learned_rating25_365'],values['wide_learned_rating25_365']],axis=0)
    return X,anchor,names
def extended_design(data,legacy,prior,round2,out,full=False):
    X,anchor,names=design(data,legacy,prior,round2,out,full)
    z=np.load(out/('full' if full else 'validation')/'ordinal_features.npz')
    return np.c_[X,np.column_stack([z[k] for k in z.files])].astype(np.float32),anchor,names+list(z.files)
def ordinal_ridge(data,prior,round2,out,full=False):
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(full=full, **{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']})
    guard=out/'manifest.json'
    if guard.exists():assert json.loads(guard.read_text())==manifest
    guard.write_text(json.dumps(manifest,indent=2))
    if (out/'ordinal_features.npz').exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full)
    with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as h:bias=pickle.load(h)
    r=y-bias.training_prediction;bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));shape=(len(users),len(items))
    R=coo_matrix((r,(u,i)),shape=shape).tocsr();Y=coo_matrix((y,(u,i)),shape=shape).tocsr();T=coo_matrix((t,(u,i)),shape=shape).tocsr();C=R.tocsc();W=load_npz(round2/('full' if full else 'validation')/'learned_rating25.npz').tocsc()
    settings=[(kind,reg) for kind in ['joint','ordinal'] for reg in [10,50]];result={f'book_{kind}{reg}_{tau}':bq.copy() for kind,reg in settings for tau in [0,365]};groups={};qt=q.timestamp.to_numpy(float)
    for z,j in enumerate(qi):
        if j>=0 and qu[z]>=0:groups.setdefault(j,[]).append(z)
    rows=[[] for _ in settings];cols=[[] for _ in settings];vals=[[] for _ in settings];start=time.time()
    for z,(j,queries) in enumerate(groups.items()):
        lo,hi=W.indptr[j:j+2];nb=W.indices[lo:hi]
        if not len(nb):continue
        lo,hi=C.indptr[j:j+2];us=C.indices[lo:hi];yy=C.data[lo:hi];queries=np.array(queries);ay=Y[us][:,nb].toarray();ar=R[us][:,nb].toarray();qy=Y[qu[queries]][:,nb].toarray();qr=R[qu[queries]][:,nb].toarray();times=T[qu[queries]][:,nb].toarray();tw=np.exp(-np.abs(times-qt[queries,None])/86400/365)*((times>=0)&(times<=4102444800))
        for kind in ['joint','ordinal']:
            if kind=='joint': A=np.c_[ay>0,ar];B=np.c_[qy>0,qr];BW=B*np.tile(tw,(1,2))
            else:A=np.concatenate([ay==v for v in range(1,6)],axis=1).astype(float);B=np.concatenate([qy==v for v in range(1,6)],axis=1).astype(float);BW=B*np.tile(tw,(1,5))
            dual=A.shape[0]<A.shape[1];gram=A@A.T if dual else A.T@A;eig,V=np.linalg.eigh(gram);rhs=V.T@yy if dual else V.T@(A.T@yy)
            for k,(label,reg) in enumerate(settings):
                if label!=kind:continue
                w=((A.T@V if dual else V)@(rhs/(np.maximum(eig,0)+reg))).astype(np.float32);result[f'book_{kind}{reg}_0'][queries]+=B@w;result[f'book_{kind}{reg}_365'][queries]+=BW@w
                rows[k].extend(np.concatenate([nb+block*len(items) for block in range(2 if kind=='joint' else 5)]));cols[k].extend([j]*len(w));vals[k].extend(w)
        if z%2048==0:log('Ordinal ridge',z,len(groups),'seconds',round(time.time()-start))
    for k,(kind,reg) in enumerate(settings):save_npz(out/f'book_{kind}{reg}.npz',coo_matrix((vals[k],(rows[k],cols[k])),shape=(len(items)*(2 if kind=='joint' else 5),len(items))).tocsr())
    temp=out/'ordinal_features.partial.npz';np.savez_compressed(temp,**{k:clip(v) for k,v in result.items()});temp.replace(out/'ordinal_features.npz')
''',
    'task3_scaled_bias.py': r'''from final_imports import *
class ScaledTemporalBias:

    def __init__(self, base, epochs=20):
        self.base = copy.deepcopy(base)
        self.epochs = epochs
        self.specs = [('user', 20.0), ('drift', 40.0), ('user90', 50.0), ('userday', 100.0)]

    def fit(self, u, i, t, y):
        u = np.asarray(u, dtype=np.int64)
        i = np.asarray(i, dtype=np.int64)
        keys = self.base.keys(u, i, t)
        terms = []
        for original in self.base.terms:
            name = original['name']
            lab, val = keys[name]
            code = original['index'].get_indexer(lab)
            assert (code >= 0).all()
            terms.append(dict(original, code=code, value=val, item=name.startswith('item')))
        self.scales = []
        for name, pen in self.specs:
            original = next((a for a in terms if a['name'] == ('user' if name == 'drift' else name)))
            lab, val = keys[name]
            code = original['index'].get_indexer(lab)
            assert (code >= 0).all()
            self.scales.append({'name': name, 'penalty': pen, 'index': original['index'], 'code': code, 'value': val, 'coef': np.zeros(len(original['index']))})
        item = sum((a['coef'][a['code']] * a['value'] for a in terms if a['item']))
        scale = np.zeros(len(y))
        pred = self.base.mu + sum((a['coef'][a['code']] * a['value'] for a in terms))
        res = y - pred
        self.objectives = []
        for epoch in range(self.epochs):
            for a in terms:
                code, val = (a['code'], a['value'])
                feature = val * (1 + scale) if a['item'] else val
                old = a['coef']
                res += old[code] * feature
                new = np.bincount(code, weights=res * feature, minlength=len(old)) / (np.bincount(code, weights=feature * feature, minlength=len(old)) + a['penalty'])
                a['coef'] = new
                res -= new[code] * feature
                if a['item']:
                    item += (new - old)[code] * val
            for a in self.scales:
                code, val = (a['code'], a['value'])
                feature = val * item
                old = a['coef']
                res += old[code] * feature
                new = np.bincount(code, weights=res * feature, minlength=len(old)) / (np.bincount(code, weights=feature * feature, minlength=len(old)) + a['penalty'])
                a['coef'] = new
                res -= new[code] * feature
                scale += (new - old)[code] * val
            objective = float(res @ res + sum((a['penalty'] * float(a['coef'] @ a['coef']) for a in terms + self.scales)))
            self.objectives.append(objective)
        assert np.all(np.diff(self.objectives) <= 1e-06 * np.maximum(1, np.asarray(self.objectives[:-1])))
        for old, new in zip(self.base.terms, terms):
            old['coef'] = new['coef']
        for a in self.scales:
            del a['code'], a['value']
        self.training_prediction = y - res
        return self

    def parts(self, u, i, t):
        u = np.asarray(u, dtype=np.int64)
        i = np.asarray(i, dtype=np.int64)
        keys = self.base.keys(u, i, t)
        item = np.zeros(len(u))
        add = np.full(len(u), self.base.mu)
        scale = np.zeros(len(u))
        for a in self.base.terms:
            lab, val = keys[a['name']]
            ix = a['index'].get_indexer(lab)
            ok = ix >= 0
            v = np.zeros(len(u))
            v[ok] = a['coef'][ix[ok]] * val[ok]
            add += v
            if a['name'].startswith('item'):
                item += v
        for a in self.scales:
            lab, val = keys[a['name']]
            ix = a['index'].get_indexer(lab)
            ok = ix >= 0
            scale[ok] += a['coef'][ix[ok]] * val[ok]
        return (add + item * scale, 1 + scale, item)

    def save(self, path):
        np.savez_compressed(path, **{'base_' + str(j): a['coef'].astype('f') for j, a in enumerate(self.base.terms)}, **{'scale_' + str(j): a['coef'].astype('f') for j, a in enumerate(self.scales)})

    def restore(self, path):
        z = np.load(path)
        for j, a in enumerate(self.base.terms):
            a['coef'] = z['base_' + str(j)].astype(float)
        self.scales = []
        for j, (name, pen) in enumerate(self.specs):
            index = next((a['index'] for a in self.base.terms if a['name'] == ('user' if name == 'drift' else name)))
            self.scales.append(dict(name=name, penalty=pen, index=index, coef=z['scale_' + str(j)].astype(float)))
''',
    'task3_temporal_model.py': r'''from final_imports import *
class TemporalBias:
    def __init__(self, config, epochs=25):
        self.config=config;self.epochs=epochs
    def keys(self,u,i,t):
        good=(t>=0)&(t<=4102444800)
        days=np.floor(np.clip(t,0,4102444800)/86400).astype(np.int64)
        days=np.where(good,days,-1)
        center=self.mean_time[np.maximum(u,0)]
        delta=(t-center)/(86400*365.25)
        drift=np.where(good,np.clip(np.sign(delta)*abs(delta)**.4,-5,5),0)
        stride=1000000000
        return {
            'user':(u,np.ones(len(u))), 'item':(i,np.ones(len(u))),
            'month':(np.where(good,days//30,-1),np.ones(len(u))),
            'drift':(u,drift),
            'drift_pos':(u,np.maximum(drift,0)),
            'drift_neg':(u,np.minimum(drift,0)),
            'user180':((u+1)*stride+np.where(good,days//180,-1),np.ones(len(u))),
            'user90':((u+1)*stride+np.where(good,days//90,-1),np.ones(len(u))),
            'item180':((i+1)*stride+np.where(good,days//180,-1),np.ones(len(u))),
            'item90':((i+1)*stride+np.where(good,days//90,-1),np.ones(len(u))),
            'userday':((u+1)*stride+days,np.ones(len(u))),
        }
    def fit(self,u,i,t,y,n_users):
        self.mu=float(np.mean(y));good=(t>=0)&(t<=4102444800)
        count=np.bincount(u[good],minlength=n_users);total=np.bincount(u[good],weights=t[good],minlength=n_users)
        self.mean_time=np.divide(total,count,out=np.full(n_users,float(np.mean(t[good]))),where=count>0)
        keys=self.keys(u,i,t);self.terms=[];res=y.astype(float)-self.mu
        for name,penalty in self.config.items():
            labels,value=keys[name];code,index=pd.factorize(labels,sort=False)
            self.terms.append({'name':name,'penalty':penalty,'index':pd.Index(index),'code':code,'value':value,'coef':np.zeros(len(index)),'den':np.bincount(code,weights=value**2)+penalty})
        for epoch in range(self.epochs):
            for term in self.terms:
                code=term['code'];value=term['value'];res+=term['coef'][code]*value
                term['coef']=np.bincount(code,weights=res*value,minlength=len(term['index']))/term['den']
                res-=term['coef'][code]*value
        self.training_prediction=y-res
        for term in self.terms:
            del term['code'],term['value'],term['den']
        return self
    def predict(self,u,i,t):
        keys=self.keys(u,i,t);out=np.full(len(u),self.mu)
        for term in self.terms:
            labels,value=keys[term['name']];idx=term['index'].get_indexer(labels);good=idx>=0
            out[good]+=term['coef'][idx[good]]*value[good]
        return out
''',
    # TASK 4 MODEL SOURCE
    'task4_drift.cpp': r'''#include <cmath>
#include <random>
extern "C" void train(int n,int I,int K,int B,int epochs,const int*u,const int*i,const int*b,const float*t,const float*P,const float*Q,float*D,float*C,int seed){
std::mt19937 rng(seed);std::uniform_int_distribution<int> row(0,n-1),item(0,I-1);
for(int ep=0;ep<epochs;ep++)for(int z=0;z<n;z++){
 int r=row(rng),v=u[r],a=i[r],j=item(rng),bin=b[r];if(a==j)continue;float time=t[r],s=C[a*B+bin]-C[j*B+bin];
 for(int k=0;k<K;k++)s+=P[v*K+k]*(Q[a*K+k]-Q[j*K+k]);
 for(int k=0;k<16;k++)s+=time*D[v*16+k]*(Q[a*K+k]-Q[j*K+k]);
 float g=1/(1+std::exp(fmax(-30,fmin(30,s)))),lr=.015f;
 C[a*B+bin]+=lr*(g-.1f*C[a*B+bin]);C[j*B+bin]+=lr*(-g-.1f*C[j*B+bin]);
 for(int k=0;k<16;k++)D[v*16+k]+=lr*(g*time*(Q[a*K+k]-Q[j*K+k])-.1f*D[v*16+k]);
}}
''',
    'task4_drift.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from tasks24_reader_neighbors import inputs
W=Path(os.environ['ASSIGNMENT_WORK_DIR']);DATA=Path(os.environ['ASSIGNMENT_DATA_DIR'])
OUT=W/'task4_drift';OUT.mkdir(exist_ok=True)
def component(full=False):
 X,T,qu,qi,qt=inputs(4,full);df=pd.read_csv(DATA/'tasks_3_4_train.csv');u,_=pd.factorize(df.user);i,_=pd.factorize(df.item);days=df.timestamp.to_numpy(float)/86400;order=np.random.RandomState(42).permutation(len(u));idx=np.arange(len(u)) if full else order[:int(.9*len(u))];root=W/'task4_optimized'/('full' if full else 'validation');z=np.load(root/'bpr128.npz');P=np.ascontiguousarray(z['user'],dtype='f');Q=np.ascontiguousarray(z['item'],dtype='f');N,I=X.shape;K=P.shape[1];B=16
 valid=(days[idx]>=0)&(days[idx]<4102444800/86400);ix=idx[valid];lo,hi=np.quantile(days[ix],[.01,.99]);width=max((hi-lo)/B,1);bins=np.clip((days[ix]-lo)/width,0,B-1).astype('i');qbins=np.clip((qt-lo)/width,0,B-1).astype('i');counts=np.bincount(u[ix],minlength=N);means=np.bincount(u[ix],weights=days[ix],minlength=N)/np.maximum(counts,1);means[counts==0]=np.median(days[ix]);t=np.clip((days[ix]-means[u[ix]])/365,-3,3).astype('f');tq=np.clip((qt-means[qu])/365,-3,3).astype('f');uu=u[ix].astype('i');ii=i[ix].astype('i');D=np.zeros((N,16),'f');C=np.zeros((I,B),'f');lib=ctypes.CDLL(str(OUT/'drift.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train.argtypes=[ctypes.c_int]*5+[ip]*3+[fp]*5+[ctypes.c_int];result={};old=0
 base=np.einsum('ij,ij->i',P[qu],Q[qi]);qvalid=(qt>=0)&(qt<4102444800/86400)
 for ep in [3,10,25]:
  lib.train(len(ix),I,K,B,ep-old,uu,ii,bins,t,P,Q,D,C,47+old);old=ep;drift=tq*np.einsum('ij,ij->i',D[qu],Q[qi,:16]);bias=C[qi,qbins];drift[~qvalid]=0;bias[~qvalid]=0
  for name,v in [('drift',drift),('calendar',bias),('joint',base+drift+bias)]:result[f'{name}{ep}']=v.astype('f')
  print('FIT',full,ep,flush=True)
 assert all(np.isfinite(v).all() for v in result.values());np.savez_compressed(OUT/('full.npz' if full else 'validation.npz'),**result);return result
''',
    'task4_ensemble.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
def train_models(X, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_hash=hashlib.sha256(X.indptr.tobytes()+X.indices.tobytes()+X.data.tobytes()).hexdigest()
    trials=[('bpr32',BayesianPersonalizedRanking(factors=32,iterations=100,regularization=.01,learning_rate=.01,random_state=42,num_threads=1)),
            ('bpr64',BayesianPersonalizedRanking(factors=64,iterations=200,regularization=.01,learning_rate=.03,random_state=17,num_threads=1)),
            ('als64',AlternatingLeastSquares(factors=64,iterations=20,regularization=.1,alpha=20,random_state=42,num_threads=2)),
            ('als128',AlternatingLeastSquares(factors=128,iterations=20,regularization=.1,alpha=40,random_state=42,num_threads=2))]
    models={}
    for name,model in trials:
        params={k:getattr(model,k,None) for k in ['factors','iterations','regularization','learning_rate','alpha','random_state','verify_negative_samples','num_threads']}
        params['implicit_version']=__import__('implicit').__version__
        signature=hashlib.sha256((matrix_hash+json.dumps(params,sort_keys=True)).encode()).hexdigest()[:24]
        cache=cache_dir/(name+'_'+signature+'.npz')
        if cache.exists():
            with np.load(cache) as saved:
                models[name]=SimpleNamespace(user_factors=saved['user_factors'],item_factors=saved['item_factors'])
            print('Loaded cached factors',name,flush=True)
            continue
        print('Training',name,flush=True);start=time.time();model.fit(X,show_progress=False)
        models[name]=model
        np.savez_compressed(cache,user_factors=model.user_factors,item_factors=model.item_factors)
        print(name,'seconds',round(time.time()-start,1),flush=True)
    return models
def time_period(t,days):
    good=(t>=0)&(t<=4102444800)
    return np.where(good,np.floor(np.clip(t,0,4102444800)/(days*86400)),-1).astype(np.int64)
def train_time_models(u,i,t,shape,cache_dir):
    output=[]
    for days in [180,365]:
        keys=(u+1)*1000000000+time_period(t,days);code,index=pd.factorize(keys)
        B=coo_matrix((np.ones(len(u),dtype=np.float32),(code,i)),shape=(len(index),shape[1])).tocsr();B.data[:]=1
        signature=hashlib.sha256(B.indptr.tobytes()+B.indices.tobytes()+B.data.tobytes()+str(B.shape).encode()+('factors64_iter20_reg0.1_alpha20_seed42_threads2_'+__import__('implicit').__version__).encode()).hexdigest()[:24]
        path=cache_dir/('timeALS'+str(days)+'_'+signature+'.npz')
        if path.exists():
            with np.load(path) as f:m=SimpleNamespace(user_factors=f['user_factors'],item_factors=f['item_factors'])
            print('Loaded temporal ALS',days,flush=True)
        else:
            print('Training temporal ALS',days,'days; user-periods',len(index),flush=True)
            m=AlternatingLeastSquares(factors=64,iterations=20,regularization=.1,alpha=20,random_state=42,num_threads=2)
            m.fit(B,show_progress=False)
            np.savez_compressed(path,user_factors=m.user_factors,item_factors=m.item_factors)
        output.append(('als_user'+str(days),days,pd.Index(index),m))
    return output
def features(X,models,qu,qi,Times,qt,time_models=None):
    pop=np.asarray(X.sum(axis=0)).ravel();U,I=X.shape
    out={"popularity":np.log1p(pop[np.maximum(qi,0)])}
    out['popularity'][qi<0]=0
    valid=(qu>=0)&(qi>=0)
    seen=np.zeros(len(qu))
    seen[valid]=np.asarray(X[qu[valid],qi[valid]]).ravel()>0
    out['already_seen']=seen
    for name,m in models.items():
        score=np.zeros(len(qu));score[valid]=np.einsum('ij,ij->i',m.user_factors[qu[valid]],m.item_factors[qi[valid]])
        out[name]=score
    G=X.T.tocsr();groups={}
    for q,key in enumerate(qi):
        if valid[q]:groups.setdefault(key,[]).append(q)
    keys=np.array(list(groups));top=np.zeros(len(qu));total=np.zeros(len(qu));jacc=np.zeros(len(qu))
    temporal={name:np.zeros(len(qu)) for name in ['cosine_time30','cosine_time180','cosine_time365','cosine_top10_time180','item_nearest_time','item_age']}
    for days in [30,90,365]:
        temporal['item_count'+str(days)]=np.zeros(len(qu));temporal['item_fraction'+str(days)]=np.zeros(len(qu))
    TC=Times.tocsc()
    for item,qs in groups.items():
        lo,hi=TC.indptr[item:item+2];dates=TC.data[lo:hi];dates=np.sort(dates[(dates>=0)&(dates<=4102444800)])
        if not len(dates):continue
        dates_query=qt[qs]
        pos=np.searchsorted(dates,dates_query)
        nearest=np.minimum(abs(dates_query-dates[np.clip(pos,0,len(dates)-1)]),abs(dates_query-dates[np.clip(pos-1,0,len(dates)-1)]))/86400
        temporal['item_nearest_time'][qs]=-np.log1p(nearest)
        diff=(dates_query-dates[0])/86400;temporal['item_age'][qs]=np.sign(diff)*np.log1p(abs(diff))
        for days in [30,90,365]:
            count=np.searchsorted(dates,dates_query+days*86400,side='right')-np.searchsorted(dates,dates_query-days*86400,side='left')
            temporal['item_count'+str(days)][qs]=np.log1p(count)
            temporal['item_fraction'+str(days)][qs]=count/max(pop[item],1)
    if time_models:
        for name,periods,index,m in time_models:
            keys_query=(qu+1)*1000000000+time_period(qt,periods)
            code=index.get_indexer(keys_query);ok=valid&(code>=0)
            score=out['als64'].copy()
            score[ok]=np.einsum('ij,ij->i',m.user_factors[code[ok]],m.item_factors[qi[ok]])
            out[name]=score
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).toarray()
        for row,key in enumerate(batch):
            cosine=co[row]/np.sqrt(np.maximum(pop[key]*pop,1));cosine[key]=0
            jac=co[row]/np.maximum(pop[key]+pop-co[row],1);jac[key]=0
            for q in groups[key]:
                lo,hi=X.indptr[qu[q]:qu[q]+2];hist=X.indices[lo:hi];w=cosine[hist]
                if len(w):
                    top[q]=w.max();total[q]=np.sort(w)[-10:].sum();jacc[q]=jac[hist].max()
                    distance=abs(Times.data[lo:hi]-qt[q])/86400
                    for days in [30,180,365]:
                        tw=w*np.exp(-distance/days)
                        temporal['cosine_time'+str(days)][q]=tw.max()
                        if days==180:temporal['cosine_top10_time180'][q]=np.sort(tw)[-10:].sum()
        if offset%10000<64:print('Similarity items',offset,'/',len(keys),flush=True)
    out.update(cosine_max=top,cosine_top10=total,jaccard_max=jacc)
    out.update(temporal)
    return list(out),np.column_stack(list(out.values()))
''',
    'task4_item2vec.cpp': r'''#include <algorithm>
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
''',
    'task4_item2vec.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task4_two_approaches import queries
from task4_tree_combiners import W,DATA
OUT=W/'task4_item2vec';OUT.mkdir(exist_ok=True)
NAMES=['cos_mean','cos_max','cos_prev','cos_next','cos_prev3','cos_next3','cos_time30','cos_time180','cos_time730','sgns_mean','sgns_time180','sgns_prev']
def library():
 so=OUT/'item2vec.so'
 if not so.exists():subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_suffix('.cpp')),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');lib.train.argtypes=[ctypes.c_int]*4+[ip]*3+[ctypes.c_int]+[fp]*2+[ctypes.c_int]*3;lib.score.argtypes=[ctypes.c_int]*2+[ip]*2+[dp]+[ip]*2+[dp]+[fp]*4;return lib
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');a=ap.parse_args();lib=library();u,i,t,qu,qi,qt,U,I=queries(a.full);assert (qu>=0).all() and (qi>=0).all();valid=(t>=0)&(t<4102444800);order=np.lexsort((np.arange(len(u)),t,u));order=order[valid[order]];uu=u[order].astype('i');ii=i[order].astype('i');tt=t[order].astype('d');counts=np.bincount(ii,minlength=I);keep=counts[ii]>=2;uu,ii,tt=uu[keep],ii[keep],tt[keep];lens=np.bincount(uu,minlength=U);source=np.bincount(ii[lens[uu]>1],minlength=I);active=np.flatnonzero(source>0);prob=source[active]**.75;prob/=prob.sum();rng=np.random.RandomState(733);pool=rng.choice(active,size=262144,p=prob).astype('i');ptr=np.searchsorted(uu,np.arange(U+1)).astype('i');k=32;E=rng.uniform(-.5/k,.5/k,(I,k)).astype('f');O=np.zeros_like(E);features={};old=0;start=time.time();qu=qu.astype('i');qi=qi.astype('i');qt=qt.astype('d')
 for epoch in [5,12]:
  lib.train(len(ii),k,5,3,uu,ii,pool,len(pool),E,O,old,epoch-old,733);old=epoch;assert np.isfinite(E).all() and np.isfinite(O).all();E[source==0]=0;O[source==0]=0;V=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-8);value=np.zeros((len(qu),12),'f');lib.score(len(qu),k,ptr,ii,tt,qu,qi,qt,E,O,V,value);assert np.isfinite(value).all();features.update({f'item2vec_e{epoch}_{name}':value[:,col].copy() for col,name in enumerate(NAMES)});print('ITEM2VEC',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 stage='full' if a.full else 'validation';np.savez_compressed(OUT/(stage+'_features.npz'),**features);np.savez_compressed(OUT/(stage+'_embeddings.npz'),input=E,output=O,source_count=source);report={'stage':stage,'factors':k,'window':5,'negative_samples':3,'negative_distribution':'eligible interaction frequency power .75, 262144-entry sampled table','min_count':2,'epochs':[5,12],'seed':733,'training_interactions_in_sequence':len(ii),'active_books':len(active),'features':list(features),'queries':len(qu),'ratings_used':False,'zero_rating_events_retained':True,'context':'only fitted training rows; query book excluded from pooled history; prior and following KNOWN training books relative to supplied timestamp; invalid dates omitted from ordered sequence','input_hashes':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['tasks_3_4_train.csv','task_4_test.csv']}};(OUT/(stage+'_report.json')).write_text(json.dumps(report,indent=2));print('EXPORTED',stage,flush=True)
if __name__=='__main__':main()
''',
    'task4_optimize.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from task2_optimize import linear_neighbors
def log(*a):print(*a,flush=True)
def binary(u,i,shape):
    X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=shape).tocsr();X.data[:]=1;return X
def fit_factor(name,X,out,kind,params):
    path=out/(name+'.npz');start=time.time()
    if path.exists():
        z=np.load(path);return SimpleNamespace(user_factors=z['user'],item_factors=z['item'])
    cls=BayesianPersonalizedRanking if kind=='bpr' else AlternatingLeastSquares
    m=cls(**params);m.fit(X,show_progress=False)
    np.savez_compressed(path,user=m.user_factors,item=m.item_factors)
    log('Fit',name,'seconds',round(time.time()-start,1));return m
def period_keys(u,t,offset):
    good=(t>=0)&(t<=4102444800)
    p=np.where(good,np.floor((np.clip(t,0,4102444800)/86400-offset)/365),-1000).astype(np.int64)
    return (u+1)*100000+p
def factors_and_scores(X,u,i,t,qu,qi,qt,out):
    periods=[]
    for offset in [0,182]:
        code,index=pd.factorize(period_keys(u,t,offset));B=binary(code,i,(len(index),X.shape[1]));periods.append((offset,code,pd.Index(index),B))
    specs=[('bpr128',X,'bpr',dict(factors=128,iterations=250,regularization=.03,learning_rate=.03,random_state=42,num_threads=1)),
           ('als128new',X,'als',dict(factors=128,iterations=30,regularization=.1,alpha=40,random_state=42,num_threads=1))]
    for offset,code,index,B in periods:
        specs.append(('time_bpr365_'+str(offset),B,'bpr',dict(factors=64,iterations=150,regularization=.03,learning_rate=.03,random_state=42+offset,num_threads=1)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={name:pool.submit(fit_factor,name,A,out,kind,params) for name,A,kind,params in specs}
        models={name:f.result() for name,f in futures.items()}
    pop=np.asarray(X.sum(0)).ravel();up=np.asarray(X.sum(1)).ravel();valid=(pop[qi]>0)&(up[qu]>0);s={}
    for name in ['bpr128','als128new']:
        m=models[name];v=np.einsum('ij,ij->i',m.user_factors[qu],m.item_factors[qi]);v[~valid]=0;s[name]=v
    for offset,code,index,B in periods:
        name='time_bpr365_'+str(offset);m=models[name];keys=period_keys(qu,qt,offset);c=index.get_indexer(keys);ok=(c>=0)&valid
        v=s['bpr128'].copy();v[ok]=np.einsum('ij,ij->i',m.user_factors[c[ok]],m.item_factors[qi[ok]])
        s[name]=v;s[name+'_available']=ok.astype(np.float32)
        counts=np.zeros(len(qu));counts[ok]=np.diff(B.indptr)[c[ok]];s[name+'_history_count']=np.log1p(counts)
    V=models['als128new'].item_factors;V=V/np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-8)
    return s,V
def ease(X,qu,qi,out):
    pop=np.asarray(X.sum(0)).ravel();top=np.argsort(pop)[-min(4000,X.shape[1]):];Z=X[:,top].astype(np.float64);G=(Z.T@Z).toarray()
    mapping=np.full(X.shape[1],-1,np.int32);mapping[top]=np.arange(len(top));cq=mapping[qi];scores={};context=None
    for reg in [100,500,2000]:
        path=out/('ease'+str(reg)+'.npy')
        if path.exists():B=np.load(path)
        else:
            A=G.copy();A.flat[::len(top)+1]+=reg;P=cho_solve(cho_factor(A),np.eye(len(top)));B=-P/np.diag(P)[None,:];np.fill_diagonal(B,0);B=B.astype(np.float32);np.save(path,B);del A,P
        v=np.zeros(len(qu),np.float32);valid=np.flatnonzero(cq>=0)
        for ix in np.array_split(valid,max(1,(len(valid)+999)//1000)):
            if len(ix):v[ix]=np.asarray(Z[qu[ix]].multiply(B[:,cq[ix]].T).sum(1)).ravel()
        scores['ease'+str(reg)]=v
        if reg==500:context=B
    scores['ease_covered']=(cq>=0).astype(np.float32);np.save(out/'ease_vocabulary.npy',top)
    return scores,context,mapping
def kde_library(out):
    path=out/'temporal_kernel.so'
    if not path.exists():subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task4_temporal_kernel.cpp')),'-o',str(path)],check=True)
    lib=ctypes.CDLL(str(path.resolve()));fn=lib.temporal_kde;ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_double,ptr];fn.restype=None
    def evaluate(x,q,h):
        x=np.ascontiguousarray(x,dtype=np.float64);q=np.ascontiguousarray(q,dtype=np.float64);v=np.zeros(len(q));fn(len(x),x,len(q),q,h,v);return v
    return evaluate
def time_features(u,i,t,qu,qi,qt,out):
    kde=kde_library(out);valid=(t>=0)&(t<=4102444800);global_dates=np.sort(t[valid]/86400);q=qt/86400;good=(qt>=0)&(qt<=4102444800)
    result={'query_valid_time':good.astype(float),'query_year':np.clip(q/365.25,0,130),'query_year_sin':np.sin(2*np.pi*q/365.25),'query_year_cos':np.cos(2*np.pi*q/365.25)}
    for name in ['book_date_count','book_time_position','book_time_z','book_last_age','book_prev_gap','book_next_gap','book_has_prev','book_has_next']:
        result[name]=np.zeros(len(q))
    for h in [7,30,90,365]:
        result['book_kde'+str(h)]=np.full(len(q),np.log(1e-10));result['book_kde_lift'+str(h)]=np.zeros(len(q))
    windows=[1,7,14,60,180,730]
    global_counts={h:np.searchsorted(global_dates,q+h,side='right')-np.searchsorted(global_dates,q-h,side='left') for h in windows}
    global_density={h:kde(global_dates,q,h) for h in [7,30,90,365]}
    for h in windows:
        for n in ['book_count','book_fraction','book_lift']:result[n+str(h)]=np.zeros(len(q))
    sort=np.argsort(i[valid],kind='stable');it=i[valid][sort];ts=t[valid][sort]/86400;keys,starts=np.unique(it,return_index=True);ends=np.r_[starts[1:],len(it)];dates={k:np.sort(ts[a:b]) for k,a,b in zip(keys,starts,ends)}
    order=np.argsort(qi,kind='stable');keys,starts=np.unique(qi[order],return_index=True);ends=np.r_[starts[1:],len(order)]
    for j,a,b in zip(keys,starts,ends):
        ix=order[a:b];x=dates.get(j,np.array([]));N=len(x)
        if not N:continue
        y=q[ix];p=np.searchsorted(x,y);hasprev=p>0;hasnext=p<N;prev=y-x[np.maximum(p-1,0)];nxt=x[np.minimum(p,N-1)]-y
        result['book_date_count'][ix]=np.log1p(N);result['book_time_position'][ix]=p/N
        q25,med,q75=np.quantile(x,[.25,.5,.75]);result['book_time_z'][ix]=np.clip((y-med)/(q75-q25+30),-20,20)
        age=y-x[-1];result['book_last_age'][ix]=np.sign(age)*np.log1p(np.abs(age))
        result['book_prev_gap'][ix]=np.log1p(np.where(hasprev,prev,100000));result['book_next_gap'][ix]=np.log1p(np.where(hasnext,nxt,100000));result['book_has_prev'][ix]=hasprev;result['book_has_next'][ix]=hasnext
        for h in windows:
            count=np.searchsorted(x,y+h,side='right')-np.searchsorted(x,y-h,side='left')
            result['book_count'+str(h)][ix]=np.log1p(count);result['book_fraction'+str(h)][ix]=count/N
            result['book_lift'+str(h)][ix]=np.log((count+.5)/(N+1))-np.log((global_counts[h][ix]+.5)/(len(global_dates)+1))
        for h in [7,30,90,365]:
            density=kde(x,y,h);result['book_kde'+str(h)][ix]=np.log(density+1e-10);result['book_kde_lift'+str(h)][ix]=np.log(density+1e-10)-np.log(global_density[h][ix]+1e-10)
    for name,value in result.items():
        if name!='query_valid_time':value[~good]=0
    log('Fine-grained book activity and KDE complete');return result
def history_features(X,Times,qu,qi,qt,W,E,mapping,V,out):
    assert np.array_equal(X.indptr,Times.indptr) and np.array_equal(X.indices,Times.indices)
    score_names=['history_count','history_valid_count','history_prev_gap','history_next_gap','history_has_prev','history_has_next','history_position','als_prev','als_next','slim_prev','slim_next','ease_prev','ease_next']
    for h in [30,180,365]:score_names += ['history_local'+str(h),'slim_time'+str(h),'slim_mean'+str(h),'ease_time'+str(h),'als_time'+str(h),'als_max_time'+str(h)]
    result={n:np.zeros(len(qu),np.float32) for n in score_names};C=W.tocsc();order=np.argsort(qi,kind='stable');keys,starts=np.unique(qi[order],return_index=True);ends=np.r_[starts[1:],len(order)]
    for z,(j,a,b) in enumerate(zip(keys,starts,ends)):
        weight=np.zeros(X.shape[1],np.float32);lo,hi=C.indptr[j:j+2];weight[C.indices[lo:hi]]=C.data[lo:hi];ej=mapping[j]
        for ix in order[a:b]:
            lo,hi=X.indptr[qu[ix]:qu[ix]+2];hist=X.indices[lo:hi];ht=Times.data[lo:hi]/86400;good=(ht>=0)&(ht<=4102444800/86400)
            result['history_count'][ix]=np.log1p(len(hist));hist=hist[good];ht=ht[good];result['history_valid_count'][ix]=np.log1p(len(hist))
            if not len(hist) or qt[ix]<0 or qt[ix]>4102444800:continue
            date=qt[ix]/86400;distance=np.abs(ht-date);sw=weight[hist];vh=V[hist]@V[j];vh[hist==j]=0
            ew=np.zeros(len(hist),np.float32);ehi=mapping[hist];ok=ehi>=0
            if ej>=0:ew[ok]=E[ehi[ok],ej]
            before=np.flatnonzero(ht<date);after=np.flatnonzero(ht>=date);result['history_position'][ix]=len(before)/len(hist)
            for side,ids in [('prev',before),('next',after)]:
                if len(ids):
                    k=ids[np.argmin(distance[ids])];result['history_'+side+'_gap'][ix]=np.log1p(distance[k]);result['history_has_'+side][ix]=1
                    result['als_'+side][ix]=vh[k];result['slim_'+side][ix]=sw[k];result['ease_'+side][ix]=ew[k]
                else:result['history_'+side+'_gap'][ix]=np.log1p(100000)
            for h in [30,180,365]:
                w=np.exp(-distance/h);den=max(w.sum(),1e-8);result['history_local'+str(h)][ix]=np.log1p(w.sum());s=sw@w
                result['slim_time'+str(h)][ix]=s;result['slim_mean'+str(h)][ix]=s/den;result['ease_time'+str(h)][ix]=ew@w;result['als_time'+str(h)][ix]=vh@w/den;result['als_max_time'+str(h)][ix]=np.max(vh*w)
        if z%4096==0:log('Temporal history candidates',z,'/',len(keys))
    return result
def extended_features(X,u,i,t,qu,qi,qt,out):
    path=out/'extended_features.npz'
    if path.exists():
        z=np.load(path);return z.files,np.column_stack([z[n] for n in z.files])
    result,V=factors_and_scores(X,u,i,t,qu,qi,qt,out)
    es,E,mapping=ease(X,qu,qi,out);result.update(es);log('EASE complete')
    linear=linear_neighbors(X,qu,qi,out);result.update({k:v for k,v in linear.items() if k.startswith('slim')})
    result.update(time_features(u,i,t,qu,qi,qt,out))
    Times=coo_matrix((t,(u,i)),shape=X.shape).tocsr();W=load_npz(out/'slim02.npz')
    result.update(history_features(X,Times,qu,qi,qt,W,E,mapping,V,out))
    assert all(np.isfinite(v).all() for v in result.values());np.savez_compressed(path,**result)
    return list(result),np.column_stack(list(result.values()))
''',
    'task4_pair_fm.cpp': r'''#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
static float score(int m,int k,const int* ids,const float*w,const float*V,std::vector<float>&s){
 std::fill(s.begin(),s.end(),0.f);float r=0;
 for(int a=0;a<m;a++){int j=ids[a];if(j<0)continue;r+=w[j];for(int f=0;f<k;f++){float v=V[j*k+f];r-=.5f*v*v;s[f]+=v;}}
 for(int f=0;f<k;f++)r+=.5f*s[f]*s[f];return r;
}
extern "C" void predict(int n,int m,int k,const int*ids,const float*w,const float*V,float*out){std::vector<float>s(k);for(int r=0;r<n;r++)out[r]=score(m,k,ids+r*m,w,V,s);}
extern "C" void train(int n,int m,int k,int I,int item_offset,const int*ids,const int*pool,float*w,float*V,int start,int epochs,int seed){
 std::vector<float>s(k);std::vector<int>x(m);std::vector<int>order(n);
 for(int ep=start;ep<start+epochs;ep++){
  std::mt19937 rng(seed+ep);std::uniform_int_distribution<int>pop(0,n-1),uniform(0,I-1);for(int r=0;r<n;r++)order[r]=r;std::shuffle(order.begin(),order.end(),rng);
  float lr=.02f*std::pow(.97f,ep);
  for(int row:order){for(int label=1;label>=0;label--){std::copy(ids+row*m,ids+(row+1)*m,x.begin());
   if(!label){int j;do{j=(rng()%2)?uniform(rng):pool[pop(rng)];}while(item_offset+j==x[1]);x[1]=item_offset+j;}
   float pred=score(m,k,x.data(),w,V,s);float g=label-1.f/(1.f+std::exp(-std::max(-30.f,std::min(30.f,pred))));
   for(int a=0;a<m;a++){int j=x[a];if(j<0)continue;w[j]+=lr*(g-.02f*w[j]);for(int f=0;f<k;f++){float old=V[j*k+f];V[j*k+f]+=lr*(g*(s[f]-old)-.04f*old);}}
  }}
 }
}
''',
    'task4_temporal_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <vector>
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
''',
    'task4_tree_combiners.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
xgb = None  # XGBoost was experimental; the accepted final model uses LightGBM.
W=Path('/workspace/work');ROOT=W/'task4_two_approaches';OUT=W/'task4_tree_combiners';OUT.mkdir(exist_ok=True);DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';NSEL=71462
def matrices(full=False):
 z=np.load(ROOT/('fresh_full' if full else 'fresh_validation')/'scores.npz')['scores'];f=np.load(ROOT/('fm_full.npz' if full else 'fm_validation.npz'));F=np.c_[z,np.column_stack([f[k] for k in f.files])].astype('f');n=len(F)//2;d=F[:n]-F[n:];s=F[:n]+F[n:];assert np.isfinite(d).all() and np.isfinite(s).all();return d,s
def predict(model,kind,d,s):
 return model.predict_proba(np.c_[d,s])[:,1]-model.predict_proba(np.c_[-d,s])[:,1]
def fit(kind,params,count,xx,yy):
 model=lgb.LGBMClassifier(n_estimators=count,learning_rate=.035,reg_lambda=20,subsample=.9,subsample_freq=1,colsample_bytree=.9,n_jobs=2,random_state=619,verbosity=-1,objective='binary',**params)
 model.fit(xx,yy)
 return model
''',
    'task4_two_approaches.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
W=Path(os.environ['ASSIGNMENT_WORK_DIR']);DATA=Path(os.environ['ASSIGNMENT_DATA_DIR'])
OUT=W/'task4_two_approaches';OUT.mkdir(exist_ok=True)
NSEL=71462
def split():
 df=pd.read_csv(DATA/'tasks_3_4_train.csv');u,users=pd.factorize(df.user);i,items=pd.factorize(df.item);t=df.timestamp.to_numpy(float);order=np.random.RandomState(42).permutation(len(df));a=int(.9*len(df));idx=order[:a];hold=order[a:];rng=np.random.RandomState(43);neg=[]
 for pos in i[hold]:
  j=i[idx[rng.randint(len(idx))]]
  while j==pos:j=i[idx[rng.randint(len(idx))]]
  neg.append(j)
 return df,u,i,t,users,items,idx,hold,np.array(neg)
def queries(full):
 df,u,i,t,users,items,idx,hold,neg=split()
 if not full:return u[idx],i[idx],t[idx],np.tile(u[hold],2),np.r_[i[hold],neg],np.tile(t[hold],2),len(users),len(items)
 q=pd.read_csv(DATA/'task_4_test.csv');qu=pd.Index(users).get_indexer(q.user);qi=np.r_[pd.Index(items).get_indexer(q.item1),pd.Index(items).get_indexer(q.item2)];return u,i,t,np.tile(qu,2),qi,np.tile(q.timestamp.to_numpy(float),2),len(users),len(items)
def fm(full=False):
 path=OUT/('fm_full.npz' if full else 'fm_validation.npz')
 if path.exists():return dict(np.load(path))
 u,i,t,qu,qi,qt,U,I=queries(full);days=t/86400;good=(days>=0)&(days<4102444800/86400);lo,hi=np.quantile(days[good],[.01,.99]);width=max((hi-lo)/24,1);b=np.clip((days-lo)/width,0,23).astype('i');qb=np.clip((qt/86400-lo)/width,0,23).astype('i');b[~good]=24;qb[(qt<0)|(qt>=4102444800)]=24
 period=b//4;keys=u.astype('int64')*10+period;code,index=pd.factorize(keys);qcode=pd.Index(index).get_indexer(qu.astype('int64')*10+qb//4);off=U+I+25;ids=np.ascontiguousarray(np.c_[u,U+i,U+I+b,off+code],dtype='i');qids=np.ascontiguousarray(np.c_[qu,U+qi,U+I+qb,np.where(qcode>=0,off+qcode,-1)],dtype='i');nfeatures=off+len(index);k=16;rng=np.random.RandomState(419);V=rng.normal(0,.03,(nfeatures,k)).astype('f');w=np.zeros(nfeatures,'f');pool=i.astype('i');lib=ctypes.CDLL(str(OUT/'pair_fm.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train.argtypes=[ctypes.c_int]*5+[ip]*2+[fp]*2+[ctypes.c_int]*3;lib.predict.argtypes=[ctypes.c_int]*3+[ip]+[fp]*3
 result={};old=0
 for epoch in [5,15,30]:
  lib.train(len(ids),4,k,I,U,ids,pool,w,V,old,epoch-old,419);old=epoch;value=np.empty(len(qids),'f');lib.predict(len(qids),4,k,qids,w,V,value);assert np.isfinite(value).all();result['fm_e'+str(epoch)]=value
  print('FM',full,epoch,flush=True)
 np.savez_compressed(path,**result);return result
def fresh_stage(full):
 dest=OUT/('fresh_full' if full else 'fresh_validation');dest.mkdir(exist_ok=True);final=dest/'scores.npz'
 if final.exists():return np.load(final)['scores'],json.loads((dest/'names.json').read_text())
 import task4_ensemble as base,task4_optimize as ext,task4_drift as drift
 u,i,t,qu,qi,qt,U,I=queries(full);X=ext.binary(u,i,(U,I));Times=coo_matrix((t,(u,i)),shape=X.shape).tocsr()
 oldsavez=np.savez_compressed;oldsave=np.save
 def save_npz(path,*args,**kw):
  name=Path(path).name
  if ('user_factors' in kw or 'user' in kw) and name!='bpr128.npz':return
  return oldsavez(path,*args,**kw)
 def save_npy(path,*args,**kw):
  if Path(path).name.startswith('ease') and not 'vocabulary' in Path(path).name:return
  return oldsave(path,*args,**kw)
 np.savez_compressed=save_npz;np.save=save_npy
 try:
  models=base.train_models(X,dest/'factors');tm=base.train_time_models(u,i,t,X.shape,dest/'factors');bn,bs=base.features(X,models,qu,qi,Times,qt,tm);del models,tm;gc.collect();en,es=ext.extended_features(X,u,i,t,qu,qi,qt,dest)
 finally:np.savez_compressed=oldsavez;np.save=oldsave
 src=Path(drift.__file__).read_text();src=src[src.index('def component('):];src=src.replace("W/'task4_optimized'/('full' if full else 'validation')",'dest');ns=dict(drift.__dict__);ns.update(OUT=OUT,dest=dest);exec(src,ns);dz=ns['component'](full);ds=np.column_stack(list(dz.values()));scores=np.c_[bs,es,ds].astype('f');names=bn+en+['drift_'+n for n in dz];assert len(names)==scores.shape[1];
 for scratch in ['extended_features.npz','linear_scores.npz']:
  p=dest/scratch
  if p.exists():p.unlink()
 np.savez_compressed(final,scores=scores);(dest/'names.json').write_text(json.dumps(names));
 for p in dest.rglob('bpr128.npz'):p.unlink()
 print('ALL_COMPONENTS_REFIT',full,len(names),flush=True);return scores,names
''',
    # TASK 5 MODEL SOURCE
    'task5_attention.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve import data,graph
def scores(model,contexts,qu,qi,personal,query):
 E,V,bias,pos,pop=[model[k] for k in ['E','V','bias','position','pop']];N=len(contexts);result={k:np.zeros(len(qi),'f') for k in ['attention','joint','last','recent3','max10']};k=E.shape[1]
 for side in [0,1]:
  for row,h in enumerate(contexts):
   index=side*N+row;j=qi[index]
   if j<0 or pop[j]==0:continue
   hh=h[-10:][::-1];values=E[hh]@V[j];logits=values/np.sqrt(k)+pos[:len(hh)];weight=np.exp(logits-logits.max());weight/=weight.sum();value=weight@values;result['attention'][index]=value;result['joint'][index]=value+bias[j]+personal[qu[index]]@query[j];result['last'][index]=values[0];result['recent3'][index]=values[:3].mean();result['max10'][index]=values.max()
 return result
def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True);target=part/'attention.npz'
 if target.exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(list(map(len,h)))],np.int32);seq=np.concatenate(h).astype('i');sp=X.indptr.astype('i');seen=X.indices.astype('i');warm=dict(np.load(a.previous/('full' if a.full else 'validation')/'fpmc32_uniform_model.npz'));P=warm['P'];Q=warm['Q'];E=warm['A'];V=warm['B'];bias=warm['bias'];pop=np.asarray(X.sum(0)).ravel();E[(pop==0)|(warm['source_count']==0)]=0;V[pop==0]=0;active=np.flatnonzero(pop>0).astype('i');position=(-np.log1p(np.arange(10))).astype('f');k=E.shape[1];so=part/'attention.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_attention_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train_attention.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int]+[fp]*6+[ctypes.c_int]*3+[ctypes.c_float];old=0;result={};start=time.time()
 for epoch in [0,3,10,20]:
  if epoch:lib.train_attention(U,k,ptr,seq,sp,seen,active,len(active),P,Q,E,V,bias,position,old,epoch-old,53,.005)
  old=epoch;assert all(np.isfinite(x).all() for x in [E,V,bias,position]);model=dict(E=E,V=V,bias=bias,position=position,pop=pop);values=scores(model,contexts,qu,qi,P,Q);result.update({f'attention_e{epoch}_{name}':v for name,v in values.items()});print('ATTENTION',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(part/'attention_model.npz',E=E,V=V,bias=bias,position=position,pop=pop);temp=part/'attention.tmp.npz';np.savez_compressed(temp,**result);temp.replace(target)
 (part/'attention_manifest.json').write_text(json.dumps({'full':a.full,'history_limit':10,'seed':53,'checkpoints':[0,3,10,20],'learning_rate':.005,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']}},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_attention_kernel.cpp': r'''#include <algorithm>
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
extern "C" void train_attention(int U,int k,const int* ptr,const int* seq,const int* sp,const int* seen,const int* active,int na,const float* P,const float* Q,float* E,float* V,float* bias,float* position,int start,int epochs,int seed,float lr){
 int total=ptr[U];std::mt19937 rng(seed);Work work(k);
 for(int epoch=start;epoch<start+epochs;epoch++){rng.seed(seed+epoch);float rate=lr*std::pow(.98f,epoch);
  for(int z=0;z<total-U;z++){int u=rng()%U,lo=ptr[u],hi=ptr[u+1];if(hi-lo<2)continue;int available=hi-lo-1;int span=(rng()%2)?std::min(available,10):available;int pos=hi-span+rng()%span,i=seq[pos],j,tries=0;
   do{j=(rng()%2)?seq[rng()%total]:active[rng()%na];tries++;}while((j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))&&tries<100);if(j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))continue;
   update_attention(u,lo,pos,i,j,k,seq,P,Q,E,V,bias,position,rate,work);
  }
 }
}
''',
    'task5_bpr_round3.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve import data,graph
def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True)
 if (part/'bpr.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);pop=np.asarray(X.sum(0)).ravel();known=qi>=0;known[known]&=pop[qi[known]]>0;N=len(contexts);ctx=np.tile(np.arange(N),2);last=np.array([c[-1] for c in contexts]);profiles={}
 for cap in [3,20]:
  short=[c[-cap:] for c in contexts];lens=np.array(list(map(len,short)));profiles[cap]=coo_matrix((np.repeat(1/np.maximum(lens,1),lens),(np.repeat(np.arange(N),lens),np.concatenate(short))),shape=(N,I)).tocsr()
 result={};samples={};start=time.time();configs=[('bpr128',128,.03,.03,350,42,[50,150,350]),('bpr96',96,.003,.05,300,23,[50,150,300])]
 for name,k,reg,lr,iterations,seed,checkpoints in configs:
  model=BayesianPersonalizedRanking(factors=k,iterations=iterations,regularization=reg,learning_rate=lr,random_state=seed,num_threads=1)
  def callback(epoch,*extra):
   epoch+=1
   if epoch not in checkpoints:return
   Uv=model.user_factors;Iv=model.item_factors;pred=np.zeros(len(qi),'f');pred[known]=np.einsum('ij,ij->i',Uv[qu[known]],Iv[qi[known]]);result[f'{name}_e{epoch}_personal']=pred;V=Iv[:,:-1].copy();V[pop==0]=0;norm=np.linalg.norm(V,axis=1);valid=known&(pop[last[ctx]]>0);cos=np.zeros(len(qi),'f');cos[valid]=np.einsum('ij,ij->i',V[last[ctx[valid]]],V[qi[valid]])/np.maximum(norm[last[ctx[valid]]]*norm[qi[valid]],1e-9);result[f'{name}_e{epoch}_lastcos']=cos
   for cap,C in profiles.items():
    mean=C@V;length=np.linalg.norm(mean,axis=1);cos=np.zeros(len(qi),'f');cos[known]=np.einsum('ij,ij->i',mean[ctx[known]],V[qi[known]])/np.maximum(length[ctx[known]]*norm[qi[known]],1e-9);result[f'{name}_e{epoch}_recent{cap}cos']=cos
   if epoch==iterations:
    ix=np.random.RandomState(seed).choice(np.flatnonzero(known),min(512,known.sum()),replace=False);samples[name+'_rows']=ix;samples[name+'_users']=Uv[qu[ix]].copy();samples[name+'_items']=Iv[qi[ix]].copy();samples[name+'_scores']=pred[ix]
   print('GLOBAL_BPR',a.full,name,epoch,'seconds',round(time.time()-start),flush=True)
  model.fit(X,show_progress=False,callback=callback)
  assert name+'_e'+str(iterations)+'_personal' in result
 temp=part/'bpr.tmp.npz';np.savez_compressed(temp,**result);temp.replace(part/'bpr.npz');np.savez_compressed(part/'bpr_replay_sample.npz',**samples);(part/'bpr_config.json').write_text(json.dumps({'full':a.full,'configs':configs,'library_version':__import__('implicit').__version__,'factor_states':'replay sample retained; full models regenerate from deterministic fits'},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_ease_next.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve import data,graph
def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(exist_ok=True,parents=True)
 if (out/'ease.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);pop=np.asarray(X.sum(0)).ravel();top=np.argsort(pop)[-4000:];mapping=np.full(I,-1);mapping[top]=np.arange(len(top));Z=X[:,top].astype(float);G=(Z.T@Z).toarray();ct=[];N=len(contexts)
 for mode in ['whole','last','last3','decay']:
  rr=[];cc=[];vv=[]
  for z,hist in enumerate(contexts):
   hist=hist[-1:] if mode=='last' else (hist[-3:] if mode=='last3' else hist);ids=mapping[hist];weight=.9**np.arange(len(hist))[::-1] if mode=='decay' else np.ones(len(hist));ok=ids>=0;rr.extend([z]*int(ok.sum()));cc.extend(ids[ok]);vv.extend(weight[ok])
  ct.append(coo_matrix((vv,(rr,cc)),shape=(N,len(top))).tocsr())
 ctx=np.tile(np.arange(N),2);target=np.full(len(qi),-1);valid=qi>=0;target[valid]=mapping[qi[valid]];valid=np.flatnonzero(target>=0);result={};start=time.time()
 for reg in [50,300,1000]:
  A=G.copy();A.flat[::len(top)+1]+=reg;P=cho_solve(cho_factor(A),np.eye(len(top)));B=(-P/np.diag(P)[None,:]).astype('f');np.fill_diagonal(B,0)
  for mode,C in zip(['whole','last','last3','decay'],ct):
   pred=np.zeros(len(qi),'f')
   for ix in np.array_split(valid,max(1,(len(valid)+511)//512)):pred[ix]=np.asarray(C[ctx[ix]].multiply(B[:,target[ix]].T).sum(1)).ravel()
   result[f'ease{reg}_{mode}']=pred
  print('EASE',a.full,reg,'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(out/'ease.npz',**result,covered=(target>=0).astype('f'))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_ensemble.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
def cached_fit(name,model,X,folder):
    folder.mkdir(parents=True,exist_ok=True)
    params={k:getattr(model,k,None) for k in ['factors','iterations','regularization','learning_rate','alpha','random_state','num_threads','verify_negative_samples']}
    params['version']=__import__('implicit').__version__
    key=hashlib.sha256(X.indptr.tobytes()+X.indices.tobytes()+X.data.tobytes()+str(X.shape).encode()+json.dumps(params,sort_keys=True).encode()).hexdigest()[:24]
    path=folder/(name+'_'+key+'.npz')
    if path.exists():
        with np.load(path) as z:result=SimpleNamespace(user_factors=z['user_factors'],item_factors=z['item_factors'])
        print('Loaded',name,flush=True);return result
    print('Fitting',name,flush=True);start=time.time();model.fit(X,show_progress=False)
    np.savez_compressed(path,user_factors=model.user_factors,item_factors=model.item_factors)
    print(name,'seconds',round(time.time()-start,1),flush=True);return model
def prepare(history,I,folder):
    U=len(history);u=np.repeat(np.arange(U),[len(h) for h in history]);i=np.concatenate(history)
    X=coo_matrix((np.ones(len(i),dtype=np.float32),(u,i)),shape=(U,I)).tocsr();X.data[:]=1
    prev=np.concatenate([h[:-1] for h in history if len(h)>1]);nxt=np.concatenate([h[1:] for h in history if len(h)>1])
    T=coo_matrix((np.ones(len(prev),dtype=np.float32),(prev,nxt)),shape=(I,I)).tocsr()
    specs=[('bpr32',BayesianPersonalizedRanking(factors=32,iterations=100,learning_rate=.01,regularization=.01,random_state=42,num_threads=1)),('bpr64',BayesianPersonalizedRanking(factors=64,iterations=150,learning_rate=.03,regularization=.01,random_state=17,num_threads=1)),('als64',AlternatingLeastSquares(factors=64,iterations=20,alpha=20,regularization=.1,random_state=42,num_threads=2))]
    models={name:cached_fit(name,m,X,folder) for name,m in specs}
    models['transition_als32']=cached_fit('transition_als32',AlternatingLeastSquares(factors=32,iterations=20,alpha=10,regularization=.1,random_state=42,num_threads=2),T,folder)
    return X,T,models
def features(X,T,models,contexts,qu,qi):
    I=X.shape[1];N=len(contexts);assert len(qu)==2*N and len(qi)==2*N
    ctx=np.r_[np.arange(N),np.arange(N)];length=np.array([len(h) for h in contexts]);pop=np.asarray(X.sum(axis=0)).ravel();valid=(qi>=0)&(qu>=0);known=valid.copy();known[valid]&=pop[qi[valid]]>0
    row=np.repeat(np.arange(N),length);items=np.concatenate(contexts)
    C=coo_matrix((np.ones(len(items),dtype=np.float32),(row,items)),shape=(N,I)).tocsr();C.data[:]=1
    seen=np.zeros(len(qi));seen[valid]=np.asarray(C[ctx[valid],qi[valid]]).ravel()>0
    popularity=np.zeros(len(qi));popularity[valid]=np.log1p(pop[qi[valid]])
    out={'popularity':popularity,'already_seen':seen,'cold_item':(~known).astype(float),'history_count':np.log1p(length[ctx])}
    for name in ['bpr32','bpr64']:
        m=models[name];score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',m.user_factors[qu[known]],m.item_factors[qi[known]]);out[name]=score
    base=models['als64'];als=AlternatingLeastSquares(factors=64,iterations=20,alpha=20,regularization=.1,random_state=42,num_threads=2);als.item_factors=base.item_factors
    uf=als.recalculate_user(np.arange(N),C)
    score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',uf[ctx[known]],base.item_factors[qi[known]]);out['als_history']=score
    recent=[h[-5:] for h in contexts];rr=np.repeat(np.arange(N),[len(h) for h in recent]);ri=np.concatenate(recent);CR=coo_matrix((np.ones(len(ri),dtype=np.float32),(rr,ri)),shape=C.shape).tocsr()
    uf=als.recalculate_user(np.arange(N),CR);score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',uf[ctx[known]],base.item_factors[qi[known]]);out['als_recent5']=score
    last=np.array([h[-1] for h in contexts]);prev=last[ctx];trans=models['transition_als32'];score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',trans.user_factors[prev[known]],trans.item_factors[qi[known]]);out['factorized_markov_last']=score
    meanprev=np.zeros((N,trans.user_factors.shape[1]))
    for n,h in enumerate(contexts):
        past=h[-3:][::-1];w=.7**np.arange(len(past));meanprev[n]=w@trans.user_factors[past]/w.sum()
    score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',meanprev[ctx[known]],trans.item_factors[qi[known]]);out['factorized_markov_recent3']=score
    fwd=np.zeros(len(qi));rev=np.zeros(len(qi));fwd[valid]=np.asarray(T[prev[valid],qi[valid]]).ravel();rev[valid]=np.asarray(T[qi[valid],prev[valid]]).ravel()
    out['markov_count']=np.log1p(fwd);out['markov_reverse']=np.log1p(rev)
    lift=np.zeros(len(qi));lift[valid]=np.log1p(fwd[valid]/(5*pop[qi[valid]]/max(pop.sum(),1)+1e-9));out['markov_lift']=lift
    for k,counts in models['tail_popularity'].items():
        score=np.zeros(len(qi));fraction=np.zeros(len(qi));score[valid]=np.log1p(counts[qi[valid]]);fraction[valid]=counts[qi[valid]]/np.maximum(pop[qi[valid]],1)
        out['tail_pop'+str(k)]=score;out['tail_fraction'+str(k)]=fraction
    out.update(cosine_max=np.zeros(len(qi)),cosine_top10=np.zeros(len(qi)),cosine_last=np.zeros(len(qi)),cosine_recent3=np.zeros(len(qi)),cosine_recent5=np.zeros(len(qi)))
    groups={}
    for q,key in enumerate(qi):
        if known[q]:groups.setdefault(key,[]).append(q)
    keys=np.array(list(groups));G=X.T.tocsr()
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).tocsr();co.sort_indices()
        for j,key in enumerate(batch):
            lo,hi=co.indptr[j:j+2];neighbors=co.indices[lo:hi];values=co.data[lo:hi]/np.sqrt(np.maximum(pop[key]*pop[neighbors],1))
            for q in groups[key]:
                h=contexts[ctx[q]];pos=np.searchsorted(neighbors,h);ok=pos<len(neighbors);ok[ok]&=neighbors[pos[ok]]==h[ok];w=np.zeros(len(h));w[ok]=values[pos[ok]];w[h==key]=0
                out['cosine_max'][q]=w.max();out['cosine_top10'][q]=np.sort(w)[-10:].sum();out['cosine_last'][q]=w[-1]
                for k in [3,5]:
                    rw=w[-k:][::-1];decay=.7**np.arange(len(rw));out['cosine_recent'+str(k)][q]=rw@decay/decay.sum()
        if offset%20000<64:print('Similarity candidate items',offset,'/',len(keys),flush=True)
    return list(out),np.column_stack(list(out.values()))
def tail_pop(history,I):return {k:np.bincount(np.concatenate([h[-k:] for h in history]),minlength=I) for k in [1,5]}
if __name__ == '__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--predict',action='store_true');a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 p=pd.read_csv(a.data_dir/'task_5_train.csv',usecols=['user','item']);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);U,I=len(users),len(items);ptr=np.searchsorted(u,np.arange(U+1));H=[i[ptr[v]:ptr[v+1]] for v in range(U)];h=[v[:-2] for v in H];ctx=h+[v[:-1] for v in H];pos=np.r_[[v[-2] for v in H],[v[-1] for v in H]];pool=np.concatenate(h);rng=np.random.RandomState(43);neg=[]
 for target in pos:
  j=pool[rng.randint(len(pool))]
  while j==target:j=pool[rng.randint(len(pool))]
  neg.append(j)
 qu=np.tile(np.arange(U),4);qi=np.r_[pos,neg];X,T,m=prepare(h,I,a.output_dir/'factors');m['tail_popularity']=tail_pop(h,I);names,f=features(X,T,m,ctx,qu,qi);np.savez_compressed(a.output_dir/'pair_features.npz',scores=f,users=qu,items=qi);(a.output_dir/'results.json').write_text(json.dumps({'feature_names':names}))
 X,T,m=prepare(H,I,a.output_dir/'factors');m['tail_popularity']=tail_pop(H,I);q=pd.read_csv(a.data_dir/'task_5_test.csv');tu=pd.Index(users).get_indexer(q.user);qi=np.r_[pd.Index(items).get_indexer(q.item1),pd.Index(items).get_indexer(q.item2)];qu=np.tile(tu,2);nn,f=features(X,T,m,[H[v] for v in tu],qu,qi);assert nn==names;np.savez_compressed(a.output_dir/'test_pair_features.npz',scores=f,users=qu,items=qi)
''',
    'task5_era.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_order import transitions
from task5_improve import data
def ordering(H,pen):
 incoming=np.asarray(H.sum(0)).ravel();outgoing=np.asarray(H.sum(1)).ravel();degree=incoming+outgoing;A=diags(degree+pen)-H-H.T;M=LinearOperator(A.shape,matvec=lambda x:x/(degree+pen));theta,info=cg(A.astype(float),incoming-outgoing,M=M,rtol=1e-6,maxiter=400);assert info==0;return theta
def era_scores(theta,contexts,qi):
 N=len(contexts);features={k:np.zeros(len(qi),'f') for k in ['level','gap_last','gap_mean3','gap_mean','gap_max','distance_trend5','distance_trend20','percentile']}
 for row,h in enumerate(contexts):
  hist=theta[h];last=hist[-1];mean3=hist[-3:].mean();mean=hist.mean();maximum=hist.max();future=[]
  for cap in [5,20]:
   v=hist[-cap:];x=np.arange(len(v),dtype=float);xc=x-x.mean();slope=(xc@(v-v.mean()))/max(xc@xc,1e-9);future.append(v.mean()+slope*(len(v)-x.mean()))
  for side in [0,1]:
   z=row+side*N;j=qi[z]
   if j<0:continue
   value=theta[j];vals=[value,value-last,value-mean3,value-mean,value-maximum,abs(value-future[0]),abs(value-future[1]),np.mean(hist<value)]
   for key,val in zip(features,vals):features[key][z]=val
 return features
def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True)
 if (part/'era.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);H=None;start=time.time()
 for window,H in transitions(h,I):
  if window==3:break
 result={};params={}
 for pen in [.1,1.,10.]:
  theta=ordering(H,pen);params['theta_'+str(pen)]=theta;result.update({f'era{pen}_{k}':v for k,v in era_scores(theta,contexts,qi).items()});print('COMMUNITY_ORDER',a.full,pen,'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(part/'era_parameters.npz',**params);temp=part/'era.tmp.npz';np.savez_compressed(temp,**result);temp.replace(part/'era.npz')
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_fold.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve import data
def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True)
 if (part/'fold.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);N=len(contexts);w=dict(np.load(a.previous/('full' if a.full else 'validation')/'fpmc64_useruniform_model.npz'));P=w['P'];Q=w['Q'];bias=w['bias'];pop=w['pop'];active=np.flatnonzero(pop>0).astype('i');known=(pop>0).astype('i');lengths=np.array(list(map(len,contexts)));seq=np.concatenate(contexts).astype('i');ptr=np.ascontiguousarray(np.r_[0,np.cumsum(lengths)],np.int32);X=coo_matrix((np.ones(len(seq)),(np.repeat(np.arange(N),lengths),seq)),shape=(N,I)).tocsr();X.sort_indices();sp=X.indptr.astype('i');seen=X.indices.astype('i');users=qu[:N].astype('i');ctx=np.tile(np.arange(N),2);valid=qi>=0;valid[valid]&=pop[qi[valid]]>0;k=P.shape[1];normq=np.linalg.norm(Q,axis=1);so=part/'fold.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_fold_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.fold_users.argtypes=[ctypes.c_int]*2+[ip]*6+[ctypes.c_int,ip]+[fp]*4+[ctypes.c_int]*3;result={};sample={};start=time.time()
 for cap in [3,10,0]:
  factors=P[users].copy();old=0
  for step in [25,100]:
   lib.fold_users(N,k,ptr,seq,sp,seen,users,active,len(active),known,P,Q,bias,factors,cap,old,step-old);old=step;score=np.zeros(len(qi),'f');score[valid]=np.einsum('ij,ij->i',factors[ctx[valid]],Q[qi[valid]]);result[f'fold{cap}_e{step}_score']=score;cos=np.zeros(len(qi),'f');cos[valid]=score[valid]/np.maximum(np.linalg.norm(factors,axis=1)[ctx[valid]]*normq[qi[valid]],1e-9);result[f'fold{cap}_e{step}_cos']=cos
   if step==100:
    ix=np.random.RandomState(91).choice(np.flatnonzero(valid),256,replace=False);sample[f'fold{cap}_rows']=ix;sample[f'fold{cap}_user']=factors[ctx[ix]];sample[f'fold{cap}_item']=Q[qi[ix]];sample[f'fold{cap}_score']=score[ix]
   print('QUERY_FOLD',a.full,cap,step,'seconds',round(time.time()-start),flush=True)
 temp=part/'fold.tmp.npz';np.savez_compressed(temp,**result);temp.replace(part/'fold.npz');np.savez_compressed(part/'fold_replay_sample.npz',**sample)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_fold_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <random>
extern "C" void fold_users(int N,int k,const int* ptr,const int* seq,const int* sp,const int* seen,const int* users,const int* active,int na,const int* known,const float* P,const float* Q,const float* bias,float* result,int cap,int start,int steps){
 for(int row=0;row<N;row++){int u=users[row],lo=ptr[row],hi=ptr[row+1];std::mt19937 rng(97+row+start*31+cap*13);int begin=cap?std::max(lo,hi-cap):lo;
  for(int step=0;step<steps;step++){int i=-1;for(int tries=0;tries<100;tries++){int candidate=seq[begin+rng()%(hi-begin)];if(known[candidate]){i=candidate;break;}}if(i<0)continue;int j,tries=0;do{j=active[rng()%na];tries++;}while(std::binary_search(seen+sp[row],seen+sp[row+1],j)&&tries<100);if(std::binary_search(seen+sp[row],seen+sp[row+1],j))continue;
   float diff=bias[i]-bias[j];for(int f=0;f<k;f++)diff+=result[row*k+f]*(Q[i*k+f]-Q[j*k+f]);float g=1/(1+std::exp(std::clamp(diff,-30.f,30.f)));
   for(int f=0;f<k;f++)result[row*k+f]+=.02f*(g*(Q[i*k+f]-Q[j*k+f])-.1f*(result[row*k+f]-P[u*k+f]));
  }
 }
}
''',
    'task5_fossil_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
extern "C" void fossil_train(int U,int k,const int* ptr,const int* seq,const int* sp,const int* seen,const int* active,int na,int start,int epochs,int seed,float lr,float reg,float* A,float* B,float* wu,float* wg,float* bias){
 int total=ptr[U];std::vector<float> x(k),delta(k),ad(20*k),wd(3);std::vector<int> hist(20);std::vector<float> coef(20);std::mt19937 rng(seed);
 for(int epoch=start;epoch<start+epochs;epoch++){rng.seed(seed+epoch);float rate=lr*std::pow(.98f,epoch);
  for(int z=0;z<total-U;z++){int u=rng()%U,lo=ptr[u],hi=ptr[u+1];if(hi-lo<2)continue;int pos=lo+1+rng()%(hi-lo-1),i=seq[pos],j,tries=0;
   do{j=(rng()%2)?seq[rng()%total]:active[rng()%na];tries++;}while((j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))&&tries<100);if(j==i||std::binary_search(seen+sp[u],seen+sp[u+1],j))continue;
   int n=std::min(20,pos-lo);float norm=1/std::sqrt(float(n));std::fill(x.begin(),x.end(),0);
   for(int h=0;h<n;h++){hist[h]=seq[pos-1-h];coef[h]=norm+(h<3?wg[h]+wu[u*3+h]:0);for(int f=0;f<k;f++)x[f]+=coef[h]*A[hist[h]*k+f];}
   float diff=bias[i]-bias[j];for(int f=0;f<k;f++)diff+=x[f]*(B[i*k+f]-B[j*k+f]);float g=1/(1+std::exp(std::clamp(diff,-30.f,30.f)));
   for(int f=0;f<k;f++)delta[f]=B[i*k+f]-B[j*k+f];
   for(int h=0;h<std::min(3,n);h++){float v=0;for(int f=0;f<k;f++)v+=A[hist[h]*k+f]*delta[f];wd[h]=g*v;}
   for(int h=0;h<n;h++)for(int f=0;f<k;f++)A[hist[h]*k+f]+=rate*(g*coef[h]*delta[f]-reg*A[hist[h]*k+f]);
   for(int f=0;f<k;f++){B[i*k+f]+=rate*(g*x[f]-reg*B[i*k+f]);B[j*k+f]+=rate*(-g*x[f]-reg*B[j*k+f]);}
   for(int h=0;h<std::min(3,n);h++){wu[u*3+h]+=rate*(wd[h]-reg*wu[u*3+h]);wg[h]+=rate*.1f*(wd[h]-reg*wg[h]);}
   bias[i]+=rate*(g-reg*bias[i]);bias[j]+=rate*(-g-reg*bias[j]);
  }
 }
}
''',
    'task5_fossil_next.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve import data,graph
def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(exist_ok=True,parents=True)
 if (out/'fossil.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(list(map(len,h)))],np.int32);seq=np.concatenate(h).astype(np.int32);sp=X.indptr.astype(np.int32);seen=X.indices.astype(np.int32);pop=np.asarray(X.sum(0)).ravel();active=np.flatnonzero(pop>0).astype(np.int32);so=out/'fossil.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_fossil_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.fossil_train.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int]*4+[ctypes.c_float]*2+[fp]*5
 k=24;rng=np.random.RandomState(71);A=rng.normal(0,.05,(I,k)).astype('f');B=rng.normal(0,.05,(I,k)).astype('f');wu=np.zeros((U,3),'f');wg=np.array([.5,.25,.1],'f');bias=np.zeros(I,'f');old=0;result={};known=qi>=0;known[known]&=pop[qi[known]]>0;ctx=np.tile(np.arange(len(contexts)),2);start=time.time()
 for epoch in [5,15,35]:
  lib.fossil_train(U,k,ptr,seq,sp,seen,active,len(active),old,epoch-old,71,.015,.03,A,B,wu,wg,bias);old=epoch;A[pop==0]=0;long=np.zeros((len(contexts),k),'f');recent=np.zeros_like(long)
  for z,hist in enumerate(contexts):
   hh=hist[-20:];long[z]=A[hh].sum(0)/np.sqrt(len(hh));last=hist[-3:][::-1];user=qu[z];recent[z]=(wg[:len(last)]+wu[user,:len(last)])@A[last]
  for name,values in [('long',long),('recent',recent),('joint',long+recent)]:
   pred=np.zeros(len(qi),'f');pred[known]=np.einsum('ij,ij->i',values[ctx[known]],B[qi[known]])+(bias[qi[known]] if name=='joint' else 0);result[f'fossil_e{epoch}_{name}']=pred
  print('FOSSIL',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 assert all(np.isfinite(x).all() for x in [A,B,wu,wg,bias]);np.savez_compressed(out/'fossil.npz',**result);np.savez_compressed(out/'fossil_model.npz',A=A,B=B,wu=wu,wg=wg,bias=bias,pop=pop)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_fpmc_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
extern "C" void train_fpmc(int n,int k,const int* users,const int* last,const int* next,
 const int* ptr,const int* seen,const int* pool,int poolsize,const int* active,int active_size,
 float* P,float* Q,float* A,float* B,float* bias,int start,int epochs,int seed,
 float lr,float reg,float popularity_mix,int uniform_users){
 std::vector<int> order(n);int U=users[n-1]+1;std::vector<int> transitionptr(U+1,0);
 for(int z=0;z<n;z++)transitionptr[users[z]+1]++;for(int u=1;u<=U;u++)transitionptr[u]+=transitionptr[u-1];
 for(int epoch=start;epoch<start+epochs;epoch++){
  std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+epoch);if(!uniform_users)std::shuffle(order.begin(),order.end(),rng);float rate=lr*std::pow(.98f,epoch);
  for(int sampled:order){int row=sampled;if(uniform_users){int u=rng()%U;row=transitionptr[u]+rng()%(transitionptr[u+1]-transitionptr[u]);}int u=users[row],l=last[row],i=next[row],j;int tries=0;
   do {j=(double(rng())/rng.max()<popularity_mix)?pool[rng()%poolsize]:active[rng()%active_size];tries++;}
   while((j==i||std::binary_search(seen+ptr[u],seen+ptr[u+1],j))&&tries<100);
   if(j==i||std::binary_search(seen+ptr[u],seen+ptr[u+1],j))continue;
   float diff=bias[i]-bias[j];
   for(int f=0;f<k;f++)diff+=P[u*k+f]*(Q[i*k+f]-Q[j*k+f])+A[l*k+f]*(B[i*k+f]-B[j*k+f]);
   float g=1.f/(1.f+std::exp(std::min(30.f,std::max(-30.f,diff))));
   bias[i]+=rate*(g-reg*bias[i]);bias[j]+=rate*(-g-reg*bias[j]);
   for(int f=0;f<k;f++){
    float p=P[u*k+f],qi=Q[i*k+f],qj=Q[j*k+f],a=A[l*k+f],bi=B[i*k+f],bj=B[j*k+f];
    P[u*k+f]+=rate*(g*(qi-qj)-reg*p);Q[i*k+f]+=rate*(g*p-reg*qi);Q[j*k+f]+=rate*(-g*p-reg*qj);
    A[l*k+f]+=rate*(g*(bi-bj)-reg*a);B[i*k+f]+=rate*(g*a-reg*bi);B[j*k+f]+=rate*(-g*a-reg*bj);
   }
  }
 }
}
''',
    'task5_improve.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
def data(data,baseline,full=False):
    p=pd.read_csv(data/'task_5_train.csv',usecols=['user','item']);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);assert np.all(np.diff(u)>=0);U,I=len(users),len(items);ptr=np.searchsorted(u,np.arange(U+1));history=[i[ptr[j]:ptr[j+1]].astype(np.int32) for j in range(U)];assert min(map(len,history))>=4
    if full:
        q=pd.read_csv(data/'task_5_test.csv');qu=pd.Index(users).get_indexer(q.user);assert np.all(qu>=0),'Unknown test user';qi=np.r_[pd.Index(items).get_indexer(q.item1),pd.Index(items).get_indexer(q.item2)];contexts=[history[j] for j in qu];qu=np.r_[qu,qu];train=history
    else:
        train=[h[:-2] for h in history];contexts=train+[h[:-1] for h in history];positive=np.r_[[h[-2] for h in history],[h[-1] for h in history]];pool=np.concatenate(train);rng=np.random.RandomState(43);negative=np.empty(len(positive),np.int32)
        for z,pos in enumerate(positive):
            choice=pool[rng.randint(len(pool))]
            while choice==pos:choice=pool[rng.randint(len(pool))]
            negative[z]=choice
        qu=np.tile(np.arange(U),4);qi=np.r_[positive,negative];q=None
    cached=np.load(baseline/('test_pair_features.npz' if full else 'pair_features.npz'));assert np.array_equal(qu,cached['users']) and np.array_equal(qi,cached['items']);return train,contexts,qu.astype(np.int32),qi.astype(np.int32),U,I,q
def stage(args):
    part=args.out/('full' if args.full else 'validation');part.mkdir(parents=True,exist_ok=True);manifest={'full':args.full,'inputs':{f:hashlib.sha256((args.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']}};guard=part/'manifest.json'
    if guard.exists():assert json.loads(guard.read_text())==manifest
    temp=part/f'manifest_{os.getpid()}.partial';temp.write_text(json.dumps(manifest,indent=2));temp.replace(guard);return part
def graph(history,U,I):
    row=np.repeat(np.arange(U),list(map(len,history)));items=np.concatenate(history);X=coo_matrix((np.ones(len(items),np.float32),(row,items)),shape=(U,I)).tocsr();X.sort_indices();assert np.max(X.data)==1;return X
def fpmc(args,uniform_only=False):
    out=stage(args)
    target=out/('fpmc_uniform_features.npz' if uniform_only else 'fpmc_features.npz');configs=[('fpmc64_useruniform',64,.03,.01,.5,23,1)] if uniform_only else [('fpmc32_uniform',32,.02,.02,0.,42,0),('fpmc64_mix',64,.02,.03,.5,17,0)]
    result=dict(np.load(target)) if target.exists() else {}
    if all(name+'_e100_joint' in result for name,*_ in configs):return
    history,contexts,qu,qi,U,I,q=data(args.data_dir,args.baseline,args.full);X=graph(history,U,I);pop=np.asarray(X.sum(0)).ravel();active=np.flatnonzero(pop>0).astype(np.int32);pool=np.concatenate(history).astype(np.int32);tu=np.repeat(np.arange(U),[len(h)-1 for h in history]).astype(np.int32);prev=np.concatenate([h[:-1] for h in history]);nxt=np.concatenate([h[1:] for h in history]);ptr=X.indptr.astype(np.int32);seen=X.indices.astype(np.int32)
    source=Path(__file__).with_name('task5_fpmc_kernel.cpp');so=out/('fpmc_uniform.so' if uniform_only else 'fpmc_v2.so');subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so.resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train_fpmc.argtypes=[ctypes.c_int]*2+[ip]*6+[ctypes.c_int,ip,ctypes.c_int]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*3+[ctypes.c_int]
    ctx=np.tile(np.arange(len(contexts)),2);valid=qi>=0;known=valid.copy();known[valid]&=pop[qi[valid]]>0;last=np.array([h[-1] for h in contexts]);source_count=np.bincount(prev,minlength=I);start=time.time()
    for name,k,lr,reg,mix,seed,uniform in configs:
        if name+'_e100_joint' in result:continue
        rng=np.random.RandomState(seed);P=rng.normal(0,.05,(U,k)).astype(np.float32);Q=rng.normal(0,.05,(I,k)).astype(np.float32);A=rng.normal(0,.05,(I,k)).astype(np.float32);B=rng.normal(0,.05,(I,k)).astype(np.float32);bias=np.zeros(I,np.float32);epoch0=0
        if (out/(name+'_model.npz')).exists() and name+'_e40_joint' in result:
            saved=np.load(out/(name+'_model.npz'));P=saved['P'];Q=saved['Q'];A=saved['A'];B=saved['B'];bias=saved['bias'];epoch0=int(saved['epoch']) if 'epoch' in saved else 40
        for epoch in [15,40,100]:
            if epoch<=epoch0:continue
            lib.train_fpmc(len(tu),k,tu,prev,nxt,ptr,seen,pool,len(pool),active,len(active),P,Q,A,B,bias,epoch0,epoch-epoch0,seed,lr,reg,mix,uniform);epoch0=epoch;assert all(np.isfinite(z).all() for z in [P,Q,A,B,bias]);context=np.zeros((len(contexts),k),np.float32)
            for z,h in enumerate(contexts):
                hh=h[-3:][::-1];w=.7**np.arange(len(hh));w*=source_count[hh]>0;context[z]=w@A[hh]/max(w.sum(),1e-8)
            personal=np.zeros(len(qi));markov=np.zeros(len(qi));recent=np.zeros(len(qi));ib=np.zeros(len(qi));personal[known]=np.einsum('ij,ij->i',P[qu[known]],Q[qi[known]]);ok=known&(source_count[last[ctx]]>0);markov[ok]=np.einsum('ij,ij->i',A[last[ctx[ok]]],B[qi[ok]]);recent[known]=np.einsum('ij,ij->i',context[ctx[known]],B[qi[known]]);ib[known]=bias[qi[known]]
            for label,z in [('personal',personal),('last',markov),('recent3',recent),('bias',ib),('joint',personal+markov+ib)]:result[f'{name}_e{epoch}_{label}']=z
            if epoch==100:np.savez_compressed(out/(name+'_model.npz'),P=P,Q=Q,A=A,B=B,bias=bias,source_count=source_count,pop=pop,epoch=epoch)
            print(name,epoch,'seconds',round(time.time()-start),flush=True)
    np.savez_compressed(target,**result)
def neighbors(args):
    out=stage(args)
    if (out/'neighbor_features.npz').exists():return
    history,contexts,qu,qi,U,I,q=data(args.data_dir,args.baseline,args.full);X=graph(history,U,I);N=len(contexts);ctx=np.tile(np.arange(N),2);pop=np.asarray(X.sum(0)).ravel();length=np.array(list(map(len,history)));down=(1/(1+length/max(length.mean(),1))).astype(np.float32);weighted=X.multiply(down[:,None]).tocsr();wp=np.asarray(weighted.sum(0)).ravel();G=X.T.tocsr();groups={}
    for z,j in enumerate(qi):
        if j>=0 and pop[j]>0:groups.setdefault(j,[]).append(z)
    result={f'weighted_knn_a{alpha}_s{shrink}_{agg}':np.zeros(len(qi),np.float32) for alpha in [.25,.5,.75] for shrink in [0,5] for agg in ['max','mean','last','recent3','recent10','decay']};keys=np.array(list(groups));start=time.time()
    for off in range(0,len(keys),64):
        batch=keys[off:off+64];co=(G[batch]@weighted).tocsr();co.sort_indices()
        for z,j in enumerate(batch):
            lo,hi=co.indptr[z:z+2];nb=co.indices[lo:hi];count=co.data[lo:hi]
            for row in groups[j]:
                h=contexts[ctx[row]];idx=np.searchsorted(nb,h);ok=idx<len(nb);ok[ok]&=nb[idx[ok]]==h[ok];shared=np.zeros(len(h));shared[ok]=count[idx[ok]];shared[h==j]=0
                for alpha in [.25,.5,.75]:
                    similarity=shared/(max(wp[j],1e-8)**(1-alpha)*np.maximum(wp[h],1e-8)**alpha)
                    for shrink in [0,5]:
                        value=similarity if shrink==0 else similarity*shared/(shared+shrink);prefix=f'weighted_knn_a{alpha}_s{shrink}_';result[prefix+'max'][row]=value.max();result[prefix+'mean'][row]=value.mean();result[prefix+'last'][row]=value[-1];result[prefix+'recent3'][row]=value[-3:].mean();result[prefix+'recent10'][row]=value[-10:].mean();w=.9**np.arange(len(h))[::-1];result[prefix+'decay'][row]=value@w/w.sum()
        if off%10000<64:print('Weighted item neighbors',off,len(keys),'seconds',round(time.time()-start),flush=True)
    positions=coo_matrix((np.concatenate([np.arange(len(h)) for h in history]),(np.repeat(np.arange(U),length),np.concatenate(history))),shape=X.shape).tocsc();XC=X.tocsc();assert np.array_equal(XC.indptr,positions.indptr) and np.array_equal(XC.indices,positions.indices)
    idf=np.log1p(U/np.maximum(pop,1)).astype(np.float32);D=X.multiply(idf).tocsr();norm=np.sqrt(np.asarray(D.multiply(D).sum(1)).ravel());recent=[h[-20:] for h in contexts];rr=np.repeat(np.arange(N),list(map(len,recent)));ii=np.concatenate(recent);ww=np.concatenate([(.9**np.arange(len(h))[::-1])*idf[h] for h in recent]).astype(np.float32);C=coo_matrix((ww,(rr,ii)),shape=(N,I)).tocsr();cnorm=np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel());DT=D.T.tocsr()
    for k in [50,200]:
        for name in ['all','tail5','forward5']:result[f'sessionnn_{k}_{name}']=np.zeros(len(qi),np.float32)
    result['sessionnn_support']=np.zeros(len(qi),np.float32)
    def matches(j,neighbors):
        present=np.zeros(len(neighbors),bool);pos=np.zeros(len(neighbors))
        if j<0:return present,pos
        lo,hi=XC.indptr[j:j+2];other=XC.indices[lo:hi];where=np.searchsorted(other,neighbors);ok=where<len(other);ok[ok]&=other[where[ok]]==neighbors[ok];present[ok]=True;pos[ok]=positions.data[lo:hi][where[ok]];return present,pos
    for off in range(0,N,64):
        co=(C[off:off+64]@DT).tocsr()
        for z in range(co.shape[0]):
            row=off+z;lo,hi=co.indptr[z:z+2];nb=co.indices[lo:hi];sim=co.data[lo:hi]/np.maximum(cnorm[row]*norm[nb],1e-8);ok=nb!=qu[row];nb=nb[ok];sim=sim[ok];take=np.argpartition(sim,-200)[-200:] if len(sim)>200 else np.arange(len(sim));take=take[np.lexsort((nb[take],-sim[take]))];nb=nb[take];sim=sim[take];haslast,lastpos=matches(contexts[row][-1],nb)
            for target in [row,N+row]:
                present,pos=matches(qi[target],nb);tail=np.exp(-(length[nb]-1-pos)/5)*present;delta=pos-lastpos;future=present&haslast&(delta>=1)&(delta<=5);forward=np.where(future,1/np.maximum(delta,1),0);result['sessionnn_support'][target]=np.log1p(present.sum())
                for k in [50,200]:
                    w=sim[:k];den=w.sum()+1e-8;result[f'sessionnn_{k}_all'][target]=w@present[:k]/den;result[f'sessionnn_{k}_tail5'][target]=w@tail[:k]/den;result[f'sessionnn_{k}_forward5'][target]=w@forward[:k]/den
        if off%10000<64:print('Session neighbors',off,N,'seconds',round(time.time()-start),flush=True)
    assert all(np.isfinite(v).all() for v in result.values());np.savez_compressed(out/'neighbor_features.npz',**result)
def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--full',action='store_true');ap.add_argument('--worker',choices=['fpmc','uniform','neighbors'],required=True);args=ap.parse_args()
    part=args.out/('full' if args.full else 'validation');part.mkdir(parents=True,exist_ok=True)
    with open(part/(args.worker+'.lock'),'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.worker=='fpmc':fpmc(args)
        elif args.worker=='uniform':fpmc(args,True)
        else:neighbors(args)
if __name__=='__main__':main()
''',
    'task5_improve_stack.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
def load(args,full=False):
    old=np.load(args.baseline/('test_pair_features.npz' if full else 'pair_features.npz'))['scores'];part=args.out/('full' if full else 'validation');names=[];cols=[]
    for kind in ['fpmc','fpmc_uniform','neighbor']:
        z=np.load(part/(kind+'_features.npz'));keys=z.files
        if kind=='fpmc':
            models=['fpmc32_uniform','fpmc64_mix'];fields=['personal','last','recent3','bias','joint'];keys=[f'{m}_e{e}_{f}' for m in models for e in [15,40] for f in fields]+[f'{m}_e100_{f}' for m in models for f in fields];assert set(keys)==set(z.files)
        names+=keys;cols += [z[k] for k in keys]
    scores=np.c_[old,np.column_stack(cols)].astype(np.float32);N=len(scores)//2;return scores[:N]-scores[N:],scores[:N]+scores[N:],old[:N]-old[N:],old[:N]+old[N:],names
''',
    'task5_metric_kernel.cpp': r'''#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
extern "C" void metric_train(int U,int k,const int* ptr,const int* seq,const int* seenptr,const int* seen,const int* pool,int poolsize,const int* active,int na,float* P,float* Q,float* global,float* bias,int kind,int start,int epochs,int seed,float lr,float reg){
 int n=poolsize-U;std::mt19937 rng(seed);std::vector<float> gp(k),gl(k),gi(k),gj(k);
 auto project=[&](float* v){float norm=0;for(int f=0;f<k;f++)norm+=v[f]*v[f];if(norm>1){float scale=1/std::sqrt(norm);for(int f=0;f<k;f++)v[f]*=scale;}};
 for(int epoch=start;epoch<start+epochs;epoch++){rng.seed(seed+epoch);float rate=lr*std::pow(.99f,epoch);
  for(int z=0;z<n;z++){int u=rng()%U,lo=ptr[u],hi=ptr[u+1];if(hi-lo<2)continue;int pos=lo+1+rng()%(hi-lo-1);int l=seq[pos-1],i=seq[pos],j;int tries=0;
   do{j=(rng()%2)?pool[rng()%poolsize]:active[rng()%na];tries++;}while((j==i||std::binary_search(seen+seenptr[u],seen+seenptr[u+1],j))&&tries<100);
   if(j==i||std::binary_search(seen+seenptr[u],seen+seenptr[u+1],j))continue;
   float diff=bias[i]-bias[j];
   for(int f=0;f<k;f++){float p=P[u*k+f],lq=Q[l*k+f],a=Q[i*k+f],b=Q[j*k+f];if(kind==0){diff+=.5f*((p-b)*(p-b)-(p-a)*(p-a)+(lq-b)*(lq-b)-(lq-a)*(lq-a));gp[f]=a-b;gl[f]=a-b;gi[f]=p+lq-2*a;gj[f]=-p-lq+2*b;}else{float c=p+lq+global[f];diff+=(c-b)*(c-b)-(c-a)*(c-a);gp[f]=2*(a-b);gl[f]=gp[f];gi[f]=2*(c-a);gj[f]=-2*(c-b);}}
   float g=1/(1+std::exp(std::clamp(diff,-30.f,30.f)));bias[i]+=rate*(g-reg*bias[i]);bias[j]+=rate*(-g-reg*bias[j]);
   for(int f=0;f<k;f++){P[u*k+f]+=rate*(g*gp[f]-reg*P[u*k+f]);Q[l*k+f]+=rate*(g*gl[f]-reg*Q[l*k+f]);Q[i*k+f]+=rate*(g*gi[f]-reg*Q[i*k+f]);Q[j*k+f]+=rate*(g*gj[f]-reg*Q[j*k+f]);if(kind==1)global[f]+=rate*(g*gp[f]-reg*global[f]);}
   project(P+u*k);project(Q+l*k);project(Q+i*k);project(Q+j*k);if(kind==1)project(global);
  }
 }
}
''',
    'task5_next.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve import data,graph
def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(list(map(len,h)))],np.int32);seq=np.concatenate(h).astype(np.int32);sp=X.indptr.astype(np.int32);seen=X.indices.astype(np.int32);pop=np.asarray(X.sum(0)).ravel();active=np.flatnonzero(pop>0).astype(np.int32);so=out/'metric.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_metric_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.metric_train.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int,ip,ctypes.c_int]+[fp]*4+[ctypes.c_int]*4+[ctypes.c_float]*2
 ctx=np.tile(np.arange(len(contexts)),2);last=np.array([x[-1] for x in contexts]);known=qi>=0;known[known]&=pop[qi[known]]>0;result={};start=time.time()
 for kind,name in [(0,'prme'),(1,'transrec')]:
  k=32;rng=np.random.RandomState(92+kind);P=rng.normal(0,.05,(U,k)).astype('f');Q=rng.normal(0,.05,(I,k)).astype('f');globalv=np.zeros(k,'f');bias=np.zeros(I,'f');old=0
  for epoch in [20,60,120]:
   lib.metric_train(U,k,ptr,seq,sp,seen,seq,len(seq),active,len(active),P,Q,globalv,bias,kind,old,epoch-old,92+kind,.02,.01);old=epoch;assert all(np.isfinite(x).all() for x in [P,Q,globalv,bias]);qq=qi[known];us=qu[known];prev=last[ctx[known]];pers=-np.sum((P[us]-Q[qq])**2,axis=1);transition=-np.sum((Q[prev]-Q[qq])**2,axis=1);joint=.5*(pers+transition) if kind==0 else -np.sum((P[us]+Q[prev]+globalv-Q[qq])**2,axis=1)
   source_known=pop[prev]>0;transition[~source_known]=0;joint[~source_known]=0
   for label,val in [('personal',pers),('last',transition),('bias',bias[qq]),('joint',joint+bias[qq])]:
    z=np.zeros(len(qi),'f');z[known]=val;result[f'{name}_e{epoch}_{label}']=z
   print(name,epoch,'seconds',round(time.time()-start),flush=True)
  np.savez_compressed(out/(name+'_model.npz'),P=P,Q=Q,globalv=globalv,bias=bias,pop=pop)
 np.savez_compressed(out/'metrics.npz',**result)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_next_stack.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve_stack import load
from task5_improve import data
def matrices(a,full=False):
 d,s,od,oscore,names=load(SimpleNamespace(baseline=a.baseline,out=a.previous),full);part=a.out/('full' if full else 'validation');cols=[];extra=[]
 for kind in ['metrics','ease','fossil']:
  z=dict(np.load(part/(kind+'.npz')))
  if kind=='metrics':
   h,ctx,qu,qi,U,I,q=data(a.data_dir,a.baseline,full);pop=np.load(part/'prme_model.npz')['pop'];cold=np.tile(np.array([pop[x[-1]]==0 for x in ctx]),2)
   for name in z:
    if name.endswith('_last'):z[name][cold]=0
    if name.endswith('_joint'):z[name][cold]=z[name.replace('_joint','_bias')][cold]
  cols.extend(list(z.values()));extra+=list(z)
 scores=np.column_stack(cols);N=len(scores)//2;return np.c_[d,scores[:N]-scores[N:]],np.c_[s,scores[:N]+scores[N:]],d,s,names+extra
''',
    'task5_order.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_improve import data
def transitions(history,I):
 lengths=np.array(list(map(len,history)));users=np.repeat(np.arange(len(history)),lengths);seq=np.concatenate(history);down=1/np.sqrt(lengths);H=csr_matrix((I,I),dtype=np.float32)
 for distance in range(1,11):
  ok=users[:-distance]==users[distance:];s=seq[:-distance][ok];t=seq[distance:][ok];w=(down[users[:-distance][ok]]/distance).astype('f');H=H+coo_matrix((w,(s,t)),shape=(I,I)).tocsr();H.sort_indices()
  if distance in [1,3,10]:yield distance,H.copy()
def score_graph(H,contexts,qi):
 N=len(contexts);recent=[h[-20:][::-1] for h in contexts];lengths=np.array(list(map(len,recent)));owner=np.repeat(np.arange(N),lengths);src=np.concatenate(recent);pos=np.concatenate([np.arange(n) for n in lengths]);weight=.8**pos;rowsum=np.asarray(H.sum(1)).ravel();pop=np.asarray(H.sum(0)).ravel();out={};last=np.array([h[-1] for h in contexts]);P=H.multiply((1/np.maximum(rowsum,1e-8))[:,None]).tocsr()
 for side in [0,1]:
  target=np.repeat(qi[side*N:(side+1)*N],lengths);good=target>=0;fwd=np.zeros(len(src));rev=np.zeros(len(src));fwd[good]=np.asarray(H[src[good],target[good]]).ravel();rev[good]=np.asarray(H[target[good],src[good]]).ravel();p=fwd/np.maximum(rowsum[src],1e-8);r=np.zeros_like(rev);r[good]=rev[good]/np.maximum(rowsum[target[good]],1e-8);lift=np.zeros_like(p);lift[good]=p[good]/np.sqrt(np.maximum(pop[target[good]]/max(pop.sum(),1),1e-12));den=np.bincount(owner,weights=weight,minlength=N)
  vals={'last':np.bincount(owner[pos==0],weights=p[pos==0],minlength=N),'recent3':np.bincount(owner[pos<3],weights=p[pos<3],minlength=N)/np.minimum(lengths,3),'decay':np.bincount(owner,weights=p*weight,minlength=N)/den,'reverse_decay':np.bincount(owner,weights=r*weight,minlength=N)/den,'order_evidence':np.bincount(owner,weights=weight*np.log((fwd+.1)/(rev+.1)),minlength=N)/den,'lift_decay':np.bincount(owner,weights=lift*weight,minlength=N)/den};mx=np.zeros(N);np.maximum.at(mx,owner,fwd);vals['count_max']=np.log1p(mx)
  for name,v in vals.items():
   if name not in out:out[name]=np.zeros(len(qi),'f')
   out[name][side*N:(side+1)*N]=v
 return out,P
def diffusion(P,contexts,qi,so):
 lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.paths.argtypes=[ctypes.c_int]+[ip]*4+[fp]+[ip]*2+[fp]*2;C=P.tocsc();rp=P.indptr.astype('i');ri=P.indices.astype('i');rv=P.data.astype('f');cp=C.indptr.astype('i');ci=C.indices.astype('i');cv=C.data.astype('f');N=len(contexts);recent=[h[-3:][::-1] for h in contexts];lens=np.array(list(map(len,recent)));src=np.concatenate(recent).astype('i');owner=np.repeat(np.arange(N),lens);out={k:np.zeros(len(qi),'f') for k in ['twohop_last','twohop_recent3','twohop_reverse']};first=np.r_[0,np.cumsum(lens)[:-1]]
 for side in [0,1]:
  dst=np.repeat(qi[side*N:(side+1)*N],lens).astype('i');value=np.zeros(len(src),'f');lib.paths(len(src),src,dst,rp,ri,rv,cp,ci,cv,value);out['twohop_last'][side*N:(side+1)*N]=value[first];out['twohop_recent3'][side*N:(side+1)*N]=np.bincount(owner,weights=value,minlength=N)/lens;back=np.zeros(len(src),'f');lib.paths(len(src),dst,src,rp,ri,rv,cp,ci,cv,back);out['twohop_reverse'][side*N:(side+1)*N]=np.bincount(owner,weights=back,minlength=N)/lens
 return out
def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True);target=part/'order.npz'
 if target.exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);source=Path(__file__).with_name('task5_order_kernel.cpp');so=part/'paths.so';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(so)],check=True);result={};start=time.time()
 for window,H in transitions(h,I):
  vals,P=score_graph(H,contexts,qi);result.update({f'skip{window}_{k}':v for k,v in vals.items()})
  if window==3:result.update(diffusion(P,contexts,qi,so))
  print('ORDER_GRAPH',a.full,window,'edges',H.nnz,'seconds',round(time.time()-start),flush=True)
 temp=part/'order.tmp.npz';np.savez_compressed(temp,**result);temp.replace(target);(part/'order_manifest.json').write_text(json.dumps({'full':a.full,'sequence_only':True,'all_ratings_retained':True,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']}},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
''',
    'task5_order_kernel.cpp': r'''extern "C" void paths(int n,const int* src,const int* dst,const int* rp,const int* ri,const float* rv,const int* cp,const int* ci,const float* cv,float* out){
 for(int z=0;z<n;z++){int s=src[z],t=dst[z];double total=0;if(s<0||t<0){out[z]=0;continue;}int a=rp[s],ae=rp[s+1],b=cp[t],be=cp[t+1];while(a<ae&&b<be){if(ri[a]<ci[b])a++;else if(ri[a]>ci[b])b++;else{total+=double(rv[a])*cv[b];a++;b++;}}out[z]=total;}
}
''',
    'task5_preference_round3.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_round3 import load as core_load
def load(a,full=False):
 parts=core_load(a,full);d,s=parts[:2];cols=[];names=[]
 for kind in ['bpr','fold']:
  z=np.load(a.out/('full' if full else 'validation')/(kind+'.npz'));cols.extend([z[k] for k in z.files]);names+=list(z.files)
 scores=np.column_stack(cols);N=len(scores)//2;return np.c_[d,scores[:N]-scores[N:]].astype('f'),np.c_[s,scores[:N]+scores[N:]].astype('f'),parts,names
''',
    'task5_rolling.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
W=Path('/workspace/work');ROOT=Path(__file__).parent;OUT=W/'task5_rolling';OUT.mkdir(exist_ok=True)
DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
SEEDS=[43,441,997]
def cohort(*unused,**kw):
 if hasattr(cohort,'cache'):return cohort.cache
 p=pd.read_csv(DATA/'task_5_train.csv',usecols=['user','item']);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);U,I=len(users),len(items);ptr=np.searchsorted(u,np.arange(U+1));H=[i[ptr[v]:ptr[v+1]].astype('i') for v in range(U)];h=[x[:-1] for x in H];pos=np.array([x[-1] for x in H],'i');pool=np.concatenate(h);neg=[]
 for seed in SEEDS:
  rng=np.random.RandomState(seed);n=np.empty(U,'i')
  for z,target in enumerate(pos):
   j=pool[rng.randint(len(pool))]
   while j==target:j=pool[rng.randint(len(pool))]
   n[z]=j
  neg.append(n)
 contexts=h*len(SEEDS);N=len(contexts);qu=np.tile(np.arange(U,dtype='i'),2*len(SEEDS));qi=np.r_[np.tile(pos,len(SEEDS)),np.concatenate(neg)].astype('i')
 assert all(target not in hh for target,hh in zip(pos,h));pop=np.bincount(pool,minlength=I);assert np.all(pop[[hh[-1] for hh in h]]>0)
 cohort.cache=(h,contexts,qu,qi,U,I,None);return cohort.cache
def _worker(kind):
 import task5_improve as imp
 imp.data=cohort
 original=np.savez_compressed
 def compact(path,*args,**kw):
  name=Path(path).name
  if name.endswith('_model.npz') and name not in ['fpmc32_uniform_model.npz','fpmc64_useruniform_model.npz']:return
  original(path,*args,**{k:v.astype('f') if hasattr(v,'dtype') and v.dtype.kind=='f' else v for k,v in kw.items()})
 np.savez_compressed=compact
 a=SimpleNamespace(data_dir=DATA,baseline=W/'task5/results',out=OUT,previous=OUT,full=False)
 if kind=='base':
  import task5_ensemble as m
  h,contexts,qu,qi,U,I,_=cohort();X,T,models=m.prepare(h,I,OUT/'factors');models['tail_popularity']=m.tail_pop(h,I);names,scores=m.features(X,T,models,contexts,qu,qi);original(OUT/'validation/base.npz',scores=scores.astype('f'));(OUT/'base_names.json').write_text(json.dumps(names))
  for f in (OUT/'factors').glob('*.npz'):f.unlink()
 elif kind in ['fpmc','uniform','neighbors']:
  imp.fpmc(a,kind=='uniform') if kind!='neighbors' else imp.neighbors(a)
 else:
  module={'metrics':'task5_next','ease':'task5_ease_next','fossil':'task5_fossil_next','order':'task5_order','era':'task5_era','attention':'task5_attention','bpr':'task5_bpr_round3','fold':'task5_fold'}[kind]
  m=__import__(module);m.data=cohort;m.run(a)
 if kind=='attention':(OUT/'validation/fpmc32_uniform_model.npz').unlink(missing_ok=True)
 if kind=='fold':(OUT/'validation/fpmc64_useruniform_model.npz').unlink(missing_ok=True)
 print('WORKER_DONE',kind,flush=True)
def worker(kind):
 with open(OUT/(kind+'.lock'),'w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (OUT/(kind+'.done')).exists():return
  _worker(kind)
  (OUT/(kind+'.done')).write_text('ok')
def args():return SimpleNamespace(data_dir=DATA,baseline=W/'task5/results',previous=W/'task5_improved',reference=W/'task5_next',out=W/'task5_round3')
def refreshed(width=260):
 from task5_preference_round3 import load
 d,s,parts,extra=load(args());names=(json.loads((W/'task5/results/results.json').read_text())['feature_names']+parts[-1]+extra)[:width];del d,s,parts;gc.collect()
 bank={};base=np.load(OUT/'validation/base.npz')['scores'];base_names=json.loads((OUT/'base_names.json').read_text());bank.update(zip(base_names,base.T))
 for f in ['fpmc_features','fpmc_uniform_features','neighbor_features','metrics','ease','fossil','order','era','attention','bpr','fold']:
  if width==224 and f in ['bpr','fold']:continue
  with np.load(OUT/'validation'/(f+'.npz')) as z:
   for key in z.files:
    assert key not in bank,key
    bank[key]=z[key]
 assert len(names)==width and all(n in bank for n in names)
 score=np.column_stack([bank[n] for n in names]).astype('f');del bank;N=len(score)//2;return score[:N]-score[N:],score[:N]+score[N:],names
if __name__ == '__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);worker(ap.parse_args().worker)
''',
    'task5_round3.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task5_next_stack import matrices as previous_matrices
def design(d,s):return np.c_[d,s].astype('f')
def load(a,full=False):
 arg=SimpleNamespace(data_dir=a.data_dir,baseline=a.baseline,previous=a.previous,out=a.reference);old,olds,earlier,earliers,names=previous_matrices(arg,full);cols=[];extra=[]
 for kind in ['order','era','attention']:
  z=dict(np.load(a.out/('full' if full else 'validation')/(kind+'.npz')));cols.extend(z.values());extra+=list(z)
 scores=np.column_stack(cols);N=len(scores)//2;d=np.c_[old,scores[:N]-scores[N:]].astype('f');s=np.c_[olds,scores[:N]+scores[N:]].astype('f');assert np.isfinite(d).all() and np.isfinite(s).all();return d,s,old,olds,earlier,earliers,names+extra
def nn_margin(model,scale,d,s):return model.predict_proba(scale.transform(design(d,s)))[:,1]-model.predict_proba(scale.transform(design(-d,s)))[:,1]
def nn_fit(d,s,epochs,seed=71,hidden=(64,32),alpha=.1):
 X=np.r_[design(d,s),design(-d,s)];y=np.r_[np.ones(len(d)),np.zeros(len(d))];scale=StandardScaler().fit(X);X=scale.transform(X);model=MLPClassifier(hidden_layer_sizes=hidden,alpha=alpha,learning_rate_init=.001,batch_size=512,random_state=seed)
 for _ in range(epochs):model.partial_fit(X,y,classes=[0,1])
 return (model,scale),None
''',
    # TASK 6 MODEL SOURCE
    'task6_history_features.py': r'''from final_imports import *
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
 for period in [7,365.25]:
  angle=2*np.pi*(t[k]/86400 % period)/period
  out.extend([np.sin(angle),np.cos(angle)])
 return out
def basic_features(times,k):
    gaps=np.diff(times[:k+1]);recent=gaps[-5:]
    last=list(gaps[-5:][::-1]);last += [np.nan]*(5-len(last))
    return last+[len(gaps),float(np.median(gaps)),float(np.mean(gaps)),float(np.std(gaps)),float(np.min(gaps)),float(np.max(gaps)),float(np.quantile(gaps,.25)),float(np.quantile(gaps,.75)),float(np.median(recent)),float(np.mean(recent)),float(np.std(recent)),float(times[k]),float(times[k]%604800),float(times[k]-times[0])]
''',
    'task6_improve.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from task6_pipeline import load_data
def event_features(item,t,allowed,candidate,last,pace,target=None):
    target=np.full(len(candidate),-1,int) if target is None else np.asarray(target);valid=allowed&(t>=0)&(t<=4102444800);rows=np.flatnonzero(valid);rows=rows[np.lexsort((t[rows],item[rows]))];codes=item[rows];dates=t[rows]/86400;ptr=np.searchsorted(codes,np.arange(item.max()+2));rank=np.full(len(t),-1,int);rank[rows]=np.arange(len(rows));out=np.full((len(candidate),29),np.nan,np.float32);out[:,0]=0;out[:,-1]=1;groups={}
    for z,j in enumerate(candidate):
        if j>=0:groups.setdefault(j,[]).append(z)
    for j,indices in groups.items():
        lo,hi=ptr[j:j+2];A=dates[lo:hi]
        if not len(A):continue
        for qs in np.array_split(np.array(indices),max(1,(len(indices)+127)//128)):
            q=last[qs]/86400;gap=A[None,:]-q[:,None];weights=(gap>=0)&(last[qs,None]>=0)&(last[qs,None]<=4102444800);removed=np.where(target[qs]>=0,rank[np.maximum(target[qs],0)]-lo,-1);rem=(removed>=0)&(removed<len(A));weights[np.flatnonzero(rem),removed[rem]]=False;count=weights.sum(1);ok=count>0;out[qs,0]=np.log1p(count);out[qs,-1]=~ok
            total=weights.sum(1);mean=(np.maximum(gap,0)*weights).sum(1)/np.maximum(total,1);var=((np.maximum(gap,0)**2)*weights).sum(1)/np.maximum(total,1)-mean**2;out[qs,1]=np.where(ok,np.log1p(np.maximum(var,0)**.5),np.nan);out[qs,2]=np.where(ok,np.log1p(mean),np.nan);cumulative=np.cumsum(weights,axis=1)
            for col,quantile in enumerate([0,.1,.25,.5,.75,.9,1],3):
                frac=quantile*np.maximum(count-1,0);left=np.floor(frac).astype(int);right=np.ceil(frac).astype(int);il=np.argmax(cumulative>left[:,None],axis=1);ir=np.argmax(cumulative>right[:,None],axis=1);value=A[il]*(1-(frac-left))+A[ir]*(frac-left)-q;out[qs,col]=np.where(ok,np.log1p(np.maximum(value,0)),np.nan)
            for col,window in enumerate([7,30,90,365],10):
                c=(weights&(gap<=window)).sum(1);out[qs,col]=np.log1p(c);out[qs,col+4]=np.divide(c,count,out=np.zeros(len(q)),where=ok)
            for k,sigma in enumerate([1.,2.]):
                w=np.exp(-.5*((np.log1p(np.maximum(gap,0))-pace[qs,None])/sigma)**2)*weights;mass=w.sum(1);cw=np.cumsum(w,axis=1);good=mass>1e-12;offset=18+5*k
                for col,quantile in enumerate([.25,.5,.75]):
                    where=np.argmax(cw>=quantile*mass[:,None],axis=1);out[qs,offset+col]=np.where(good,np.log1p(np.maximum(A[where]-q,0)),np.nan)
                mean=(w*np.maximum(gap,0)).sum(1)/np.maximum(mass,1e-12);out[qs,offset+3]=np.where(good,np.log1p(mean),np.nan);out[qs,offset+4]=np.log1p(mass)
    assert not np.isinf(out).any();return out
def build(a):
    a.out.mkdir(parents=True,exist_ok=True);manifest={f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_6_train.csv','task_6_test.csv']};path=a.out/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    path.write_text(json.dumps(manifest,indent=2));p,test,u,users,i,items,t,ratings,starts,early,v,e,rows=load_data(a.data_dir);z=np.load(a.legacy/'task6_rich.npz');old=np.load(a.legacy/'task6_final/book_features.npz');full=np.load(a.legacy/'task6_final/full_features.npz');assert np.array_equal(rows,old['all_rows']);core=np.r_[z['X'],z['V'],z['T']];y=np.r_[z['y'],z['vy'],z['ty']];assert np.allclose(y,(t[rows]-t[rows-1])/86400);n=len(early);m=len(v)
    if not (a.out/'validation.npz').exists():
        allowed=np.ones(len(p),bool);allowed[v]=False;allowed[e]=False;extras=event_features(i,t,allowed,i[rows],t[rows-1],core[:,19],rows);book=np.r_[old['X'],old['V'],old['T']];X=np.c_[core,book,extras];progress=(rows-starts[u[rows]])/(starts[u[rows]+1]-starts[u[rows]]-1);np.savez_compressed(a.out/'validation.npz',X=X,y=y,rows=rows,n=n,m=m,progress=progress);print('Task6 conditional validation features ready',X.shape,flush=True)
    if a.full and not (a.out/'full.npz').exists():
        extras=event_features(i,t,np.ones(len(p),bool),i[rows],t[rows-1],core[:,19],rows);um=pd.Index(users).get_indexer(test.user);last=t[starts[um+1]-1];qi=pd.Index(items).get_indexer(test.item);qextras=event_features(i,t,np.ones(len(p),bool),qi,last,full['query_rich'][:,19]);X=np.c_[core,full['book'],extras];Q=np.c_[full['query_rich'],full['query_book'],qextras];np.savez_compressed(a.out/'full.npz',X=X,Q=Q,rows=rows,y=y);print('Task6 full conditional features ready',X.shape,Q.shape,flush=True)
def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--full',action='store_true');a=ap.parse_args();build(a)
if __name__=='__main__':main()
''',
    'task6_improve_models.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
def model(name,iters=1800,threads=1):
    if name=='cat':return CatBoostRegressor(iterations=iters,depth=6,learning_rate=.04,l2_leaf_reg=5,random_seed=23,loss_function='MAE',thread_count=threads,verbose=300,allow_writing_files=False,cat_features=['user','next_item'],max_ctr_complexity=1)
    return LGBMRegressor(objective='regression_l1',n_estimators=iters,num_leaves=31 if name=='lgb' else 63,learning_rate=.04,min_child_samples=200,reg_lambda=50,max_bin=127,colsample_bytree=.9,subsample=.9,subsample_freq=1,path_smooth=10,n_jobs=1,random_state=42,verbosity=-1)
def framing(X,p,rows=None,test=None):
    f=pd.DataFrame(X,columns=['f'+str(j) for j in range(X.shape[1])]);source=p.iloc[rows] if test is None else test;f['user']=source.user.to_numpy();f['next_item']=source.item.to_numpy();return f
def user_weights(p,rows):
    user=pd.factorize(p.user.iloc[rows])[0];position=p.groupby('user',sort=False).cumcount().to_numpy()[rows];last=np.zeros(user.max()+1);np.maximum.at(last,user,position);raw=1+4*(position/np.maximum(last[user],1))**2;total=np.bincount(user,weights=raw);return raw/total[user]*(len(rows)/len(total))
def full_fit(a,name,iterations,p,full,progress,test):
    M=model(name,iterations,threads=2);XX=full['X'];QQ=full['Q'];path=a.out/(name+'_full.model');weight=1+4*progress**2 if name=='recent' else (user_weights(p,full['rows']) if name=='balanced' else None)
    if name=='cat':XX=framing(XX,p,full['rows']);QQ=framing(QQ,p,test=test)
    if path.exists():
        if name=='cat':M.load_model(str(path))
        else:
            M=lgb.Booster(model_file=str(path))
    else:
        if name=='cat':M.fit(XX,full['y']);M.save_model(str(path))
        else:M.fit(XX,full['y'],sample_weight=weight);M.booster_.save_model(str(path))
    return np.maximum(0,M.predict(QQ))
''',
    'task6_item_time.py': r'''from final_imports import *
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
''',
    'task6_next.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from task6_pipeline import load_data
def peers(i,t,u,allowed,candidate,last,queryuser,pace):
 gap=np.r_[0,np.diff(t)/86400];same=np.r_[False,u[1:]==u[:-1]];valid=allowed&same&(gap>=0)&(t>=0)&(t<=4102444800)
 lg=np.log1p(np.maximum(gap,0));# past mean log gap excludes the current event
 csum=np.cumsum(np.where(same,lg,0));starts=np.searchsorted(u,np.arange(u.max()+2));position=np.arange(len(u))-starts[u];base=np.r_[0,csum[:-1]];prior=(base-np.where(starts[u]>0,csum[np.maximum(starts[u]-1,0)],0))/np.maximum(position-1,1);prior[position<=1]=lg[position<=1] # no pace available at first event
 valid&=position>1
 rows=np.flatnonzero(valid);rows=rows[np.argsort(i[rows],kind='stable')];ids=i[rows];ptr=np.searchsorted(ids,np.arange(i.max()+2));out=np.full((len(candidate),23),np.nan,'f');out[:,0]=0;groups={}
 for z,j in enumerate(candidate):
  if j>=0:groups.setdefault(j,[]).append(z)
 for j,indices in groups.items():
  lo,hi=ptr[j:j+2];rr=rows[lo:hi]
  if not len(rr):continue
  for qs in np.array_split(np.array(indices),max(1,(len(indices)+63)//64)):
   mask=u[rr][None,:]!=queryuser[qs,None];g=lg[rr];norm=g-prior[rr];D=abs(t[rr][None,:]-last[qs,None])/86400;count=mask.sum(1);out[qs,0]=np.log1p(count)
   for z,row in enumerate(qs):
    good=mask[z]
    if not good.any():continue
    vals=g[good];out[row,1:6]=np.quantile(vals,[.1,.25,.5,.75,.9]);out[row,6:9]=[vals.mean(),vals.std(),np.mean(vals==0)];out[row,9:14]=np.quantile(norm[good],[.1,.25,.5,.75,.9])+pace[row]
   for kind,w in enumerate([np.exp(-D/180)*mask,np.exp(-D/730)*mask,np.exp(-.5*((prior[rr][None,:]-pace[qs,None])/1.5)**2)*mask]):
    mass=w.sum(1);out[qs,14+3*kind]=np.log1p(mass);out[qs,15+3*kind]=(w@g)/np.maximum(mass,1e-12);order=np.argsort(g);cw=np.cumsum(w[:,order],axis=1);where=np.argmax(cw>=.5*mass[:,None],axis=1);out[qs,16+3*kind]=np.where(mass>1e-8,g[order[where]],np.nan)
 assert not np.isinf(out).any();return out
def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','previous','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True);path=a.out/('full_peer.npz' if a.full else 'validation_peer.npz')
 if path.exists():return
 p,test,u,users,i,items,t,ratings,starts,early,v,e,rows=load_data(a.data_dir);allowed=np.ones(len(p),bool)
 if not a.full:allowed[v]=False;allowed[e]=False
 old=np.load(a.previous/('full.npz' if a.full else 'validation.npz'));core=old['X'];out=peers(i,t,u,allowed,i[rows],t[rows-1],u[rows],core[:,19]);result={'X':out}
 if a.full:
  qu=pd.Index(users).get_indexer(test.user);qi=pd.Index(items).get_indexer(test.item);result['Q']=peers(i,t,u,allowed,qi,t[starts[qu+1]-1],qu,old['Q'][:,19])
 np.savez_compressed(path,**result);print('Task6 peer features ready',a.full,out.shape,flush=True)
if __name__=='__main__':main()
''',
    'task6_pipeline.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1'
from task6_history_features import history_features
def load_data(data_dir):
    p=pd.read_csv(data_dir/'task_6_train.csv');test=pd.read_csv(data_dir/'task_6_test.csv')
    u,users=pd.factorize(p.user);i,items=pd.factorize(p.item)
    if np.any(np.diff(u)<0):raise ValueError('Expected contiguous user histories')
    t=p.timestamp.to_numpy(dtype=float);ratings=p.rating.to_numpy(dtype=float);starts=np.searchsorted(u,np.arange(len(users)+1));early=[];v=[];e=[]
    for user in range(len(users)):
        lo,hi=starts[user:user+2];n=hi-lo-1
        if n<3 or np.any(np.diff(t[lo:hi])<0):raise ValueError('Expected chronological histories with at least four interactions')
        v.append(hi-2);e.append(hi-1)
        for k in np.unique(np.linspace(1,n-3,min(5,max(0,n-3)),dtype=int)):early.append(lo+k+1)
    early=np.asarray(early);v=np.asarray(v);e=np.asarray(e);rows=np.r_[early,v,e]
    return p,test,u,users,i,items,t,ratings,starts,early,v,e,rows
def core_matrix(t,ratings,u,starts,rows):
    out=[]
    for idx,row in enumerate(rows):
        lo=starts[u[row]];out.append(history_features(t[lo:row],ratings[lo:row],int(row-lo-1)))
        if idx%100000==0:print('History feature rows',idx,'/',len(rows),flush=True)
    return np.asarray(out,dtype=np.float32)
def frame(core,book,p,rows=None,query=None):
    x=pd.DataFrame(np.column_stack([core,book]));x.columns=['f'+str(k) for k in range(x.shape[1])]
    source=p.iloc[rows] if query is None else query
    x['user']=source.user.to_numpy();x['next_item']=source.item.to_numpy()
    return x
''',
    # SHARED RUNTIME AND FIXED SETTINGS
    'tasks24_reader_neighbors.py': r'''from final_imports import *
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
def inputs(task,full):
 if task==5:
  from task5_improve import data,graph
  h,ctx,qu,qi,U,I,_=data(DATA,W/'task5/results',full);X=graph(h,U,I);return X,X.copy(),qu,qi,np.zeros(len(qu))
 df=pd.read_csv(DATA/('tasks_1_2_train.csv' if task==2 else 'tasks_3_4_train.csv'));u,users=pd.factorize(df.user);i,items=pd.factorize(df.item);t=df.timestamp.to_numpy(float)
 if full:
  ix=np.arange(len(df));test=pd.read_csv(DATA/f'task_{task}_test.csv');um=dict(zip(users,range(len(users))));im=dict(zip(items,range(len(items))));qu=np.tile(test.user.map(um).to_numpy(),2);qi=np.r_[test.item1.map(im),test.item2.map(im)];qt=np.tile(test.timestamp.to_numpy(float),2) if task==4 else np.zeros(len(qu))
 else:
  order=np.random.RandomState(42).permutation(len(df));a=int(.9*len(df));ix=order[:a];hold=order[a:];rng=np.random.RandomState(43)
  if task==2:
   neg=i[ix[rng.randint(len(ix),size=len(hold))]];same=neg==i[hold]
   while same.any():neg[same]=i[ix[rng.randint(len(ix),size=same.sum())]];same=neg==i[hold]
  else:
   neg=[]
   for p in i[hold]:
    j=i[ix[rng.randint(len(ix))]]
    while j==p:j=i[ix[rng.randint(len(ix))]]
    neg.append(j)
  qu=np.tile(u[hold],2);qi=np.r_[i[hold],neg];qt=np.tile(t[hold],2)
 shape=(len(users),len(items));X=csr_matrix((np.ones(len(ix),dtype='f'),(u[ix],i[ix])),shape=shape);X.data[:]=1
 T=csr_matrix((t[ix]/86400,(u[ix],i[ix])),shape=shape)
 return X,T,qu,qi,qt/86400
''',
}

if __name__ == "__main__":
    main()
