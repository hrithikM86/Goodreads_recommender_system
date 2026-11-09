"""Train/evaluate the temporal Task 3 ensemble and optionally generate its CSV."""
import os,argparse,hashlib,json,subprocess,sys,shutil
from pathlib import Path
from importlib.metadata import version

def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=here.parent/'Assignment 1/assignment1_2026')
    parser.add_argument('--output-dir',type=Path,default=here/'artifacts'/'task3')
    parser.add_argument('--predict',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    for filename in ['tasks_3_4_train.csv','task_3_test.csv']:
        h.update((args.data_dir/filename).read_bytes())
    for path in sorted(here.glob('task3*.py')):h.update(path.read_bytes())
    h.update((here/'task3_configs.json').read_bytes())
    for package in ['numpy','pandas','scipy','scikit-surprise','catboost']:h.update((package+version(package)).encode())
    h.update(sys.version.encode());fingerprint=h.hexdigest()
    cache=args.output_dir/fingerprint[:16];cache.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(TASK3_DATA_DIR=str(args.data_dir.resolve()),TASK3_ARTIFACTS_DIR=str(cache.resolve()),OPENBLAS_NUM_THREADS='1',SURPRISE_DATA_FOLDER=str((cache/'surprise_data').resolve()))
    shutil.copyfile(here/'task3_configs.json',cache/'configs.json')
    steps=[('experiment','predictions.npz'),('grid','grid_predictions.npz'),('neighbors','neighbors_predictions.npz'),('kernels','kernel_predictions.npz'),('blend','blend_results.json'),('meta','meta_results.json'),('select','selection.json')]
    for script,output in steps:
        if not (cache/output).exists():
            subprocess.run([sys.executable,str(here/('task3_'+script+'.py'))],env=env,check=True)
    selection=json.loads((cache/'selection.json').read_text());print('Selected model:',selection,flush=True)
    if args.predict:
        if not (cache/'task_3_predictions.csv').exists():
            subprocess.run([sys.executable,str(here/'task3_predict.py')],env=env,check=True)
        shutil.copyfile(cache/'task_3_predictions.csv',args.output_dir/'task_3_predictions.csv')
        print('Saved:',args.output_dir/'task_3_predictions.csv',flush=True)
    (args.output_dir/'run.json').write_text(json.dumps({'fingerprint':fingerprint,'selection':selection,'cache':str(cache)},indent=2))

if __name__=='__main__':main()
