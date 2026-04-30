
import ast, collections, csv, datetime, functools, glob, gzip, hashlib, inspect, importlib, io, itertools, json, math, operator, os, os.path, pickle, random, re, requests, shutil, sqlite3, subprocess, string, sys, warnings, zipfile
import numpy as np, pandas as pd
import tqdm.contrib.concurrent

if importlib.util.find_spec('IPython') is not None:
    import IPython.display

GUARANTEED_RANDOM = 4 # https://xkcd.com/221

def uf(x):
    return '{:,}'.format(x)

def ul(x):
    return uf(len(x))

def printsrc(*args, **kwargs):
    """
        https://stackoverflow.com/questions/3056048/filename-and-line-number-of-python-script
        https://stackoverflow.com/questions/3711184/how-to-use-inspect-to-get-the-callers-info-from-callee-in-python
        https://github.com/snakemake/snakemake/blob/main/snakemake/exceptions.py#L17
    """
    #pprint(dir(inspect.currentframe().f_back))
    #pprint(dir(inspect.getframeinfo(inspect.currentframe().f_back)))
    frameinfo_ = inspect.getframeinfo(inspect.currentframe().f_back)
    #pprint(frameinfo_)
    #pprint(dir(frameinfo_))
    filename = frameinfo_.filename
    lineno = frameinfo_.lineno
    #lineno = workflow.linemaps[filename][ frameinfo_.lineno ]
    print(f'{os.path.basename(filename)}:{lineno}', *args, **kwargs)

def printlen(x, *args, **kwargs):
    name_ = inspect.stack()[1][3] #https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback
    if name_ != '<module>':
        print(f'{name_}:', uf(len(x)), *args, **kwargs)
    else:
        print(uf(len(x)), *args, **kwargs)

@functools.cache
def guess_prefix(euler_prefix, local_prefix):
    if os.path.isdir(euler_prefix):
        return euler_prefix
    elif os.path.isdir(local_prefix):
        return local_prefix
    else:
        assert False

def projectpath(path):
    dir_ = guess_prefix('/cluster/project/beltrao/jjaenes', '/Users/jjaenes/euler-home/project')
    return os.path.join(dir_, path)

def workpath(path):
    dir_ = guess_prefix('/cluster/work/beltrao/jjaenes', '/Users/jjaenes/euler-home/work')
    return os.path.join(dir_, path)

def flatten(l):
    return [item for sublist in l for item in sublist]

@functools.cache
def get_max_workers():
    try:
        ntasks = int(os.environ['SLURM_NTASKS'])
        source = 'SLURM_NTASKS'
    except:
        ntasks = int(subprocess.check_output(['nproc', '--all']))
        source = 'nproc --all'
    print(f'Using {ntasks} cores inferred from {source}')
    return ntasks

def parallel_map(fn, *iterables):
    """
    Example that returns multiple columns:
        interface_residues[['ifresid1', 'ifresid2']] = pd.DataFrame(mf.parallel_map(mf.structure.get_ifresid, interface_residues['path']))

    See also:
        https://tqdm.github.io/docs/contrib.concurrent/#process_map
    """
    return tqdm.contrib.concurrent.process_map(fn, *iterables, max_workers=get_max_workers(), chunksize=10)

def parallel_from_records(fn, *iterables, columns):
    parallel_map_ = parallel_map(fn, *iterables)
    # Maybe try & infer columns from iterables (fields argument?)
    return pd.DataFrame.from_records(flatten(parallel_map_), columns=columns)

def as_array(s):
    if type(s) is str:
        return np.array(json.loads(s))
    elif type(s) is list:
        return np.array(s)
    return s

def rm_prefix(frame, prefix):
    frame.columns = [ col.removeprefix(prefix) for col in frame.columns ]
    return frame

def rm_suffix(frame, prefix):
    frame.columns = [ col.removesuffix(prefix) for col in frame.columns ]
    return frame

def dispall(frame, max_rows=100, max_columns=None, max_colwidth=None):
    if max_rows is None:
        max_rows = len(frame)
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns, 'display.max_colwidth', max_colwidth):
        IPython.display.display(frame)

def sorted_pair_id(id1, id2):
    # sorted pair of uniprot_id-s to compare interacting pairs from different sources
    if id1 == id1 and id2 == id2:
        return '_'.join(sorted([id1, id2]))
    else:
        return ''
