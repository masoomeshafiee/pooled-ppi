
import argparse, collections, collections.abc, copy, functools, glob, hashlib, itertools, gzip, json, os, os.path, re, string, subprocess, sys
import io, os, os.path, zipfile, warnings
from pathlib import Path
from pprint import pprint
import numpy as np, pandas as pd
import Bio, Bio.PDB, Bio.PDB.mmcifio, foldcomp
import af3io, foldcomp, pooled_ppi
import functools, glob, gzip, itertools, io, json, math, os, re, zipfile
from pprint import pprint
from pathlib import Path

import pandas as pd

import glob, gzip, itertools, io, json, math, os, re, zipfile
from pprint import pprint
from pathlib import Path

import numpy as np, pandas as pd, snakemake.io

import af3io

from .core import *

'''
class PooledPredictions:
    def __init__(self, path):
        # find/assign name (from path)
        self.path = Path(path)
        ids, = snakemake.io.glob_wildcards(os.path.join(path, '{id}.zip'))
        self.predictions = pd.DataFrame({'id': ids})
        self.predictions['zip'] = self.predictions['id'].map(lambda id: os.path.join(path, f'{id}.zip'))
        self.summary_confidences = pd.concat(parallel_map(af3io.predictions.read_summary_confidences, self.predictions['zip']), axis=0).reset_index(drop=True)
'''

def glob_alphafold3_input_jsons(path):
    predictions = pd.DataFrame({'path': list(path.resolve().glob(f'**/alphafold3_jsons/*.json'))}).sort_values(['path']).reset_index(drop=True)
    predictions['name'] = predictions['path'].map(lambda path: str(Path(path).stem))
    predictions['batch_id'] = predictions['path'].map(lambda path_: str(Path(path_).parent.parent.relative_to(path.resolve())))
    printlen(predictions, 'input JSONs total')
    return predictions[['name', 'batch_id', 'path']]

def glob_alphafold3_predictions(path):
    predictions = pd.DataFrame({'path': list(path.resolve().glob(f'**/alphafold3_predictions/*.zip'))}).sort_values(['path']).reset_index(drop=True)
    predictions['name'] = predictions['path'].map(lambda path: str(Path(path).stem))
    predictions['batch_id'] = predictions['path'].map(lambda path_: str(Path(path_).parent.parent.relative_to(path.resolve())))
    printlen(predictions, 'pooled predictions from', path.resolve())
    return predictions[['name', 'batch_id', 'path']]

class PooledPredictions:
    def __init__(self, pools, sizes):
        self.sizes = sizes
        self.pools = pools
        if not('pool_hash' in self.pools.columns):
            self.pools['pool_hash'] = self.pools['pool_id'].map(lambda pool_id: hashlib.sha1(pool_id.encode()).hexdigest())
    
    def pool_coverage(self):
        def generate_interactions(ids):
            # Generate all possible interactions between ids
            return set(itertools.combinations(sorted(ids), 2))

        printlen(self.sizes, 'proteins')
        printlen(self.pools, 'pools')

        all = set(generate_interactions(self.sizes.af3_id))
        printlen(all, 'interactions possible based on the list of proteins')

        gen = set()
        for i, r in self.pools.iterrows():
            pool_interactions = generate_interactions(r.pool_id.split('_'))
            gen |= pool_interactions

        printlen(gen, 'interactions in pools')

        printlen(all - gen, 'interactions missing')
        printlen(gen - all, 'interactions extra')

    def glob_alphafold3_status(self):
        input_jsons = glob_alphafold3_input_jsons(Path().resolve())
        printlen(input_jsons, 'input JSONs total')
        self.input_jsons = input_jsons.merge(self.pools, left_on='name', right_on='pool_hash')[['batch_id', 'name', 'pool_id', 'pool_size', 'path']]
        printlen(self.input_jsons, 'pools with an input JSON')

        predictions = glob_alphafold3_predictions(Path().resolve())
        printlen(predictions, 'predictions total')
        self.predictions = predictions.merge(self.pools, left_on='name', right_on='pool_hash')[['batch_id', 'name', 'pool_id', 'pool_size', 'path']]
        printlen(self.predictions, 'pools with at least one prediction')

        has_input_json_ = set(self.input_jsons['name'])
        has_prediction_ = set(self.predictions['name'])
        self.pools['is_queued'] = self.pools['pool_hash'].isin(has_input_json_)
        self.pools['is_predicted'] = self.pools['pool_hash'].isin(has_prediction_)
        self.pools['is_missing'] = (~self.pools['is_predicted']) & (~self.pools['is_queued'])

        printlen(self.pools, 'pools total')
        printlen(self.pools.query('is_predicted'), 'pools with finished predictions')
        printlen(self.pools.query('~is_predicted & is_queued'), 'pools in the queue')
        printlen(self.pools.query('is_missing'), 'pools missing')

#def read_summary_confidences(path):
#    pp = PooledPredictions(path)
#    return pp.summary_confidences

def chain_pair_iptm_triu(s):
    if type(s) is str:
        arr = np.array(json.loads(s))
    elif type(s) is list:
        arr = np.array(s)
    else:
        arr = s
    tri = np.triu_indices_from(arr, k=1)
    return arr[tri]

def agg_chain_pair_iptms(frame):
    """Aggregate pairwise iptms (chain_pair_iptm) using different approaches:
        chain_pair_iptm_best = best-ranked sample
        chain_pair_iptm_max = max across samples
        chain_pair_iptm_mean = mean across samples
    """
    # np.array([np.array([[1,2], [3,4]]), np.array([[4,3], [2,1]]), np.array([[1,1], [1,1]])]).max(axis=0)
    chain_pair_iptm_best = np.stack(as_array(frame.sort_values('ranking_score', ascending=False).head(1)['chain_pair_iptm'].squeeze()))

    chain_pair_iptm_list = list(frame['chain_pair_iptm'].map(as_array))
    #chain_pair_iptm_max = np.maximum.reduce(chain_pair_iptm_list)

    chain_pair_iptm_array = np.array(chain_pair_iptm_list)
    #chain_pair_iptm_max = chain_pair_iptm_array.max(axis=0)
    chain_pair_iptm_mean = np.stack(chain_pair_iptm_array.mean(axis=0))

    #return pd.Series([chain_pair_iptm_best, chain_pair_iptm_max, chain_pair_iptm_mean], index=['chain_pair_iptm_best', 'chain_pair_iptm_max', 'chain_pair_iptm_mean'])
    return pd.Series([chain_pair_iptm_best, chain_pair_iptm_mean], index=['chain_pair_iptm_best', 'chain_pair_iptm_mean'])

def explode_iptms(pools, columns_keep=[], columns_triu=['chain_pair_iptm']):
    """
    pools_id = split by '_' & generate pairwise combinations
    columns_keep = keep in exploded result, do not change
    columns_triu = select upper triu
    """
    def interactions_(s):
        l_ = list(itertools.combinations(s.split('_'), 2))
        #random.shuffle(l_)
        return l_

    def interactions_chains_(s):
        chains_ = [ chain_id for (af3_id, chain_id) in list(zip(s.split('_'), af3io.input.enumerate_chains())) ]
        l_ = list(itertools.combinations(chains_, 2))
        return l_

    pairs = pd.DataFrame({
        'pair_ids': pools['pool_id'].map(interactions_),
        'chain_ids': pools['pool_id'].map(interactions_chains_),
    })
    for column in columns_keep:
        pairs[column] = pools[column]

    for column in columns_triu:
        pairs[column] = pools[column].map(chain_pair_iptm_triu)

    return pairs.explode(['pair_ids', 'chain_ids', ] + columns_triu).reset_index(drop=True)
