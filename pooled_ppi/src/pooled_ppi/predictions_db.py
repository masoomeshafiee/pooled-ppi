
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

def download_data():
    def run_(cmd):
      print(f'Run: `{cmd}`')
      subprocess.run(args=cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    run_('apt-get update --quiet')
    run_('apt-get install --yes --quiet skopeo umoci')
    run_('rm -rf /data')
    run_('mkdir -p /data')
    run_('skopeo copy docker://jurgjn/pooled-ppi-yeast:latest oci:/tmp/_oci:latest')
    run_('umoci unpack --image /tmp/_oci /tmp/_unpack')
    run_('mv /tmp/_unpack/rootfs/data/* /data')

class PooledPredictionsDb:
    def __init__(self, path='/data'):
        self.path = path
        self.pairs = pd.read_parquet(os.path.join(self.path, 'summary_pairs.parquet'))
        self.pairs['uniprot_id1'] = self.pairs['af3_id1'].str.upper()
        self.pairs['uniprot_id2'] = self.pairs['af3_id2'].str.upper()
        print(f'{ul(self.pairs)} pairs / {uf(self.pairs["name"].nunique())} pools')

    def bait_prey(self):
        pairs_fwd = self.pairs.copy()
        pairs_fwd['bait_id'] = pairs_fwd['uniprot_id1']
        pairs_fwd['prey_id'] = pairs_fwd['uniprot_id2']

        pairs_rev = self.pairs.copy()
        pairs_rev['bait_id'] = pairs_rev['uniprot_id2']
        pairs_rev['prey_id'] = pairs_rev['uniprot_id1']

        bait_prey = pd.concat([pairs_fwd, pairs_rev], axis=0)
        return bait_prey

    def save_ids(self, ids, file):
        parser = Bio.PDB.PDBParser(QUIET=True)
        struct0 = None
        with foldcomp.open(os.path.join(self.path, 'predictions-db/predictions-db'), ids=ids) as db:
            for index, ((name, pdb), chain_id) in enumerate(itertools.islice(zip(db, af3io.input.enumerate_chains()), None)):
                struct = parser.get_structure(index, io.StringIO(pdb))
                if index == 0:
                    struct0 = struct
                else:
                    chain0 = next(struct[0].get_chains())
                    chain0.id = chain_id
                    struct0[0].add(chain0)
                
        pdbio = Bio.PDB.PDBIO()
        pdbio.set_structure(struct0)
        pdbio.save(file)
