
import functools
import pandas as pd
from cached_path import cached_path
from .core import *

@functools.cache
def read(version='5.0.255'):
    path = cached_path(f'https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{version}/BIOGRID-ALL-{version}.tab3.zip')
    with zipfile.ZipFile(path, 'r') as zf:
        df_ = pd.read_csv(zf.open(f'BIOGRID-ALL-{version}.tab3.txt'), sep='\t',
            dtype={
                'Entrez Gene Interactor A': str,
                'Entrez Gene Interactor B': str,
            },
            na_values={
                'Score': '-',
            },
            #nrows=1000
        )
    printlen(df_, 'raw records')

    q_ = '(`Organism Name Interactor A` == "Saccharomyces cerevisiae (S288c)") & (`Organism Name Interactor B` == "Saccharomyces cerevisiae (S288c)")'
    df_ = df_.query(q_).reset_index(drop=True)
    printlen(df_, 'in yeast')
    return df_.rename({df_.columns[0]: df_.columns[0].lstrip('#')}, axis=1)
