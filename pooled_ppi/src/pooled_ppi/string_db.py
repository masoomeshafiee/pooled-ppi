
import pandas as pd

from cached_path import cached_path

def read(data_id='protein.links', organism_id=4932, cols_rm_organism={}, string_db_version='v12.0'):
    """
    https://string-db.org/cgi/download
    https://stringdb-downloads.org/download/protein.links.v12.0/4932.protein.links.v12.0.txt.gz
    """
    sep = r'\s+'
    if data_id in {'protein.aliases', 'protein.info'}:
        sep = '\t'

    path = cached_path(f'https://stringdb-downloads.org/download/{data_id}.{string_db_version}/{organism_id}.{data_id}.{string_db_version}.txt.gz')
    frame = pd.read_csv(path, compression='gzip', sep=sep)
    frame.rename({frame.columns[0]: frame.columns[0].lstrip('#')}, axis=1, inplace=True)

    organism_prefix = f'{organism_id}.'
    for col in cols_rm_organism:
        frame[col] = frame[col].str.removeprefix(organism_prefix)
    return frame
