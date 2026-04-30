
import itertools
import pandas as pd
import Bio, Bio.SeqIO

def read(path, stop=None):
    columns = ['id', 'seq']
    return pd.DataFrame.from_records([ (r.id, str(r.seq)) for r in itertools.islice(Bio.SeqIO.parse(path, 'fasta'), stop) ], columns=columns)
