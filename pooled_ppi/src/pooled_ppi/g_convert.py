
import functools, requests

from .core import *

def query(query=["CASQ2", "CASQ1", "GSTO1", "DMD", "GSTM2"], target='UNIPROTSWISSPROT_ACC', organism='hsapiens', numeric_namespace='ENTREZGENE_ACC'):
    """
    Query for HGNC gene names using g:convert (https://biit.cs.ut.ee/gprofiler/convert)
    """
    r = requests.post(url='https://biit.cs.ut.ee/gprofiler/api/convert/convert/', json=locals())
    df_ = pd.DataFrame(r.json()['result'])
    return df_

@functools.cache
def query_cached(*args, **kwargs):
    return query(*args, **kwargs)

def proxy_mapping(l, r, organism='hsapiens', target='ENSG', v=False):
    """
    Generate unique mappings over a proxy namespace
    """
    if v: print(f'{ul(l)}\t{ul(r)} {l.name}/{r.name}')

    l_dedup = l.drop_duplicates(keep='first')
    r_dedup = r.drop_duplicates(keep='first')
    if v: print(f'{ul(l_dedup)}\t{ul(r_dedup)} after dedup')

    l_query = query_cached(tuple(l_dedup.tolist()), organism=organism, target=target)[['incoming', 'converted']].rename({'incoming': l.name, 'converted': target}, axis=1)
    r_query = query_cached(tuple(r_dedup.tolist()), organism=organism, target=target)[['incoming', 'converted']].rename({'incoming': r.name, 'converted': target}, axis=1)
    if v: print(f'{ul(l_query)}\t{ul(r_query)} from query')

    l_query = l_query.query(f'{target} != "None"')
    r_query = r_query.query(f'{target} != "None"')
    if v: print(f'{ul(l_query)}\t{ul(r_query)} after removing empty mappings')

    merge = l_query.merge(r_query, on=target)
    if v: print(f'{ul(merge)} after merge')

    merge = merge[[l.name, r.name]].drop_duplicates(keep='first')
    if v: print(f'{ul(merge)} after shared dedup')

    merge = merge.drop_duplicates(subset=[l.name], keep=False)
    if v: print(f'{ul(merge)} after left dedup')

    merge = merge.drop_duplicates(subset=[r.name], keep=False)
    if v: print(f'{ul(merge)} after right dedup')

    return merge.reset_index(drop=True)

def proxy_merge(left, right, left_on, right_on, right_prefix=None, organism='hsapiens', target='ENSG', v=False):
    mapping = proxy_mapping(left[left_on], right[right_on], organism, target, v)
    merged = left.merge(mapping, on=left_on, how='left').merge(right, on=right_on, how='left').reset_index(drop=True)
    if right_prefix is not None:
        mapper_ = {col: col if col.startswith(right_prefix) else right_prefix + col for col in right.columns}
        merged = merged.rename(mapper_, axis=1)
    return merged
