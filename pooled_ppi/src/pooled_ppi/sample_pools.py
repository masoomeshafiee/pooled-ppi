#!/usr/bin/env python
"""
Generate pools as in https://doi.org/10.1101/2025.07.01.662654 except that interactions are weighted by the product of the protein sizes
Optimised for maximum performance using incremental updates and Numba JIT.
"""

import argparse, itertools, sys, numpy as np, pandas as pd, tqdm, numba

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@numba.njit(parallel=True, nogil=True)
def calculate_uncovered_weights(sizes, covered):
    """Calculate the sum of uncovered interaction weights for each protein."""
    n = sizes.shape[0]
    weights = np.zeros(n, dtype=np.float64)
    for i in numba.prange(n):
        w = 0.0
        for j in range(n):
            if i != j and not covered[i, j]:
                w += sizes[i] * sizes[j]
        weights[i] = w
    return weights

@numba.njit(nogil=True)
def get_initial_large_pools_numba(sizes, covered, max_size):
    """Find pairs of proteins that exceed max_size and are not yet covered."""
    n = sizes.shape[0]
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if sizes[i] + sizes[j] > max_size and not covered[i, j]:
                res.append((i, j))
    return res

@numba.njit(parallel=True, nogil=True)
def find_best_i_parallel(pool_mask, sizes, pool_C, current_pool_size, max_size):
    n = sizes.shape[0]
    scores = np.full(n, -1e18, dtype=np.float64)
    for i in numba.prange(n):
        if not pool_mask[i] and current_pool_size + sizes[i] <= max_size:
            # Score: newly covered interaction weight per unit size of protein i.
            # Newly covered interaction weight for adding i to current pool is 2 * sizes[i] * (current_pool_size - pool_C[i])
            # Dividing by sizes[i] results in (current_pool_size - pool_C[i])
            scores[i] = (current_pool_size - pool_C[i])

    best_i = np.argmax(scores)

    if scores[best_i] <= 0:
        return -1
    return best_i

@numba.njit(parallel=True, nogil=True)
def update_pool_C_parallel(pool_C, covered, best_i, size_best_i):
    n = pool_C.shape[0]
    for i in numba.prange(n):
        if covered[i, best_i]:
            pool_C[i] += size_best_i

@numba.njit(nogil=True)
def update_coverage_numba(pool_ix, covered, sizes, uncovered_weight_per_protein):
    """Mark all interactions within the finished pool as covered."""
    newly_covered_count = 0
    for i_idx in range(len(pool_ix)):
        p1 = pool_ix[i_idx]
        for j_idx in range(i_idx + 1, len(pool_ix)):
            p2 = pool_ix[j_idx]
            if not covered[p1, p2]:
                covered[p1, p2] = 1
                covered[p2, p1] = 1
                weight = sizes[p1] * sizes[p2]
                uncovered_weight_per_protein[p1] -= weight
                uncovered_weight_per_protein[p2] -= weight
                newly_covered_count += 1
    return newly_covered_count

def generate_pools(sizes, max_size=5120, skip_pairs=[], rng = np.random.default_rng(seed=4)):
    n = len(sizes)
    covered = np.zeros((n, n), dtype=np.uint8)
    np.fill_diagonal(covered, 1)

    for i, j in skip_pairs:
        if i < n and j < n:
            covered[i, j] = 1
            covered[j, i] = 1

    uncovered_weight_per_protein = calculate_uncovered_weights(sizes, covered)

    # Initial large pairs
    large_pairs = get_initial_large_pools_numba(sizes, covered, max_size)
    for i, j in large_pairs:
        yield(set([i, j]), sizes[i] + sizes[j])
        covered[i, j] = 1
        covered[j, i] = 1
        weight = sizes[i] * sizes[j]
        uncovered_weight_per_protein[i] -= weight
        uncovered_weight_per_protein[j] -= weight

    total_interactions = n * (n - 1) // 2
    covered_interactions = int(np.sum(covered) // 2 - n // 2)

    pbar = tqdm.tqdm(total=total_interactions)
    pbar.update(covered_interactions)

    pool_mask = np.zeros(n, dtype=np.bool_)
    pool_C = np.zeros(n, dtype=np.float64)

    while covered_interactions < total_interactions:
        avail = np.where(uncovered_weight_per_protein > 1e-6)[0]
        if len(avail) == 0:
            break

        # Greedy: pick protein with most uncovered interaction weight
        avail_choice = avail[np.argmax(uncovered_weight_per_protein[avail])]

        pool_mask.fill(False)
        pool_mask[avail_choice] = True

        # Initialise pool_C for the first protein
        pool_C.fill(0)
        update_pool_C_parallel(pool_C, covered, avail_choice, sizes[avail_choice])

        current_pool_size = sizes[avail_choice]

        while True:
            best_i = find_best_i_parallel(
                pool_mask, sizes, pool_C, current_pool_size, max_size
            )
            if best_i == -1:
                break

            current_pool_size += sizes[best_i]
            pool_mask[best_i] = True
            update_pool_C_parallel(pool_C, covered, best_i, sizes[best_i])

        pool_ix = np.where(pool_mask)[0]
        yield (set(pool_ix.tolist()), float(current_pool_size))

        newly_covered = update_coverage_numba(pool_ix, covered, sizes, uncovered_weight_per_protein)
        covered_interactions += newly_covered
        pbar.update(newly_covered)

    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Sample random pools minimising overlap")
    parser.add_argument("--init_pools", "-p", help="Pools to skip")
    parser.add_argument("--max_pool_size", "-s", help="Maximum size for pool", default=5120, type=int)
    parser.add_argument("--max_pools", "-n", help="Maximum number of pools to sample", type=int)
    args = parser.parse_args()

    # Read input protein information from stdin
    proteins = pd.read_csv(sys.stdin, sep=r'\s+', names=['seq_id', 'seq_len'])
    protein_ids = proteins.iloc[:, 0].tolist()
    protein_id_to_ix = {id_: i for i, id_ in enumerate(protein_ids)}

    skip_pairs = []
    if args.init_pools is not None:
        initial_pools = pd.read_csv(args.init_pools, sep=r'\s+')
        for pool_id in initial_pools['pool_id']:
            ids = pool_id.split('_')
            ixs = [protein_id_to_ix[id_] for id_ in ids if id_ in protein_id_to_ix]
            if len(ixs) >= 2:
                skip_pairs.extend(itertools.combinations(sorted(ixs), 2))

    eprint(numba.get_num_threads(), 'threads available for numba')
    eprint(len(protein_ids), 'proteins in input')

    sizes = proteins['seq_len'].values.astype(np.float64)

    pools_data_id = []
    pools_data_size = []
    pools_data_ixs = []

    # Generate pools using the optimized algorithm
    pool_gen = generate_pools(sizes, max_size=args.max_pool_size, skip_pairs=skip_pairs)
    if args.max_pools is not None:
        pool_gen = itertools.islice(pool_gen, args.max_pools)

    for pool_ixs, pool_size in pool_gen:
        pool_ids_subset = sorted([protein_ids[ix] for ix in pool_ixs])
        pools_data_id.append('_'.join(pool_ids_subset))
        pools_data_size.append(pool_size)
        pools_data_ixs.append(list(pool_ixs))

    if not pools_data_id:
        eprint('No pools generated')
        return

    eprint(len(pools_data_id), 'pools generated')

    # Redundancy and completeness sanity checks (optimised)
    all_interactions_count = len(sizes) * (len(sizes) - 1) // 2
    all_sum = (np.sum(sizes)**2 - np.sum(sizes**2)) / 2

    # Check if all possible interactions are covered across all pools
    unique_pairs_count = 0
    covered_check = np.zeros((len(sizes), len(sizes)), dtype=np.uint8)
    actual_gen_sum = 0
    for pool_ixs in pools_data_ixs:
        for i, p1 in enumerate(pool_ixs):
            for p2 in pool_ixs[i+1:]:
                if p1 < p2:
                    low, high = p1, p2
                else:
                    low, high = p2, p1
                if not covered_check[low, high]:
                    covered_check[low, high] = 1
                    unique_pairs_count += 1
                actual_gen_sum += sizes[p1] * sizes[p2]

    eprint(all_interactions_count, 'interactions expected')
    eprint(unique_pairs_count, 'interactions across all pools generated')
    eprint(unique_pairs_count == all_interactions_count, 'pools include all possible interactions')
    eprint(actual_gen_sum / all_sum, 'length-weighted redundancy factor across all pools')

    # Output generated pools to stdout
    pd.DataFrame({'pool_id': pools_data_id, 'pool_size': pools_data_size}).to_csv(sys.stdout, sep='\t', index=False)

if __name__ == "__main__":
    main()
