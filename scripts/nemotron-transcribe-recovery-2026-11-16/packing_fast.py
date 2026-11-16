"""Stop exact subset-sum search once the earliest full-capacity solution exists.

The upstream tie rule prefers earlier items. Once capacity is reachable, later
items cannot improve utilization and would be skipped by its reconstruction.
"""
def early_best_fit(lengths, capacity, max_items=None, fallback=None):
    if max_items is not None:
        return fallback(lengths,capacity,max_items)
    if capacity < 0 or any(x < 0 for x in lengths):
        raise ValueError('Lengths and capacity must be non-negative')
    if not lengths or capacity == 0:
        return []
    mask=(1 << (capacity+1))-1
    target_bit=1 << capacity
    prefixes=[1]
    for length in lengths:
        reachable=prefixes[-1]
        reachable=reachable | ((reachable << length) & mask)
        prefixes.append(reachable)
        if reachable & target_bit:
            break
    target=prefixes[-1].bit_length()-1
    selected=[]
    for index in range(len(prefixes)-2,-1,-1):
        if (prefixes[index] >> target) & 1:
            continue
        selected.append(index)
        target-=lengths[index]
    selected.reverse()
    return selected

def install():
    from packing_cache import install as cache_install
    cache_install()
    import nemo.collections.common.data.lhotse.packed_sequence_sampler as module
    if getattr(module,'_atc_early_best_fit_installed',False):
        return
    original=module._select_best_fit_indices
    module._select_best_fit_indices=lambda lengths,capacity,max_items=None: early_best_fit(lengths,capacity,max_items,original)
    module._atc_early_best_fit_installed=True
