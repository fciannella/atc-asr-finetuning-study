"""Cache immutable per-example lengths inside each exact packing buffer.

No sampling, ordering, constraints, checkpoint schema or RNG changes. Retain
object references in the bounded cache to prevent Python object-ID reuse.
Only valid for this frozen pipeline's immutable, pretokenized examples.
"""
def install():
    from nemo.collections.common.data.lhotse.packed_sequence_sampler import ExactTokenBatcher
    cls = ExactTokenBatcher
    if getattr(cls, '_atc_length_cache_installed', False):
        return
    original_raw = cls._measure_integer_length
    original_budget = cls._measure_budget_length
    original_collect = cls._collect_batch

    def entry(self, item):
        example = self._measured_example(item)
        if not hasattr(self, '_atc_length_cache'):
            self._atc_length_cache = {}
            self._atc_cache_calls = 0
        key = id(example)
        if key not in self._atc_length_cache:
            self._atc_length_cache[key] = [example, None, None]
        value = self._atc_length_cache[key]
        assert value[0] is example
        return value

    def raw(self, item):
        value = entry(self, item)
        if value[1] is None:
            value[1] = original_raw(self, item)
        return value[1]

    def budget(self, item):
        value = entry(self, item)
        if value[2] is None:
            value[2] = original_budget(self, item)
        return value[2]

    def collect(self):
        result = original_collect(self)
        self._atc_cache_calls += 1
        if self._atc_cache_calls % 32 == 0:
            retained = {id(self._measured_example(x)) for x in self.reuse_cuts_buffer}
            self._atc_length_cache = {k: v for k, v in self._atc_length_cache.items() if k in retained}
        return result

    cls._measure_integer_length = raw
    cls._measure_budget_length = budget
    cls._collect_batch = collect
    cls._atc_length_cache_installed = True
