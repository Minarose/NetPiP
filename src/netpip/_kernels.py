"""Compiled kernels for PiP sampling.

For each random removal order, the percolation event is the first removal step
k in 1..N-1 at which the second-largest connected component (SLCC) of the
surviving graph reaches its maximum size. If the SLCC never exceeds one node,
the event is N. Every node removed up to and including step k participates.

The SLCC trace is computed exactly by adding nodes back in reverse removal
order with a union-find structure, which reconstructs every surviving graph in
O(N + E alpha(N)) time per order. Numba compiles these functions when it is
installed; otherwise the same code runs in pure Python (slow, but identical).
"""
import numpy as np

try:  # pragma: no cover - exercised implicitly
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


@njit(cache=True)
def _root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def slcc_trace(order, indptr, indices):
    """SLCC size among surviving nodes after k = 1..N-1 removals (entry k-1).

    Removed nodes are absent, so the SLCC is 0 when only one component remains.
    """
    n = len(order)
    parent = np.arange(n)
    sizes = np.ones(n, dtype=np.int64)
    active = np.zeros(n, dtype=np.bool_)
    frequency = np.zeros(n + 1, dtype=np.int64)  # number of components of each size
    second = np.zeros(max(n - 1, 0), dtype=np.int64)
    largest = 0
    components = 0
    for step in range(n - 1, 0, -1):
        node = order[step]
        active[node] = True
        frequency[1] += 1
        components += 1
        if largest < 1:
            largest = 1
        for j in range(indptr[node], indptr[node + 1]):
            other = indices[j]
            if not active[other]:
                continue
            r = _root(parent, node)
            t = _root(parent, other)
            if r == t:
                continue
            if sizes[r] < sizes[t]:
                r, t = t, r
            frequency[sizes[r]] -= 1
            frequency[sizes[t]] -= 1
            parent[t] = r
            sizes[r] += sizes[t]
            frequency[sizes[r]] += 1
            components -= 1
            if sizes[r] > largest:
                largest = sizes[r]
        # State now = graph after removing order[0..step-1], i.e. `step` removals.
        if components > 1:
            if frequency[largest] >= 2:
                second[step - 1] = largest
            else:
                s = largest - 1
                while s > 0 and frequency[s] == 0:
                    s -= 1
                second[step - 1] = s
    return second


@njit(cache=True)
def _highest_bit(word):
    pos = 0
    if word >> np.uint64(32):
        word >>= np.uint64(32)
        pos += 32
    if word >> np.uint64(16):
        word >>= np.uint64(16)
        pos += 16
    if word >> np.uint64(8):
        word >>= np.uint64(8)
        pos += 8
    if word >> np.uint64(4):
        word >>= np.uint64(4)
        pos += 4
    if word >> np.uint64(2):
        word >>= np.uint64(2)
        pos += 2
    if word >> np.uint64(1):
        pos += 1
    return pos


@njit(cache=True)
def event_step(order, indptr, indices):
    """Percolation event (1-based removal step) for one removal order.

    Same definition as `slcc_trace` followed by the first strict maximum, but the
    second-largest size is found with a bitset over occupied component sizes,
    which keeps large networks fast (this is the kernel used for the paper).
    """
    n = len(order)
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    active = np.zeros(n, dtype=np.bool_)
    freq = np.zeros(n + 1, dtype=np.int64)
    bits = np.zeros((n + 63) // 64, dtype=np.uint64)
    one = np.uint64(1)
    largest = 0
    components = 0
    best = 1
    event = n
    for k in range(n - 1, 0, -1):
        node = order[k]
        active[node] = True
        freq[1] += 1
        bits[0] |= one
        components += 1
        if largest < 1:
            largest = 1
        for j in range(indptr[node], indptr[node + 1]):
            v = indices[j]
            if not active[v]:
                continue
            a = _root(parent, node)
            b = _root(parent, v)
            if a == b:
                continue
            if size[a] < size[b]:
                a, b = b, a
            sa, sb = size[a], size[b]
            freq[sa] -= 1
            if freq[sa] == 0:
                bits[(sa - 1) // 64] &= ~(one << np.uint64((sa - 1) % 64))
            freq[sb] -= 1
            if freq[sb] == 0:
                bits[(sb - 1) // 64] &= ~(one << np.uint64((sb - 1) % 64))
            parent[b] = a
            size[a] = sa + sb
            total = sa + sb
            freq[total] += 1
            bits[(total - 1) // 64] |= one << np.uint64((total - 1) % 64)
            components -= 1
            if total > largest:
                largest = total
        if components > 1:  # state = graph after k removals
            second = largest
            if freq[largest] == 1:
                for wi in range((largest - 1) // 64, -1, -1):
                    word = bits[wi]
                    if wi == (largest - 1) // 64:
                        word &= ~(one << np.uint64((largest - 1) % 64))
                    if word:
                        second = wi * 64 + _highest_bit(word) + 1
                        break
            # scanning k downwards, ">=" keeps the earliest step of the maximum
            if second > 1 and second >= best:
                best = second
                event = k
    return event


@njit(parallel=True, cache=True)
def batch_event_steps(orders, indptr, indices):
    out = np.empty(orders.shape[0], dtype=np.int64)
    for i in prange(orders.shape[0]):
        out[i] = event_step(orders[i], indptr, indices)
    return out


@njit(cache=True)
def accumulate(orders, steps, counts, hits):
    """Add a batch of events: counts[k-1] += 1 and hits[k-1, node] += 1 for each participant."""
    for i in range(orders.shape[0]):
        k = steps[i]
        counts[k - 1] += 1
        for j in range(k):
            hits[k - 1, orders[i, j]] += 1
