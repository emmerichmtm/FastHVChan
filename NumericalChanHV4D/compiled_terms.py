"""Whole numerical contractions in Numba, with the original term algebra.

A term is (coefficient, unary arrays, pairwise cutoff arrays). Dictionaries
are shallow-copied; arrays are immutable and every update allocates a new one.
The first-winner elimination rule is the same as in the Python reference.
Hashing accelerates merging; complete bitwise comparison resolves collisions.
No fastmath, dense pair tables, changed compression policy, or object mode.
"""
from collections import namedtuple
import numpy as np
from numba import njit, types
from numba.typed import Dict, List
from NumericalChanHVND.numerical_chan_numba import (
    direction, nonzero, normalize, prefix_sum, search_array,
)

PackedTerm = namedtuple('PackedTerm', 'coef unary edges')
Bound = namedtuple('Bound', 'axis values')  # axis=-1 denotes a scalar constant
FLOATS, INTS = types.float64[::1], types.int64[::1]
EDGE_KEY = types.UniTuple(types.int64, 4)
UNARY = types.DictType(types.int64, FLOATS)
EDGES = types.DictType(EDGE_KEY, INTS)
TERM = types.NamedTuple((types.float64, UNARY, EDGES), PackedTerm)
BOUND = types.NamedTuple((types.int64, INTS), Bound)
MASK = types.Tuple((types.int64, types.int64, INTS))
COUNTERS = ('raw_terms', 'term_eliminations', 'peak_terms',
            'max_lower_candidates', 'max_upper_candidates', 'compressions')


@njit(cache=True)
def copy_term(term):
    return PackedTerm(term.coef, term.unary.copy(), term.edges.copy())


@njit(cache=True)
def alive(term):
    if term.coef == 0.:
        return False
    for values in term.unary.values():
        if not nonzero(values):
            return False
    return True


@njit(cache=True)
def multiply(term, axis, values):
    term.unary[axis] = term.unary[axis] * values
    return alive(term)


@njit(cache=True)
def add_edge(term, source, target, upper, values):
    size = len(term.unary[target])
    low, high = (-1, size - 1) if upper else (0, size)
    values = np.clip(values, low, high)
    key = (source, target, int(upper), direction(values))
    if key in term.edges:
        old = term.edges[key]
        del term.edges[key]
        values = np.minimum(values, old) if upper else np.maximum(values, old)
    if values.min() == values.max():
        indices = np.arange(size)
        mask = indices <= values[0] if upper else indices >= values[0]
        return multiply(term, target, mask)
    term.edges[key] = values
    return multiply(term, source, values >= 0 if upper else values < size)


@njit(cache=True)
def impose(term, first, second, strict=False):
    a, av = first
    b, bv = second
    if a == -1 and b == -1:
        valid = av[0] < bv[0] if strict else av[0] <= bv[0]
        return valid and alive(term)
    if a == b or a == -1 or b == -1:
        axis = b if a == -1 else a
        return multiply(term, axis, av < bv if strict else av <= bv)
    if a < b:
        if direction(bv) == 1:
            return add_edge(term, a, b, False, search_array(bv, av, strict))
        return add_edge(term, a, b, True, search_array(-bv, av, not strict, True, -1))
    if direction(av) == 1:
        return add_edge(term, b, a, True, search_array(av, bv, not strict, False, -1))
    return add_edge(term, b, a, False, search_array(-av, bv, strict, True))


@njit(cache=True)
def unique_bounds(bounds):
    result = List.empty_list(BOUND)
    for bound in bounds:
        duplicate = False
        for kept in result:
            if bound.axis == kept.axis and np.array_equal(bound.values, kept.values):
                duplicate = True
                break
        if not duplicate:
            result.append(bound)
    return result


@njit(cache=True)
def state_hash(term):
    # Canonical order makes the key independent of dictionary insertion order.
    code, prime = np.uint64(14695981039346656037), np.uint64(1099511628211)
    for axis in sorted(term.unary.keys()):
        values = term.unary[axis].view(np.uint64)
        code = (code ^ np.uint64(axis)) * prime
        code = (code ^ np.uint64(len(values))) * prime
        for value in values:
            code = (code ^ value) * prime
    for key in sorted(term.edges.keys()):
        for part in key:
            code = (code ^ np.uint64(part)) * prime
        values = term.edges[key]
        code = (code ^ np.uint64(len(values))) * prime
        for value in values:
            code = (code ^ np.uint64(value)) * prime
    return code


@njit(cache=True)
def same_state(first, second):
    if len(first.unary) != len(second.unary) or len(first.edges) != len(second.edges):
        return False
    for axis, values in first.unary.items():
        if axis not in second.unary:
            return False
        if not np.array_equal(values.view(np.uint64), second.unary[axis].view(np.uint64)):
            return False
    for key, values in first.edges.items():
        if key not in second.edges or not np.array_equal(values, second.edges[key]):
            return False
    return True


@njit(cache=True)
def emit(term, kept, heads, links, stats):
    """Normalize and merge one term, checking collisions instead of trusting hashes."""
    stats[0] += 1
    if not alive(term):
        return
    coefficient = term.coef
    for axis in term.unary:
        factor, values = normalize(term.unary[axis])
        term.unary[axis] = values
        coefficient *= factor
    term = PackedTerm(coefficient, term.unary, term.edges)
    code = state_hash(term)
    head = heads[code] if code in heads else -1
    index = head
    # Bound collision work. A missed duplicate stays as a separate, valid term.
    for _ in range(16):
        if index == -1:
            break
        old = kept[index]
        if same_state(old, term):
            kept[index] = PackedTerm(old.coef + term.coef, old.unary, old.edges)
            return
        index = links[index]
    heads[code] = len(kept)
    links.append(head)
    kept.append(term)


@njit(cache=True)
def eliminate(terms, axis, stats):
    """Sum one variable by partitioning into first lower/upper winners."""
    kept = List.empty_list(TERM)
    heads = Dict.empty(types.uint64, types.int64)
    links = List.empty_list(types.int64)
    for incoming in terms:
        stats[1] += 1
        term = copy_term(incoming)
        weights = term.unary[axis]
        del term.unary[axis]
        prefix = prefix_sum(weights)
        lowers, uppers = List.empty_list(BOUND), List.empty_list(BOUND)
        lowers.append(Bound(-1, np.array([0], dtype=np.int64)))
        uppers.append(Bound(-1, np.array([len(weights) - 1], dtype=np.int64)))
        for key in list(term.edges.keys()):
            i, j, upper, orient = key
            if axis != i and axis != j:
                continue
            values = term.edges[key]
            del term.edges[key]
            if axis == j:
                (uppers if upper else lowers).append(Bound(i, values))
            else:
                targets = np.arange(len(term.unary[j]), dtype=np.int64)
                if orient == 1:
                    cut = (search_array(values, targets, False) if upper else
                           search_array(values, targets, True, False, -1))
                    (lowers if upper else uppers).append(Bound(j, cut))
                else:
                    cut = (search_array(-values, targets, True, True, -1) if upper else
                           search_array(-values, targets, False, True))
                    (uppers if upper else lowers).append(Bound(j, cut))
        lowers, uppers = unique_bounds(lowers), unique_bounds(uppers)
        stats[3], stats[4] = max(stats[3], len(lowers)), max(stats[4], len(uppers))
        for u, upper in enumerate(uppers):
            top = copy_term(term)
            valid = True
            for q, other in enumerate(uppers):
                if q != u and not impose(top, upper, other, q < u):
                    valid = False
                    break
            if not valid:
                continue
            for ell, lower in enumerate(lowers):
                winner = copy_term(top)
                valid = True
                for q, other in enumerate(lowers):
                    if q != ell and not impose(winner, other, lower, q < ell):
                        valid = False
                        break
                if not valid or not impose(winner, lower, upper):
                    continue
                a, av = upper
                b, bv = lower
                if a == b or a == -1 or b == -1:
                    variable = b if a == -1 else a
                    if variable == -1:
                        winner = PackedTerm(winner.coef * (prefix[av[0] + 1] - prefix[bv[0]]),
                                            winner.unary, winner.edges)
                    else:
                        multiply(winner, variable, prefix[av + 1] - prefix[bv])
                    emit(winner, kept, heads, links, stats)
                else:
                    for bound, offset, sign in ((upper, 1, 1.), (lower, 0, -1.)):
                        piece = copy_term(winner)
                        piece = PackedTerm(piece.coef * sign, piece.unary, piece.edges)
                        multiply(piece, bound.axis, prefix[bound.values + offset])
                        emit(piece, kept, heads, links, stats)
    # Keep chain indices stable until all emissions (including cancellations) finish.
    result = List.empty_list(TERM)
    for term in kept:
        if term.coef != 0.:
            result.append(term)
    stats[2] = max(stats[2], len(result))
    return result


@njit(cache=True)
def sum_out(terms, count, stats):
    # Consecutive scalar eliminations are the numerical implementation of each pair.
    for axis in range(count):
        terms = eliminate(terms, axis, stats)
    return terms


@njit(cache=True)
def apply_easy(terms, slabs, masks):
    """Copy the state and absorb unary/pair constraints without leaving Numba."""
    updated = List.empty_list(TERM)
    for old in terms:
        term = copy_term(old)
        for axis, cut in slabs:
            multiply(term, axis, np.arange(len(term.unary[axis])) > cut)
        for source, target, cutoff in masks:
            add_edge(term, source, target, False, cutoff)
        if alive(term):
            updated.append(term)
    return updated


@njit(cache=True)
def rectangle_sum(terms, lo, hi, stats):
    if np.any(lo > hi):
        return 0.
    restricted = List.empty_list(TERM)
    for old in terms:
        term = copy_term(old)
        for axis in range(len(lo)):
            indices = np.arange(len(term.unary[axis]))
            multiply(term, axis, (indices >= lo[axis]) & (indices <= hi[axis]))
        if alive(term):
            restricted.append(term)
    answer = 0.
    for term in sum_out(restricted, len(lo), stats):
        answer += term.coef
    return answer


@njit(cache=True)
def base_sum(terms, boxes, lo, hi, stats):
    """All small-base inclusion-exclusion terms in one native call."""
    if len(boxes) >= 63:
        raise ValueError('The compiled base supports fewer than 63 hard boxes.')
    answer = 0.
    for mask in range(1 << len(boxes)):
        upper, sign = hi.copy(), 1.
        for j in range(len(boxes)):
            if (mask >> (len(boxes) - 1 - j)) & 1:
                upper = np.minimum(upper, boxes[j])
                sign = -sign
        answer += sign * rectangle_sum(terms, lo, upper, stats)
    return answer


@njit(cache=True)
def push_blocks(terms, blocks, stats):
    """Lift d old variables to 2d, sum the old ones, and relabel the block axes."""
    d = len(blocks)
    lifted = List.empty_list(TERM)
    for old in terms:
        term = copy_term(old)
        for axis in range(d):
            term.unary[axis + d] = np.ones(len(blocks[axis]))
        valid = True
        for axis in range(d):
            identity = Bound(axis, np.arange(len(term.unary[axis]), dtype=np.int64))
            lower = Bound(axis + d, blocks[axis][:, 0].copy())
            upper = Bound(axis + d, blocks[axis][:, 1].copy())
            if not impose(term, lower, identity) or not impose(term, identity, upper):
                valid = False
                break
        if valid:
            lifted.append(term)
    result = List.empty_list(TERM)
    for term in sum_out(lifted, d, stats):
        unary, edges = Dict.empty(types.int64, FLOATS), Dict.empty(EDGE_KEY, INTS)
        for axis, values in term.unary.items():
            unary[axis - d] = values
        for key, values in term.edges.items():
            i, j, upper, orient = key
            edges[i - d, j - d, upper, orient] = values
        result.append(PackedTerm(term.coef, unary, edges))
    stats[5] += 1
    return result


def pack(terms):
    """Create the initial native state; descendants keep this representation."""
    result = List.empty_list(TERM)
    for term in terms:
        unary, edges = Dict.empty(types.int64, FLOATS), Dict.empty(EDGE_KEY, INTS)
        for axis, values in term.unary.items():
            unary[axis] = np.ascontiguousarray(values, dtype=np.float64)
        for key, values in term.edges.items():
            edges[tuple(int(x) for x in key)] = np.ascontiguousarray(values, dtype=np.int64)
        result.append(PackedTerm(float(term.coef), unary, edges))
    return result


def record_counters(target, counters):
    for i, name in enumerate(COUNTERS):
        if i in (2, 3, 4):
            target[name] = max(target[name], int(counters[i]))
        else:
            target[name] += int(counters[i])
