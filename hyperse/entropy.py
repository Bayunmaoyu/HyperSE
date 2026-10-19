"""Encoding trees, conditional structural entropy, and partition merge gains."""
from collections import defaultdict, Counter
from itertools import combinations
import math
import numpy as np


def compress_hyperedges(edges, labels, weights=None):
    """Eq. (13): keep set-valued hyperedges; add weights of identical images."""
    out = defaultdict(float)
    if weights is None:
        weights = np.ones(len(edges))
    for edge, w in zip(edges, weights):
        out[tuple(sorted({int(labels[v]) for v in edge}))] += float(w)
    return dict(out)


def hypergraph_measure(edges, n, weights=None):
    """Eq. (6)/(7) masses; reversible, no-self-jump incidence flow for cuts.

    Each hyperedge contributes w/s at each vertex. For s>1 each ordered pair
    carries w/[s(s-1)]. A singleton contributes self mass w and zero cut.
    Cross weights are cut statistics only; the hyperedges remain available.
    """
    mass = np.zeros(n, dtype=np.float64)
    cut = np.zeros(n, dtype=np.float64)
    cross = defaultdict(float)
    if weights is None:
        weights = np.ones(len(edges))
    for edge, w in zip(edges, weights):
        s = len(edge)
        for v in edge:
            mass[v] += w/s
            if s > 1:
                cut[v] += w/s
        if s > 1:
            for u, v in combinations(sorted(edge), 2):
                cross[u, v] += float(w)/(s*(s-1))
    return mass, cut, dict(cross)


def merge_gain(cross, mass_a, mass_b, total):
    """Decrease in H(T) when inserting A union B below the root.

    H(T)=-sum_C g(C)/V * log2(v(C)/v(parent(C))).
    g(A union B)=g(A)+g(B)-2*w(A,B).
    """
    if cross <= 0 or total <= 0 or mass_a + mass_b <= 0:
        return 0.0
    return 2*cross/total * max(0.0, math.log2(total/(mass_a+mass_b)))


class EncodingTree:
    def __init__(self, mass, cut, cross, height=5, hyperedges=None):
        if height < 1:
            raise ValueError('Tree height must be positive')
        self.n = len(mass)
        self.mass = list(map(float, mass))
        self.cut = list(map(float, cut))
        self.leaves = [[i] for i in range(self.n)]
        self.parent = [-1]*self.n
        self.total = float(np.sum(mass))
        active = list(range(self.n))
        self.gains = []
        self.compressed_levels = []
        current_edges = {tuple(e): float(w) for e, w in Counter(hyperedges or []).items()}
        # Greedy matching with a maximum tree height.
        for _ in range(height-1):
            ranked = sorted(((merge_gain(w, self.mass[a], self.mass[b], self.total), a, b, w)
                             for (a, b), w in cross.items()), key=lambda x: (-x[0], x[1], x[2]))
            mapping, used = {}, set()
            for gain, a, b, w in ranked:
                if gain <= 1e-14 or a in used or b in used:
                    continue
                p = len(self.mass)
                self.mass.append(self.mass[a]+self.mass[b])
                self.cut.append(max(0., self.cut[a]+self.cut[b]-2*w))
                self.leaves.append(self.leaves[a]+self.leaves[b])
                self.parent.extend([-1])
                self.parent[a] = self.parent[b] = p
                mapping[a] = mapping[b] = p
                used.update((a,b))
                self.gains.append(gain)
            if not used:
                break
            for a in active:
                mapping.setdefault(a, a)
            new_cross = defaultdict(float)
            for (a,b), w in cross.items():
                a,b = mapping[a], mapping[b]
                if a != b:
                    new_cross[tuple(sorted((a,b)))] += w
            cross = dict(new_cross)
            active = sorted(set(mapping.values()))
            if current_edges:
                current_edges = compress_hyperedges(list(current_edges), mapping, list(current_edges.values()))
                self.compressed_levels.append(current_edges)
        self.root = len(self.mass)
        self.mass.append(self.total)
        self.cut.append(0.)
        self.leaves.append(list(range(self.n)))
        self.parent.append(-1)
        for a in active:
            self.parent[a] = self.root
        self.local = np.zeros(len(self.mass))
        if self.total > 0:
            for a in range(self.root):
                pm = self.mass[self.parent[a]]
                if self.mass[a] > 0 and pm > 0:
                    self.local[a] = -self.cut[a]/self.total * math.log2(self.mass[a]/pm)

    @property
    def entropy(self):
        return float(self.local.sum())

    def cse(self, anchors):
        """Row i, column j = H(anchor_j | i), Eq. (14)/(15), LCA excluded.

        -1 anchor means a missing historical node and yields a zero column.
        No candidate hyperedge is inserted while computing this representation.
        """
        out = np.zeros((self.n, len(anchors)), dtype=np.float32)
        for j, leaf in enumerate(anchors):
            if leaf < 0:
                continue
            a = int(leaf)
            while a != self.root:
                if self.local[a] > 0:
                    outside = np.ones(self.n, dtype=bool)
                    outside[self.leaves[a]] = False
                    out[outside, j] += self.local[a]
                a = self.parent[a]
        return out


def dual_graph(edges, times, tau, topk):
    """Exact Jaccard*exp(-|dt|/tau) within the configured history window.

    Construct via inverted incidences; exclude diagonal; symmetric union of
    directed SE top-k lists supplies a valid undirected encoding-tree graph.
    """
    m = len(edges)
    inverted = defaultdict(list)
    for i, edge in enumerate(edges):
        for v in edge:
            inverted[v].append(i)
    overlap = defaultdict(int)
    for occurrences in inverted.values():
        for i,j in combinations(occurrences, 2):
            overlap[i,j] += 1
    weights = {}
    degrees = np.zeros(m, dtype=np.float64)
    for (i,j), overlap_size in overlap.items():
        w = overlap_size/(len(edges[i])+len(edges[j])-overlap_size) * math.exp(-abs(times[i]-times[j])/tau)
        if w > 0:
            weights[i,j] = w
            degrees[i] += w
            degrees[j] += w
    total = float(degrees.sum())
    ranked = [[] for _ in range(m)]
    for (i,j), w in weights.items():
        gain = merge_gain(w, degrees[i], degrees[j], total)
        ranked[i].append((gain, j, w))
        ranked[j].append((gain, i, w))
    chosen = set()
    for i, row in enumerate(ranked):
        for _, j, _ in sorted(row, key=lambda x:(-x[0],x[1]))[:topk]:
            chosen.add(tuple(sorted((i,j))))
    sparse = {key:weights[key] for key in chosen}
    pruned_degree = np.zeros(m, dtype=np.float64)
    for (i,j), w in sparse.items():
        pruned_degree[i] += w
        pruned_degree[j] += w
    return pruned_degree, sparse, degrees, total
