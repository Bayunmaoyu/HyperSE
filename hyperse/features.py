from collections import Counter, defaultdict
from itertools import combinations
import math
import numpy as np
from .entropy import EncodingTree, hypergraph_measure, dual_graph, merge_gain


STAT_NAMES = ['log_size', 'node_count_min', 'node_count_mean', 'node_count_max',
              'pair_count_min', 'pair_count_mean', 'pair_count_max', 'pair_seen_fraction',
              'log_recurrence', 'set_recency', 'overlap_max', 'overlap_mean',
              'log_neighbor_count', 'time_weight_max', 'time_weight_sum',
              'common_neighbors_mean', 'common_neighbors_min', 'node_recency_mean']


class Snapshot:
    def __init__(self, history, anchors, dual_dim, height, dual_height, neighbors, tau):
        if not history:
            raise ValueError('A snapshot needs historical observations')
        self.history = history
        self.time = max(e.time for e in history)
        self.neighbors, self.tau = neighbors, tau
        self.vocab = sorted({v for e in history for v in e.nodes})
        self.index = {v:i for i,v in enumerate(self.vocab)}
        local_edges = [tuple(self.index[v] for v in e.nodes) for e in history]
        mass, cut, cross = hypergraph_measure(local_edges, len(self.vocab))
        tree = EncodingTree(mass, cut, cross, height, local_edges)
        nf = tree.cse([self.index.get(v,-1) for v in anchors])*100.0
        # Last row is explicit zero for unseen/padding nodes.
        self.node_features = np.concatenate([nf, np.zeros((1,len(anchors)), np.float32)])
        self.unknown = len(self.vocab)
        self.edges = [set(e.nodes) for e in history]
        self.times = np.array([e.time for e in history])
        degree, cross, self.raw_degree, self.raw_total = dual_graph(
            [e.nodes for e in history], self.times, tau, neighbors)
        dtree = EncodingTree(degree, degree, cross, dual_height)
        # Chronological landmark positions, fixed-width. Full mode pads on left.
        if dual_dim >= len(history):
            da = [-1]*(dual_dim-len(history)) + list(range(len(history)))
        else:
            da = np.linspace(0,len(history)-1,dual_dim,dtype=int).tolist()
        df = dtree.cse(da)*100.0
        self.dual_features = np.concatenate([df, np.zeros((1,dual_dim), np.float32)])
        self.dual_unknown = len(history)
        self.inverted = defaultdict(set)
        self.counts = Counter()
        self.pair_counts = Counter()
        self.set_counts = Counter()
        self.last_node, self.last_set = {}, {}
        self.adj = defaultdict(set)
        for i,e in enumerate(history):
            self.counts.update(e.nodes)
            self.set_counts[e.nodes] += 1
            self.last_set[e.nodes] = e.time
            for v in e.nodes:
                self.inverted[v].add(i)
                self.last_node[v] = e.time
            for u,v in combinations(e.nodes,2):
                self.pair_counts[u,v] += 1
                self.adj[u].add(v)
                self.adj[v].add(u)
        self.audit = dict(history_events=len(history), history_max_time=self.time,
                          primal_entropy=tree.entropy, dual_entropy=dtree.entropy,
                          primal_merges=len(tree.gains), dual_merges=len(dtree.gains),
                          dual_undirected_edges=len(cross))

    def encode(self, event, max_size):
        if self.time >= event.time:
            raise AssertionError('Temporal leakage: history must be strictly earlier than query')
        nodes = np.full(max_size, self.unknown, dtype=np.int64)
        mask = np.zeros(max_size, dtype=np.float32)
        nodes[:len(event.nodes)] = [self.index.get(v,self.unknown) for v in event.nodes]
        mask[:len(event.nodes)] = 1
        candidate = set(event.nodes)
        ids = sorted(set().union(*(self.inverted[v] for v in candidate)))
        weights, overlaps = [], []
        for i in ids:
            jac = len(candidate & self.edges[i])/len(candidate | self.edges[i])
            overlaps.append(jac)
            weights.append(jac*math.exp(-(event.time-self.times[i])/self.tau))
        dp = sum(weights)
        total = self.raw_total + 2*dp
        gains = [merge_gain(w, dp, self.raw_degree[i]+w, total) for i,w in zip(ids,weights)]
        selected = sorted(range(len(ids)),key=lambda j:(-gains[j],ids[j]))[:self.neighbors]
        nei = np.full(self.neighbors,self.dual_unknown,dtype=np.int64)
        nm = np.zeros(self.neighbors,dtype=np.float32)
        for j,k in enumerate(selected):
            nei[j], nm[j] = ids[k], 1
        nc = [self.counts[v] for v in event.nodes]
        pairs = list(combinations(event.nodes,2))
        pc = [self.pair_counts[p] for p in pairs] or [0]
        common = [len(self.adj[u]&self.adj[v]) for u,v in pairs] or [0]
        recency = math.exp(-(event.time-self.last_set[event.nodes])/self.tau) if event.nodes in self.last_set else 0.
        nr = [math.exp(-(event.time-self.last_node[v])/self.tau) if v in self.last_node else 0 for v in event.nodes]
        stats = np.array([math.log1p(len(candidate)), math.log1p(min(nc)), math.log1p(np.mean(nc)), math.log1p(max(nc)),
                          math.log1p(min(pc)), math.log1p(np.mean(pc)), math.log1p(max(pc)), np.mean(np.array(pc)>0),
                          math.log1p(self.set_counts[event.nodes]), recency, max(overlaps,default=0), np.mean(overlaps) if overlaps else 0,
                          math.log1p(len(ids)), max(weights,default=0), math.log1p(sum(weights)),
                          math.log1p(np.mean(common)), math.log1p(min(common)), np.mean(nr)], dtype=np.float32)
        return nodes,mask,nei,nm,stats


def make_chunk(snapshot, examples, labels, max_size):
    encoded = [snapshot.encode(e,max_size) for e in examples]
    names = ['nodes','node_mask','neighbors','neighbor_mask','stats']
    result = {k:np.stack(v) for k,v in zip(names,zip(*encoded))}
    result.update(node_features=snapshot.node_features, dual_features=snapshot.dual_features,
                  labels=np.asarray(labels,dtype=np.float32), times=np.array([e.time for e in examples]),
                  history_max_time=np.array(snapshot.time))
    return result
