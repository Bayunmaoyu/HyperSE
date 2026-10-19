from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from .datasets import canonical_name, name_key


@dataclass(frozen=True)
class Event:
    nodes: tuple
    time: float


DEFAULT_ROOTS = [r'D:\PythonExm\data\HG_Data_nolabeled',
                 r'D:\PythonExm\data\HG_Data_labeled',
                 str(Path(__file__).parent.parent/'data')]


def locate(root, name):
    name = canonical_name(name)
    roots = [Path(root)] if root else [Path(p) for p in DEFAULT_ROOTS]
    for base in roots:
        if not base.exists():
            continue
        for f in sorted(base.rglob('*-nverts.txt')):
            if name_key(canonical_name(f.name[:-11])) == name_key(name):
                prefix = str(f)[:-11]
                if all(Path(prefix+suffix).is_file() for suffix in ['-simplices.txt','-times.txt']):
                    return f
    raise FileNotFoundError(f'Missing {name} nverts/simplices/times triplet below {roots}. See README.txt.')


def load_events(root, name, limit=0):
    f = locate(root, name)
    prefix = str(f)[:-11]
    paths = [f, Path(prefix + '-simplices.txt'), Path(prefix + '-times.txt')]
    sizes = np.loadtxt(paths[0], dtype=np.int64, ndmin=1)
    nodes = np.loadtxt(paths[1], dtype=np.int64, ndmin=1)
    times = np.loadtxt(paths[2], dtype=np.float64, ndmin=1)
    if len(sizes) != len(times) or sizes.sum() != len(nodes):
        raise ValueError('nverts/simplices/times lengths are inconsistent')
    if np.any(sizes < 1) or not np.isfinite(times).all():
        raise ValueError('Empty hyperedge or invalid timestamp')
    raw = []
    offset = 0
    duplicate_vertices = 0
    for size, t in zip(sizes, times):
        edge = tuple(sorted(set(map(int, nodes[offset:offset+size]))))
        duplicate_vertices += int(size) - len(edge)
        raw.append(Event(edge, float(t)))
        offset += size
    raw.sort(key=lambda e: e.time)  # stable; preserve repeated events and singletons
    if limit:
        if limit < 20:
            raise ValueError('--limit must be 0 or >=20')
        cut = min(limit, len(raw))
        while cut < len(raw) and raw[cut].time == raw[cut-1].time:
            cut += 1  # never split a simultaneous group
        raw = raw[:cut]
    meta = dict(dataset=name, source_files=[str(p) for p in paths],
                source_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                original_events=len(sizes), used_events=len(raw),
                nodes=len(set(v for e in raw for v in e.nodes)),
                timestamps=len(set(e.time for e in raw)),
                max_size=max(map(lambda e: len(e.nodes), raw)),
                singletons=sum(len(e.nodes) == 1 for e in raw),
                duplicate_vertices_removed=duplicate_vertices, limited=bool(limit))
    return raw, meta


def split_events(events, seed=42, setting='transductive'):
    times = np.array([e.time for e in events])
    a, b = np.quantile(times, [0.70, 0.85])
    train = [e for e in events if e.time <= a]
    valid = [e for e in events if a < e.time <= b]
    test = [e for e in events if e.time > b]
    hidden = set()
    if setting != 'transductive':
        # ID inventory only; no future adjacency, timestamp, label or feature is used.
        vocab = sorted({v for e in events for v in e.nodes})
        rng = np.random.default_rng(seed)
        hidden = set(map(int, rng.choice(vocab, max(1, len(vocab)//10), replace=False)))
        train = [e for e in train if not hidden.intersection(e.nodes)]
        def keep(e):
            k = len(hidden.intersection(e.nodes))
            return (0 < k < len(e.nodes)) if setting == 'weak' else k == len(e.nodes)
        valid, test = [e for e in valid if keep(e)], [e for e in test if keep(e)]
    if min(map(len, (train, valid, test))) < 2:
        raise ValueError('Too few events in a split; use more data or another setting/seed')
    assert max(e.time for e in train) < min(e.time for e in valid)
    assert max(e.time for e in valid) < min(e.time for e in test)
    return train, valid, test, hidden, dict(train_end=float(a), valid_end=float(b),
                                          train=len(train), valid=len(valid), test=len(test))


def time_scale(train):
    # Fit ONLY on training timestamps. Unit interval is the training time span.
    return max(float(train[-1].time - train[0].time), 1.0)


def blocks(events, size):
    i = 0
    while i < len(events):
        j = min(i+size, len(events))
        while j < len(events) and events[j].time == events[j-1].time:
            j += 1
        yield events[i:j]
        i = j


class NodePool:
    """Reusable stratified node pool for large temporal hypergraphs."""
    def __init__(self, nodes, hidden=frozenset()):
        self.hidden = set(hidden)
        nodes = np.asarray(sorted(set(nodes)),dtype=np.int64)
        mask = np.isin(nodes,list(self.hidden))
        self.new_nodes,self.seen_nodes = nodes[mask],nodes[~mask]

    def sample(self,event,forbidden,rng,attempts=500):
        original = set(event.nodes)
        hidden_count = len(original & self.hidden)
        for _ in range(attempts):
            keep = set(map(int,rng.choice(event.nodes,len(event.nodes)//2,replace=False)))
            keep_hidden = len(keep & self.hidden)
            nh = hidden_count-keep_hidden
            ns = len(original)-hidden_count-(len(keep)-keep_hidden)
            if len(self.new_nodes)-keep_hidden < nh or len(self.seen_nodes)-(len(keep)-keep_hidden) < ns:
                raise ValueError('Negative pool too small for cardinality/inductive constraints')
            h = set(map(int,rng.choice(self.new_nodes,nh,replace=False)))
            s = set(map(int,rng.choice(self.seen_nodes,ns,replace=False)))
            if (h | s) & keep:
                continue
            candidate = tuple(sorted(keep | h | s))
            if candidate != event.nodes and candidate not in forbidden:
                return Event(candidate,event.time)
        raise ValueError('No valid negative after 500 attempts; dense timestamp or tiny node pool')


def sample_negative(event, pool, forbidden, rng, hidden=frozenset(), attempts=500):
    """Half corruption without replacement; same size and held-out-node count.

    Reject observed positives at the SAME timestamp. Do not consult future events.
    Earlier positive sets may be negative at a different time in a temporal task.
    """
    if isinstance(pool,NodePool):
        if set(hidden) != pool.hidden:
            raise ValueError('Node pool and sampler hidden-node sets disagree')
        return pool.sample(event,forbidden,rng,attempts)
    original = set(event.nodes)
    for _ in range(attempts):
        keep = set(map(int, rng.choice(event.nodes, len(event.nodes)//2, replace=False)))
        want_hidden = len(original & hidden) - len(keep & hidden)
        want_seen = len(original - hidden) - len(keep - hidden)
        available = set(pool) - keep
        h = sorted(available & hidden)
        s = sorted(available - hidden)
        if len(h) < want_hidden or len(s) < want_seen:
            raise ValueError('Negative pool too small for cardinality/inductive constraints')
        candidate = tuple(sorted(keep | set(map(int, rng.choice(h, want_hidden, replace=False)))
                                  | set(map(int, rng.choice(s, want_seen, replace=False)))))
        if candidate != event.nodes and candidate not in forbidden:
            return Event(candidate, event.time)
    raise ValueError('No valid negative after 500 attempts; dense timestamp or tiny node pool')


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
