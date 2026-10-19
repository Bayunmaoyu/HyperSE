from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
from .data import load_events, split_events, time_scale, blocks, sample_negative, save_json, NodePool
from .features import Snapshot, make_chunk, STAT_NAMES


def prepare(args):
    events, meta = load_events(args.data_root,args.dataset,args.limit)
    train,valid,test,hidden,split = split_events(events,args.seed,args.setting)
    vocab = sorted({v for e in train for v in e.nodes})
    # Fixed coordinate vocabulary comes from training IDs only.
    anchors = vocab if args.anchors == 0 else [vocab[i] for i in np.linspace(0,len(vocab)-1,min(args.anchors,len(vocab)),dtype=int)]
    if args.anchors:
        anchors += [-1]*(args.anchors-len(anchors))
    dual_dim = args.history if args.anchors == 0 else args.anchors
    tau = time_scale(train)*args.tau
    evaluation = args.evaluation
    if evaluation == 'auto':
        evaluation = 'prequential' if args.setting == 'transductive' else 'frozen'
    sampling_version = getattr(args,'sampling_version',1)
    config = dict(schema=sampling_version,source_sha256=meta['source_sha256'],dataset=args.dataset,limit=args.limit,
                  seed=args.seed,setting=args.setting,history=args.history,block=args.block,
                  anchors=anchors,dual_dim=dual_dim,height=args.height,dual_height=args.dual_height,
                  neighbors=args.neighbors,tau=tau,warmup=args.warmup,evaluation=evaluation)
    digest = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    folder = Path(args.cache)/digest
    manifest_path = folder/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if all((folder/f).exists() for key in ('train','valid','test') for f in manifest['files'][key]):
            print(f'Using cached causal features: {folder}',flush=True)
            return folder,manifest
    folder.mkdir(parents=True,exist_ok=True)
    forbidden = defaultdict(set)
    for e in events:
        forbidden[e.time].add(e.nodes)  # only same-time labels used by negative sampler
    max_size = max(len(e.nodes) for e in events)
    warm = min(max(16,int(len(train)*args.warmup)),len(train)-2)
    while warm < len(train) and train[warm].time == train[warm-1].time:
        warm += 1
    if warm >= len(train):
        raise ValueError('No training targets after whole-timestamp warmup')
    history = train[:warm]
    seen_nodes = {v for e in history for v in e.nodes}
    manifest = dict(config=config,data=meta,split=split,hidden_nodes=sorted(hidden),warmup_events=warm,
                    node_dim=len(anchors),dual_dim=dual_dim,stats_names=STAT_NAMES,max_size=max_size,
                    files={key:[] for key in ('train','valid','test')},audits=[])
    for phase_idx,(phase,targets) in enumerate([('train',train[warm:]),('valid',valid),('test',test)]):
        rng = np.random.default_rng(args.seed+1009*phase_idx)
        if phase != 'train' and evaluation == 'frozen':
            history = list(train)
            seen_nodes = {v for e in history for v in e.nodes}
        for block_idx,group in enumerate(blocks(targets,args.block)):
            context = history[-args.history:]
            snapshot = Snapshot(context,anchors,dual_dim,args.height,args.dual_height,args.neighbors,tau)
            # Same-time target IDs may be given as a query, never as graph observations.
            # Add held-out IDs only as an explicit node inventory for inductive corruption.
            pool = sorted(seen_nodes | {v for e in group for v in e.nodes} | hidden)
            if sampling_version >= 2:
                pool = NodePool(pool,hidden)
            examples,labels = [],[]
            for e in group:
                neg = sample_negative(e,pool,forbidden[e.time],rng,hidden)
                examples.extend((e,neg)); labels.extend((1,0))
            chunk = make_chunk(snapshot,examples,labels,max_size)
            name = f'{phase}_{block_idx:05d}.npz'
            np.savez_compressed(folder/name,**chunk)
            manifest['files'][phase].append(name)
            manifest['audits'].append(dict(phase=phase,block=block_idx,query_min_time=group[0].time,
                                          examples=len(examples),**snapshot.audit))
            if phase == 'train' or evaluation == 'prequential':
                history.extend(group)  # reveal true past observations only after scoring entire block
                seen_nodes.update(v for e in group for v in e.nodes)
            if block_idx%5 == 0:
                print(f'prepare {phase} block={block_idx} examples={len(examples)} history={len(context)}',flush=True)
    save_json(manifest_path,manifest)
    return folder,manifest
