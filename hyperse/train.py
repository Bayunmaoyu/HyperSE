from pathlib import Path
from argparse import Namespace
import os
import json
import random
import time
import numpy as np
import torch
from torch import nn
from .model import HyperSE
from .metrics import binary_metrics
from .data import save_json


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


class ChunkStore:
    """Load large datasets one compressed feature chunk at a time."""
    def __init__(self,folder,names):
        self.folder,self.names = Path(folder),list(names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self,index):
        with np.load(self.folder/self.names[index],allow_pickle=False) as data:
            chunk = {key:data[key] for key in data.files}
        if not np.all(chunk['times'] > chunk['history_max_time']):
            raise AssertionError('Cached temporal leakage')
        return chunk


def load_chunks(folder,manifest,phase):
    chunks = ChunkStore(folder,manifest['files'][phase])
    return chunks if len(chunks)>128 else list(chunks)


def device_chunk(chunk,device,mean,std):
    keys = ['node_features','dual_features','nodes','node_mask','neighbors','neighbor_mask','stats','labels']
    d = {k:torch.from_numpy(chunk[k]).to(device) for k in keys}
    d['stats'] = (d['stats']-mean)/std
    return d


def forward_batch(model,d,index):
    return model(d['node_features'][d['nodes'][index]],d['node_mask'][index],
                 d['dual_features'][d['neighbors'][index]],d['neighbor_mask'][index],d['stats'][index])


@torch.no_grad()
def predict_chunks(model,chunks,device,mean,std,batch):
    model.eval()
    labels,predictions = [],[]
    for chunk in chunks:
        d = device_chunk(chunk,device,mean,std)
        for i in range(0,len(chunk['labels']),batch):
            pred = torch.sigmoid(forward_batch(model,d,slice(i,i+batch)))
            predictions.extend(pred.cpu().tolist())
        labels.extend(chunk['labels'].tolist())
    return np.asarray(labels),np.asarray(predictions)


def train(args,folder,manifest):
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'best.pt').exists() and not args.overwrite:
        raise FileExistsError(f'{output}/best.pt exists; use another --output or --overwrite')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    tr = load_chunks(folder,manifest,'train')
    va = load_chunks(folder,manifest,'valid')
    stats = np.concatenate([c['stats'] for c in tr])
    mu,sigma = stats.mean(0),np.maximum(stats.std(0),0.1)
    mean,std = torch.from_numpy(mu).to(device),torch.from_numpy(sigma).to(device)
    spec = dict(node_dim=manifest['node_dim'],dual_dim=manifest['dual_dim'],hidden=args.hidden,
                dropout=args.dropout,mode=args.mode,ablation=args.ablation)
    model = HyperSE(**spec).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=3)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    best,best_epoch,stale = -1.,0,0
    log = []
    start = time.perf_counter()
    print(f'Training {args.mode} on {device}; parameters={sum(p.numel() for p in model.parameters())}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        total_loss,total_n = 0.,0
        for ci in rng.permutation(len(tr)):
            c = tr[ci]
            d = device_chunk(c,device,mean,std)
            order = rng.permutation(len(c['labels']))
            for j in range(0,len(order),args.batch_size):
                ix = torch.tensor(order[j:j+args.batch_size],dtype=torch.long,device=device)
                opt.zero_grad(set_to_none=True)
                logits = forward_batch(model,d,ix)
                loss = criterion(logits,d['labels'][ix])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                total_loss += loss.item()*len(ix); total_n += len(ix)
        y,p = predict_chunks(model,va,device,mean,std,args.batch_size)
        metrics = binary_metrics(y,p)
        selection = (metrics['auc']+metrics['ap'])/2
        scheduler.step(selection)
        row = dict(epoch=epoch,train_bce=total_loss/total_n,valid=metrics,selection=selection,
                   lr=opt.param_groups[0]['lr'])
        log.append(row)
        print(f"epoch={epoch:03d} BCE={row['train_bce']:.4f} val_AUC={metrics['auc']:.4f} val_AP={metrics['ap']:.4f}",flush=True)
        if selection > best + 1e-6:
            best,best_epoch,stale = selection,epoch,0
            checkpoint = dict(model_spec=spec,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                              stats_mean=mu,stats_std=sigma,manifest=manifest,args=vars(args),
                              cache_folder=os.path.abspath(folder),best_epoch=epoch,valid=metrics,
                              versions=dict(torch=torch.__version__,numpy=np.__version__))
            torch.save(checkpoint,output/'best.pt')
            np.savez_compressed(output/'valid_predictions.npz',labels=y,scores=p)
        else:
            stale += 1
        if stale >= args.patience:
            break
    save_json(output/'training_log.json',log)
    save_json(output/'data_manifest.json',manifest)
    summary = dict(dataset=args.dataset,setting=args.setting,mode=args.mode,ablation=args.ablation,
                   best_epoch=best_epoch,valid=checkpoint['valid'],seed=args.seed,
                   elapsed_training_seconds=time.perf_counter()-start,device=str(device),
                   test=None,test_status='deferred; not inspected',data=manifest['data'],config=manifest['config'])
    save_json(output/'metrics.json',summary)
    if not args.defer_test:
        return evaluate(output/'best.pt',args.device,args.batch_size)
    return summary


def load_checkpoint(path):
    # Only load checkpoints created by this project or another trusted source.
    # Explicit weights_only=False is needed for metadata on new PyTorch releases.
    return torch.load(path,map_location='cpu',weights_only=False)


def evaluate(path,device_name='auto',batch=256):
    path = Path(path)
    cp = load_checkpoint(path)
    seed_all(cp['args']['seed'])
    device = torch.device('cuda' if device_name == 'auto' and torch.cuda.is_available() else
                          'cpu' if device_name == 'auto' else device_name)
    model = HyperSE(**cp['model_spec']).to(device)
    model.load_state_dict(cp['state_dict'])
    mean = torch.from_numpy(cp['stats_mean']).to(device)
    std = torch.from_numpy(cp['stats_std']).to(device)
    cache = Path(cp['cache_folder'])
    if not all((cache/name).exists() for name in cp['manifest']['files']['test']):
        from .prepare import prepare
        rebuild = Namespace(**cp['args'])
        rebuild.cache = str(path.parent/'rebuilt_cache')
        cache, rebuilt_manifest = prepare(rebuild)
        if rebuilt_manifest['config'] != cp['manifest']['config']:
            raise ValueError('Rebuilt features differ from checkpoint; check source file hashes/config')
    te = load_chunks(cache,cp['manifest'],'test')
    y,p = predict_chunks(model,te,device,mean,std,batch)
    metrics = binary_metrics(y,p)
    np.savez_compressed(path.parent/'test_predictions.npz',labels=y,scores=p)
    summary_path = path.parent/'metrics.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    summary.update(test=metrics,test_status='evaluated after validation checkpoint selection')
    save_json(summary_path,summary)
    print(json.dumps(dict(test=metrics,best_epoch=cp['best_epoch']),indent=2),flush=True)
    return summary
