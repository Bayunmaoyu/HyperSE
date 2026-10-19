"""Example: python run.py --dataset email-Enron --mode enhanced --output results/enron"""
import argparse
import json
from hyperse.prepare import prepare
from hyperse.train import train
from hyperse.datasets import dataset_status, canonical_name


def parser():
    p = argparse.ArgumentParser(description='HyperSE temporal hyperedge prediction. See README.txt.')
    p.add_argument('--dataset',default='email-Enron')
    p.add_argument('--data-root',default=None,help='Search a dataset root; default: both HG_Data directories and project data/')
    p.add_argument('--list-datasets',action='store_true',help='Show availability of all six benchmark datasets')
    p.add_argument('--output',default='results/enron_enhanced')
    p.add_argument('--cache',default='cache')
    p.add_argument('--mode',choices=['core','enhanced'],default='enhanced')
    p.add_argument('--setting',choices=['transductive','weak','strong'],default='transductive')
    p.add_argument('--evaluation',choices=['auto','prequential','frozen'],default='auto')
    p.add_argument('--ablation',choices=['none','no_primal','no_dual'],default='none')
    p.add_argument('--history',type=int,default=1024,help='Max earlier events per snapshot')
    p.add_argument('--block',type=int,default=256,help='Snapshot update interval; same timestamp never split')
    p.add_argument('--anchors',type=int,default=64,help='CSE landmark columns; 0=all training nodes and all history slots')
    p.add_argument('--height',type=int,default=5)
    p.add_argument('--dual-height',type=int,default=15)
    p.add_argument('--neighbors',type=int,default=25)
    p.add_argument('--tau',type=float,default=0.2,help='Decay time as fraction of TRAIN time span')
    p.add_argument('--warmup',type=float,default=0.1)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--weight-decay',type=float,default=0.0001)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sampling-version',type=int,choices=[1,2],default=2,help='1=legacy seed sequence, 2=reusable node pools')
    p.add_argument('--device',default='auto')
    p.add_argument('--limit',type=int,default=0,help='0=full dataset; otherwise chronological prefix for smoke tests')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--defer-test',action='store_true',help='For model selection: do not evaluate test')
    p.add_argument('--overwrite',action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.list_datasets:
        print(json.dumps(dataset_status(args.data_root),ensure_ascii=False,indent=2))
        return
    args.dataset = canonical_name(args.dataset)
    if min(args.history,args.block,args.height,args.dual_height,args.neighbors,args.hidden,args.epochs,args.batch_size,args.patience) < 1:
        raise ValueError('Integer sizes must be positive')
    if args.anchors < 0 or args.tau <= 0 or not 0 < args.warmup < 0.5:
        raise ValueError('Invalid anchors/tau/warmup')
    folder,manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps(manifest['data'],indent=2))
        return
    train(args,folder,manifest)


if __name__ == '__main__':
    main()
