from dataset_creation.create_dataset import BaseDataset
from sklearn.model_selection import train_test_split
from probe.probes import MLPClassifier, MLPRegressor, LinearRegressor, LogisticRegressor, LitRegressor, LitClassifier
import pandas as pd
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import lightning as L
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch import loggers as pl_loggers

def main():
    
    parser = argparse.ArgumentParser(
        'Run probes'
    )
    parser.add_argument(
        'model', type=str, help='model name to probe, if facebook model only name is required'
    )
    parser.add_argument(
        'layer', type=int, help='layer of the model to probe'
    )
    parser.add_argument(
        '-l', '--labels', type=str, help='path to task csv of labels.'
    )
    parser.add_argument(
        '-n', '--neural_dim', type=int, default=768, help='dimension of transformer outputs.'
    )
    parser.add_argument(
        '-i', '--hidden_dim', type=int, default=512, help='Inner dimension if using MLP'
    )
    parser.add_argument(
        '-o', '--output_dim', type=int, default=1, help="Output dimension of probe."
    )
    parser.add_argument(
        '-d', '--root_dir', type=Path, default=Path('data/feats/switchboard'), help='Path to probed features'
    )
    parser.add_argument(
        '-c', '--corpus_name', type=str, default='switchboard', help='corpus probed'
    )
    parser.add_argument(
        '-t', '--task', type=str, default='phones_accents', help='name of the task to probe'
    )
    parser.add_argument(
        '-r', '--regression', type=bool, default=True, help='Whether this is a regression task'
    )
    parser.add_argument(
        '-p', '--probe', type=str, default='linear', help="Whether probe is 'linear' or 'mlp'."
    )
    parser.add_argument(
        '--gpu_count', type=int, default=1, help='Number of GPUs to train with'
    )
    
    args = parser.parse_args()
    
    """Variables
    layer: layer to probe
    root_dir: directory of probed features
    
    """
    
    #train_test_split
    log_path = Path(f"logs/{args.corpus_name}/{args.task}/{args.model}/{args.probe}")
    neural_dim = args.neural_dim
    if args.task == 'f0':
        out_dim = 1
        regression = True
    elif args.task == 'tone':
        out_dim = 5
        regression = False
    else:
        out_dim = 2
        regression = False
    hidden_dim = args.hidden_dim
    root_dir = args.root_dir / args.model / f'layer-{args.layer}'
    mlp = args.probe
    csv_path = args.labels
    
    if regression:
        
        if mlp == 'mlp':
            
            model = MLPRegressor(neural_dim, 1, hidden_dim)
        
        else:
            model = LinearRegressor(neural_dim, 1)
            
        litModel = LitRegressor(neural_dim, 1, hidden_dim=hidden_dim)
    
    else:
        if mlp == 'mlp':
            model = MLPClassifier(neural_dim, out_dim, hidden_dim)
        
        else:
            model = LogisticRegressor(neural_dim, out_dim)
            
        litModel = LitClassifier(model, neural_dim, out_dim, hidden_dim=hidden_dim)
        
    csv_data = pd.read_csv(csv_path)
    train, test = train_test_split(csv_data, test_size=0.2, random_state=42, shuffle=True)
    
    train, val = train_test_split(train, test_size=0.1, random_state=550, shuffle=True)
    
    train.sort_values(by=['file_id', 'start'], inplace=True)
    test.sort_values(by=['file_id', 'start'], inplace=True)
    val.sort_values(by=['file_id', 'start'], inplace=True)
    
    os.makedirs('tmp_datasets', exist_ok=True)
    train.to_csv(f'tmp_datasets/train_{args.task}.csv')
    test.to_csv(f'tmp_datasets/test_{args.task}.csv')
    val.to_csv(f'tmp_datasets/val_{args.task}.csv')
    
    train_loader = DataLoader(BaseDataset(Path(f'tmp_datasets/train_{args.task}.csv'), root_dir, tensor=True, filter_silences=True), batch_size=64)
    test_loader = DataLoader(BaseDataset(Path(f'tmp_datasets/test_{args.task}.csv'), root_dir, tensor=True, filter_silences=True), batch_size=64)
    val_loader = DataLoader(BaseDataset(Path(f'tmp_datasets/val_{args.task}.csv'), root_dir, tensor=True, filter_silences=True), batch_size=64)
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path)
    trainer = L.Trainer(
        devices=args.gpu_count, accelerator='cuda',
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],#, ModelSummary(model, max_depth=-1)],
        #fast_dev_run=True,
        #num_sanity_val_steps=2,
        logger=tb_logger
        )
 
    trainer.fit(litModel, train_loader, val_loader)
    trainer.test(litModel, dataloaders=test_loader)
    trainer.save_checkpoint('logs/current_model.ckpt')
    if regression:
        model = LitRegressor.load_from_checkpoint("logs/current_model.ckpt", model_class=model, in_dim=neural_dim, out_dim=out_dim)
    else:
        model = LitClassifier.load_from_checkpoint("logs/current_model.ckpt", model_class=model, in_dim=neural_dim, out_dim=out_dim)
    model.eval()
    test_results = list()
    test_dataset = BaseDataset(Path(f'tmp_datasets/test_{args.task}.csv'), root_dir, tensor=True, filter_silences=True)
    with torch.no_grad():
        for idx in test.index:
            test_results.append(model(test_dataset[idx][0]))
    test[f'{mlp}_preds'] = test_results
    os.makedirs(f'logs/test_results/', exist_ok=True)
    test.to_csv(f'logs/test_results/layer-{args.layer}_{args.task}.csv')
    
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
    
    
    
if __name__ == '__main__':
    main()   