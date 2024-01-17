from dataset_creation.create_dataset import BaseDataset
from sklearn import train_test_split
from probe.probes import MLPClassifier, MLPRegressor, LinearRegressor, LogisticRegressor, LitRegressor, LitClassifier
import pandas as pd
import os
import argparse
from pathlib import Path
from torch.utils import DataLoader

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
        '-n', '--neural-dim', type=int, default=768, help='dimension of transformer outputs.'
    )
    parser.add_argument(
        '-h', '--hidden-dim', type=int, default=512, help='Inner dimension if using MLP'
    )
    parser.add_argument(
        '-o', '--output-dim', type=int, default=1, help="Output dimension of probe."
    )
    parser.add_argument(
        '-d', '--root-dir', type=Path, default=Path('data/switchboard/feats'), help='Path to probed features'
    )
    parser.add_argument(
        '-c', '--corpus-name', type=str, default='switchboard', help='corpus probed'
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
        
    )
    
    args = parser.parse_args()
    
    """Variables
    layer: layer to probe
    root_dir: directory of probed features
    
    """
    
    #train_test_split
    log_path = Path(f"logs/{args.c}/{args.t}/{args.model}/{args.probe}")
    neural_dim = args.n
    out_dim = args.o
    hidden_dim = args.h
    root_dir = args.d
    regression = args.r
    mlp = args.p
    csv_path = args.l
    
    if regression:
        
        if mlp:
            
            model = MLPRegressor(neural_dim, 1, hidden_dim)
        
        else:
            model = LinearRegressor(neural_dim, 1)
            
        trainer = LitRegressor(neural_dim, 1, hidden_dim=hidden_dim)
    
    else:
        if mlp:
            model = MLPClassifier(neural_dim, out_dim, hidden_dim)
        
        else:
            model = LogisticRegressor(neural_dim, out_dim)
            
        trainer = LitClassifier(model, neural_dim, out_dim, hidden_dim=hidden_dim)
        
    
    csv_data = pd.read_csv(csv_path)
    train, test = train_test_split(csv_data, test_size=0.2, random_state=42, shuffle=True)
    
    train, val = train_test_split(train, test_size=0.1, random_state=550, shuffle=True)
    
    train.sort_values(by=['file_id', 'start'], inplace=True)
    test.sort_values(by=['file_id', 'start'], inplace=True)
    val.sort_values(by=['file_id', 'start'], inplace=True)
    
    os.makedirs('tmp_datasets', exist_ok=True)
    train.to_csv('tmp_datasets/train.csv')
    test.to_csv('tmp_datasets/test.csv')
    val.to_csv('tmp_datasets/val.csv')
    
    train_loader = DataLoader(BaseDataset('tmp_datasets/train.csv', root_dir, tensor=True, filter_silences=True))
    test_loader = DataLoader(BaseDataset('tmp_datasets/test.csv', root_dir, tensor=True, filter_silences=True))
    val_loader = DataLoader(BaseDataset('tmp_datasets/val.csv', root_dir, tensor=True, filter_silences=True))
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloader=test_loader)
   
    
if __name__ == '__main__':
    main()   