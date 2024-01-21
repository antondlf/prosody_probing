from dataset_creation.create_dataset import BaseDataset
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
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
from skorch import NeuralNetRegressor, NeuralNetClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, make_scorer, mean_squared_error, r2_score
import numpy as np


def log_cross_validation(scores):
    """Assumed to be a dict output of gc.cv_results_
    or cross_validate().scores"""
    
    pass


def train_regression(train_data, root_dir):
    feats = list()
    labels = list()
    for name, group in train_data.groupby('file_id'):
        feat_file = name + '.npy'
        raw_feats = np.load(root_dir / feat_file)
        
        feats.append(raw_feats[group.start_end_indices.to_numpy(), :])
        labels.append(group.label.to_numpy())
        
    X = np.concatenate(feats, axis=0)
         
    y = np.concatenate(labels)
    
    linear_model = LinearRegression()
    cv = cross_validate(linear_model, X, y, scoring=['mean_squared_error', 'r2_score'], cv=10, n_jobs=-1)
    log_cross_validation(cv.scores)
    
    linear_model.fit(X, y)
    
    return linear_model


def train_logistic_regression(train_data, root_dir, binary=False):
    
    feats = list()
    labels = list()
    for name, group in train_data.groupby('file_id'):
        feat_file = name + '.npy'
        raw_feats = np.load(root_dir / feat_file)
        
        feats.append(raw_feats[group.start_end_indices.to_numpy(), :])
        labels.append(group.label.to_numpy())
        
    X = np.concatenate(feats, axis=0)
         
    y = np.concatenate(labels)
    
    # Define parameters for search and scoring strategy.
    param_grid = [{
        'C': [1, 10, 100, 1000],'penalty': ['l1', 'l2']
    }]
    
    if binary:
        def f1_score_func(y_true, y_pred):
            return f1_score(y_true, y_pred)
    else:
        def f1_score_func(y_true, y_pred):
            return f1_score(y_true, y_pred, average='macro')
    # from https://stackoverflow.com/questions/39044686/how-to-pass-argument-to-scoring-function-in-scikit-learns-logisticregressioncv
    def roc_auc_score_proba(y_true, proba):
        return roc_auc_score(y_true, proba[:, 1])
    scores = {'f1_score': make_scorer(f1_score_func),
                'acc': 'accuracy_score',
                'roc': make_scorer(roc_auc_score_proba, response_method='predict_proba'),
                'log_loss': 'log_loss'
                }
    
    # instantiate and fit grid search
    gs = GridSearchCV(LogisticRegression, param_grid,
                 scoring=scores,
                 refit='f1_score', 
                 cv=10
                 )
    
    gs.fit(X, y)

    log_cross_validation(gs.cv_results_)
    
    return gs


def train_skorch_mlp(train_data, regression_model):
    
    net_regr = NeuralNetRegressor(
        regression_model,
        max_epochs=20,
        lr=0.1,
        #     device='cuda',  # uncomment this to train with CUDA
    )
    params = {
    'lr': [0.05, 0.1],
    'module__num_units': [10, 20],
    'module__dropout': [0, 0.5],
    'optimizer__nesterov': [False, True],
    }
    gs = GridSearchCV(net_regr, params, refit=False, cv=5, scoring='mean_squared_error', verbose=2)
    
    gs.fit()
    pass    


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
    train, test = train_test_split(csv_data, test_size=0.2, random_state=42)
    
    if regression:
        
        if mlp == 'mlp':
            
            model = MLPRegressor(neural_dim, 1, hidden_dim)
        
        else:
            
            #train, test = train_test_split(csv_data, test_size=0.2, random_state=42, shuffle=True)
            model = train_regression(train, root_dir) 
            
        litModel = LitRegressor(neural_dim, 1, hidden_dim=hidden_dim)
    
    else:
        if mlp == 'mlp':
            model = MLPClassifier(neural_dim, out_dim, hidden_dim)
        
        else:
            
            model = train_logistic_regression(train, root_dir)
            
                        
        litModel = LitClassifier(model, neural_dim, out_dim, hidden_dim=hidden_dim)
        
    csv_data = pd.read_csv(csv_path)
    
    if args.probe == 'linear':
        
        train, test = train_test_split(csv_data, test_size=0.2, random_state=42, shuffle=True)
        
        model = train_regression(train, root_dir, filter_silences=True)
        test_log_regression()
    
    elif args.probe == 'mlp':
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