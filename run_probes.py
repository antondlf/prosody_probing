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
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch import loggers as pl_loggers
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.callbacks import EarlyStopping, GradientNormClipping
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss,\
roc_auc_score, make_scorer, mean_squared_error, r2_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm


def log_cross_validation(scores, log_dir):
    """Assumed to be a dict output of gc.cv_results_
    or cross_validate().scores"""
    pd.DataFrame(scores).to_csv(log_dir)
    pass


def get_full_dataset(data, root_dir):
    
    print('Retrieving dataset...')
    feats = list()
    labels = list()
    for name, group in tqdm(list(data.groupby('file_id'))):
        feat_file = name + '.npy'
        raw_feats = np.load(root_dir / feat_file)
        
        feats.append(raw_feats[group.start_end_indices.to_numpy(), :])
        labels.append(group.label.to_numpy())
        
    X = np.concatenate(feats, axis=0)    
    y = np.concatenate(labels)
    
    return X, y


def train_mlp_regressor(
    train_data, module, log_dir,
    criterion=torch.nn.MSELoss,
    optim=torch.optim.SGD,
    batch=625,
    random_state=42,
    input_dim=768,
    h_dim=128,
    
    ):
    
    torch.manual_seed(random_state)
    callbacks = []

    # Clip gradients to L2-norm of 2.0
    callbacks.append(
        ('clipping', GradientNormClipping(2.0)))

    # Allow early stopping.
    callbacks.append(
        ('EarlyStop', EarlyStopping()))

    # Instantiate our classifier.
    model = NeuralNetRegressor(
        # Network parameters.
        module, module__out_dim=1, module__hidden_dim=h_dim,
        module__in_dim=input_dim,
        # Training batch/time/etc.
        max_epochs=50, batch_size=batch,
        # Training loss.
        criterion=criterion,
        #criterion__weight=weights,
        # Optimization parameters.
        optimizer=optim,
        # Parallelization.
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__num_workers=4,
        # Scoring callbacks.
        callbacks=callbacks,
        train_split=False,
        device='cuda'
        )
    
    params = {
    'lr': [0.01, 0.003, 3e-4],
    'max_epochs': [10, 20],
    'module__h_dim': [128, 384, 512],
    }

    gs = GridSearchCV(model, params, refit=False, scoring=['mean_squared_error', 'r2_score'], cv=3)
    gs.fit(train_data[0], train_data[1])
    
    log_cross_validation(gs.cv_results_, log_dir)
    return gs.best_estimator_

def train_mlp_classifier(train_data, module,
    log_dir,
    criterion=torch.nn.CrossEntropyLoss,
    optim=torch.optim.SGD,
    batch=625,
    random_state=42,
    input_dim=768,
    h_dim=128,
    out_dim=2,
    ):
    
    torch.manual_seed(random_state)
    callbacks = []

    # Clip gradients to L2-norm of 2.0
    callbacks.append(
        ('clipping', GradientNormClipping(2.0)))

    # Allow early stopping.
    callbacks.append(
        ('EarlyStop', EarlyStopping()))

    # Instantiate our classifier.
    model = NeuralNetRegressor(
        # Network parameters.
        module, module__out_dim=out_dim, module__hidden_dim=h_dim,
        module__in_dim=input_dim,
        # Training batch/time/etc.
        max_epochs=50, batch_size=batch,
        # Training loss.
        criterion=criterion,
        #criterion__weight=weights,
        # Optimization parameters.
        optimizer=optim,
        # Parallelization.
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__num_workers=4,
        # Scoring callbacks.
        callbacks=callbacks,
        train_split=False,
        device='cuda'
        )
    scores = {'f1_score': make_scorer(f1_score_func),
        'acc': 'accuracy_score',
        'roc': make_scorer(roc_auc_score_proba, response_method='predict_proba'),
        'log_loss': 'log_loss'
        }
    params = {
    'lr': [0.01, 0.003, 3e-4],
    'max_epochs': [10, 20],
    'module__h_dim': [128, 384, 512],
    }

    gs = GridSearchCV(model, params, refit=False, scoring=scores, cv=3)
    gs.fit(train_data[0], train_data[1])
    
    log_cross_validation(gs.cv_results_, log_dir) 
    return gs.best_estimator_

def train_regression(train_data, log_dir):

    X, y = train_data
    linear_model = LinearRegression()
    cv = cross_validate(linear_model, X, y, scoring=['mean_squared_error', 'r2_score'], cv=10, n_jobs=-1)
    log_cross_validation(cv.scores, log_dir)
    
    linear_model.fit(X, y)
    
    return linear_model


def train_logistic_classifier(train_data, log_dir, binary=False):
    
    X, y = train_data
    
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

    log_cross_validation(gs.cv_results_, log_dir)
    
    return gs.best_estimator_


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
    log_path = Path(f"logs/{args.corpus_name}/{args.task}/{args.probe}")
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
    csv_data = pd.read_csv(args.labels)
    train, test = train_test_split(csv_data, test_size=0.2, random_state=42)
    train_set = get_full_dataset(train, root_dir)
    cv_log_dir = log_path / f"cross_val_results.csv"
    print('Starting training loop...') 
    if mlp == 'mlp':
        if regression:
            model = train_mlp_regressor(train_set, MLPRegressor, cv_log_dir, out_dim=out_dim)
        else:
            model = train_mlp_classifier(train_set, MLPClassifier, cv_log_dir, out_dim=out_dim)
    else:
        if regression:
            model = train_regression(train_set, cv_log_dir)
        else:
            model = train_logistic_classifier(train_set, cv_log_dir)
    
    print('Done training!')
    print('Beginning test loop...')
    test_feats, test_labels = get_full_dataset(test, root_dir)
    y_pred = model.predict(test_feats)
    
    print()
    print('Testing done, outputting results...')
    results = pd.DataFrame({
        'file_id': test.file_id, 'neural_index': test.start_end_indices, 'y_pred': y_pred, 'y_true': test_labels
    })
   
    results.to_csv(f"{log_path}/layer_{args.layer}.csv") 
    
    print(f"Results for {args.model} on {args.task} with {args.probe} on layer {args.layer}")
    if not regression:
        print("layer\tcorpus\tf1_score\taccuracy\tprecision\trecall")
        print(f"{args.layer}\t{args.corpus}\t{f1_score(test_labels, y_pred)}\t{accuracy_score(test_labels,y_pred)}\t\
            {precision_score(test_labels, y_pred)}\t{recall_score(test_labels, y_pred)}")
        
    else:
        print("layer\tcorpus\tmse\tr2")
        print(f"{args.layer}\t{args.corpus}\t{mean_squared_error(test_labels, y_pred)}\{r2_score(test_labels, y_pred)}")
        
    print() 
    
    
if __name__ == '__main__':
    main()   