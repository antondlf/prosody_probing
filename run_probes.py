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
from skorch.dataset import Dataset
from skorch.callbacks import EarlyStopping, GradientNormClipping
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss,\
roc_auc_score, make_scorer, mean_squared_error, r2_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
from ast import literal_eval
import pickle


def log_cross_validation(scores, log_dir, best_params=None):
    """Assumed to be a dict output of gc.cv_results_
    or cross_validate().scores"""
    pd.DataFrame(scores).to_csv(log_dir)
    if best_params:
        with open(log_dir / 'best_params.pickle', 'w') as f:
            pickle.dump(best_params, f)


def get_full_dataset(data, root_dir, regression=True, average_feats=False):
    
    print('Retrieving dataset...')
    feats = list()
    labels = list()
    #data = data.loc[(data.label != 'sil') & (data.label != 'SIL')]
    for name, group in tqdm(list(data.groupby('file_id'))):
        feat_file = name + '.npy'
        raw_feats = np.load(root_dir / feat_file)
        group.dropna(inplace=True)
        group['start_end_indices'] = group.start_end_indices.map(literal_eval)
        full_group = group.explode('start_end_indices')
        indices = full_group.start_end_indices.map(int).to_numpy()            
        feats.append(raw_feats[indices, :])
        if regression:
            labels.append(full_group.label.astype(np.float32).to_numpy())
        else:
            labels.append(full_group.label.astype(int).to_numpy())
        
    X = np.concatenate(feats, axis=0)    
    y = np.concatenate(labels)
    
    return X, y


def train_mlp_regressor(
    train_data, module, log_dir,
    criterion=torch.nn.MSELoss,
    optim=torch.optim.SGD,
    batch=128,
    random_state=42,
    input_dim=768,
    h_dim=128,
    cross_validation=True
    ):
    torch.cuda.empty_cache()
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
        device='cuda'
        )
    
    params = {
    'lr': [0.01, 0.003, 3e-4],
    'max_epochs': [10, 20],
    'module__hidden_dim': [128, 384],
    }
    X, y = train_data
    if cross_validation:
        gs = GridSearchCV(model, params, refit='neg_mean_squared_error', scoring=['neg_mean_squared_error', 'r2'], cv=3, verbose=2)
        gs.fit(X, y.reshape(-1, 1))
        
        log_cross_validation(gs.cv_results_, log_dir, best_params=gs.best_params_)
        return gs.best_estimator_

    else:
        with open(log_dir / 'best_params.pickle', 'r') as f:
            best_params = pickle.load(f)
        model.set_params(best_params)
        model.fit(X, y.reshape(-1, 1))
        return model


def train_mlp_classifier(train_data, module,
    log_dir,
    criterion=torch.nn.CrossEntropyLoss,
    optim=torch.optim.SGD,
    batch=128,
    random_state=42,
    input_dim=768,
    h_dim=128,
    out_dim=2,
    cross_validation=True
    ):
    torch.cuda.empty_cache()
    torch.manual_seed(random_state)
    callbacks = []

    # Clip gradients to L2-norm of 2.0
    callbacks.append(
        ('clipping', GradientNormClipping(2.0)))

    # Allow early stopping.
    callbacks.append(
        ('EarlyStop', EarlyStopping(monitor='valid_loss')))
    X, y = train_data
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    
    #train_dataset = Dataset(X_train, y=y_train)
    #valid_dataset = Dataset(X_valid, y=y_valid)
    train_dataset = Dataset(X, y=y)
    # Instantiate our classifier.
    model = NeuralNetClassifier(
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
        #dataset
        dataset = train_dataset,
        # Scoring callbacks.
        callbacks=callbacks,
        device='cuda'
        )
    if out_dim==2:
        
        scores = {'f1_score': 'f1',
            'acc': 'accuracy',
            'roc': 'roc_auc',
            'log_loss': 'neg_log_loss'
            }
    else:
        scores = {'f1_score': 'f1_micro',
                  'f1_macro': 'f1_macro',
            'acc': 'accuracy',
            #'roc': 'roc_auc', not supported for multi-class classification
            'log_loss': 'neg_log_loss'
            }
    params = {
    'lr': [0.01, 0.003, 3e-4],
    'max_epochs': [10, 20],
    'module__hidden_dim': [128, 384],
    }

    if cross_validation:
        gs = GridSearchCV(model, params, refit='log_loss',
                          scoring=scores, cv=3, verbose=2,
                          n_jobs=-1
                          )
        gs.fit(train_dataset.X, train_dataset.y)
        log_cross_validation(gs.cv_results_, log_dir, best_params=gs.best_params_)
        return gs.best_estimator_ 
    
    else:
        with open(log_dir / 'best_params.pickle', 'r') as f:
            best_params = pickle.load(f)
        model.set_params(best_params)
        return model
     

def train_regression(train_data, log_dir):

    X, y = train_data
    linear_model = LinearRegression(n_jobs=-1)
    cv = cross_validate(linear_model, X, y, scoring=['neg_mean_squared_error', 'r2'], cv=3, n_jobs=-1, verbose=2)
    log_cross_validation(cv, log_dir, best_params=None)
    
    linear_model.fit(X, y)
    
    return linear_model


def train_logistic_classifier(train_data, log_dir, binary=False, cross_validation=True):
    
    X, y = train_data
    
    # Define parameters for search and scoring strategy.
    param_grid = [{
        'C': [1, 10, 100, 1000],'penalty': ['l2'], 'max_iter': [200]
    },
    {
        'C': [1, 10, 100, 1000], 'penalty': ['l1'], 'solver': ['liblinear']
    }
                  ]
    
    if binary:
        
        scores = {'f1_score': 'f1',
            'acc': 'accuracy',
            'roc': 'roc_auc',
            'log_loss': 'neg_log_loss'
            }
    else:
        scores = {'f1_score': 'f1_micro',
                  'f1_macro': 'f1_macro',
            'acc': 'accuracy',
            #'roc': 'roc_auc', not supported for multi class classification
            # There are some workarounds but I am not implementing any.
            'log_loss': 'neg_log_loss'}
    model = LogisticRegression(n_jobs=-1)
    if cross_validation:
        # instantiate and fit grid search
        gs = GridSearchCV(model, param_grid,
                    scoring=scores,
                    refit='log_loss', 
                    cv=3, verbose=2,
                    n_jobs=-1
                    )
        
        gs.fit(X, y)

        log_cross_validation(gs.cv_results_, log_dir, best_params=gs.best_params_)
        
        return gs.best_estimator_
    
    else:
        with open(log_dir / 'best_params.pickle', 'r') as f:
            best_params = pickle.load(f)
        model.set_params(best_params)
        return model        


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
        '-b', '--batch_size', type=int, default=128, help='batch size for mlp training'
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
    os.makedirs(log_path, exist_ok=True)
    
    csv_data = pd.read_csv(args.labels)
    train, test = train_test_split(csv_data, test_size=0.2, random_state=42)
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
    print(f'Probing all {args.layer} layers of {args.model}')
    for layer in range(0, args.layer +1):
        
        # Only cross validate on 0th, middle and second to last layer.
        if layer in [0, args.layer / 2, args.layer-1]:
            cross_validation = True
        else:
            cross_validation = False
            
        root_dir = args.root_dir / args.model / f'layer-{layer}'
        if (args.task != 'f0') and(args.task != 'tone'):
            csv_data = csv_data.loc[(csv_data.label != 'sil') & (csv_data.label != 'SIL')]
        else:
            csv_data = csv_data.loc[(csv_data.label != 0)&(csv_data.label != 'sil')& (csv_data.label != 'SIL')]
            
        train_set = get_full_dataset(train, root_dir, regression=regression)
        cv_log_dir = log_path / f"cross_val_results_{layer}.csv"
        print('Starting training loop...') 
        if args.probe == 'mlp':
            if regression:
                model = train_mlp_regressor(train_set, MLPRegressor, cv_log_dir, cross_validation=cross_validation, batch=args.batch_size)
            else:
                model = train_mlp_classifier(train_set, MLPClassifier, cv_log_dir, out_dim=out_dim, cross_validation=cross_validation, batch=args.batch_size)
        else:
            if regression:
                #LogisticRegression()
                model = train_regression(train_set, cv_log_dir, cross_validation=cross_validation)
            else:
                binary = True if args.task != 'tone' else False
                model = train_logistic_classifier(train_set, cv_log_dir, binary=binary, cross_validation=cross_validation)
        
        print('Done training!')
        print('Beginning test loop...')
        test_feats, test_labels = get_full_dataset(test, root_dir)
        y_pred = model.predict(test_feats)

        print()
        print('Testing done, outputting results...')        
        print(f"Results for {args.model} on {args.task} with {args.probe} on layer {layer}")
        if not regression:
            print("layer\tcorpus\tf1_score\taccuracy\tprecision\trecall")
            print(f"{layer}\t{args.corpus_name}\t{f1_score(test_labels, y_pred)}\t{accuracy_score(test_labels,y_pred)}\t\
                {precision_score(test_labels, y_pred)}\t{recall_score(test_labels, y_pred)}")
            y_pred_proba = model.predict_proba(test_feats)
            classes = model.classes_
            result_dict = {'file_id': test.file_id, 'neural_index': test.start_end_indices, 'y_true': test_labels, 'y_pred': y_pred, 'y_proba': (classes, y_pred_proba)}
            
        else:
            print("layer\tcorpus\tmse\tr2")
            print(f"{layer}\t{args.corpus_name}\t{mean_squared_error(test_labels, y_pred)}\{r2_score(test_labels, y_pred)}")
            result_dict = {
            'file_id': test.file_id, 'neural_index': test.start_end_indices, 'y_pred': y_pred, 'y_true': test_labels
            }
        print('Outputting result dataframe...')
        print()
        results = pd.DataFrame(result_dict)
    
        results.to_csv(f"{log_path}/layer_{layer}.csv")  
    
if __name__ == '__main__':
    main()   