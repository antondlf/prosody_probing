from dataset_creation.create_dataset import BaseDataset
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, PredefinedSplit
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
from sklearn.dummy import DummyClassifier, DummyRegressor
import numpy as np
from tqdm import tqdm
from ast import literal_eval
import pickle
import traceback
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


##############################################################
##############################################################
###### Hyperparameter tuning has been set up but has #########
###### never been fully run. Instead, only one       #########
###### choice was made for LR parameters: use of     #########
###### L1 penalty. This was motivated by convergence #########
###### issues with L2 penalty. The MLP hyperparams   #########
###### were chosen arbitrarily. A parameter search   #########
###### can be run with this codebase, however.       #########
##############################################################
##############################################################


def save_model(model, pickle_path, model_type='linear'):
    if model_type != 'mlp':
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        model.save_params(f_params=pickle_path)


def log_cross_validation(scores, log_dir, threshold=None, best_params=None):
    """Assumed to be a dict output of gc.cv_results_
    or cross_validate().scores"""
    pd.DataFrame(scores).to_csv(log_dir)
    if best_params:
        with open(log_dir / 'best_params.pickle', 'w') as f:
            pickle.dump(best_params, f)
    
    if threshold:
        pd.DataFrame(threshold, names=['threshold_value', 'f1_score'])
            

def balance_classes(data, col_name='label', random_state=None):
    
    group = data.groupby(col_name)
    return group.apply(lambda x: x.sample(group.size().min(), random_seed=random_state).reset_index(drop=True)).reset_index(drop=True)


def get_phone_type(phone):
    vowels = ['ah', 'y', 'ow',  'aa',  'ae', 'ih', 'ay', 
       'eh', 'ao', 'ax',  'uw', 'v', 'f', 'er', 'ga',
        'ey', 'aw', 'iy', 'uh', 'en', 'el', 'oy']
    silence = ['SIL', 'sil']

    consonants = ['b','t', 'dh', 'n', 'd', 'l', 's', 'k', 'r', 'th',
                  'ng', 'w', 'hh', 'm', 'z', 'p', 'g', 'ch', 'jh', 'sh', 'lg','zh']
    if phone in vowels:
        return 'V'
    elif phone in consonants:
        return 'C'
    else:
        return 'sil'     


def filter_syllable(data, feature, is_onset=True):
    
    """Filters out onsets or rhymes of syllables."""
    
    if not feature.startswith('tone'):
        
        phone_reference = pd.read_csv('data/switchboard/aligned_tasks/phones.csv')
        
        phone_reference['phone_type'] = phone_reference.label.map(get_phone_type)
        phone_reference['phone_number'] = phone_reference.phone_type.map(lambda x: 1 if x == 'V' else 0)
        phone_reference['start_end_indices'] = phone_reference.start_end_indices.map(literal_eval)
        
        data['syllable_id'] = data.index.to_list()
        data['start_end_indices'] = data.start_end_indices.map(literal_eval)
        data_exploded = data.explode('start_end_indices')
        phone_exploded = phone_reference.explode('start_end_indices')
        
        merged_data = phone_exploded.merge(data_exploded, how='inner', on=['file_id', 'start_end_indices'])
        
        merged_data['onset_number'] = merged_data.groupby(['syllable_id']).phone_number.cumsum()
        merged_data['is_onset'] = merged_data.onset_number.map(lambda x: True if x == 0 else False)

        imploded_data = merged_data.groupby(['file_id', 'syllable_id', 'is_onset', 'label_x']).agg(
            {
                'start_end_indices': lambda x: x,
                'label_y': lambda y: y.unique()[0] if len(y.unique()) == 1 else y
                }).reset_index().sort_values(by='syllable_id')
        
        final_data = imploded_data[['file_id', 'speaker', 'start_end_indices', 'is_onset', 'label_y']].loc[imploded_data.is_onset == is_onset]
        final_data.columns = ['file_id', 'speaker', 'start_end_indices', 'is_onset', 'label'] 
        
    else:
        df = pd.read_csv('data/mandarin-timit/aligned_tasks/tone_rhymes.csv')
        if is_onset == False:
            final_data = data.loc[df.label != '0']
            final_data['start_end_indices'] = final_data.start_end_indices.map(literal_eval)     
        else:
            final_data = data.loc[df.label == '0']
            final_data['start_end_indices'] = final_data.start_end_indices.map(literal_eval)
    
    return final_data                 


def get_full_dataset(
    data, root_dir, regression=True,
    testing=False, average_duplicates=False,
    average=False, balance_classes=False,
    random_state=42
    ):
    
    print('Retrieving dataset...')
    feats = list()
    labels = list()
    test_df = pd.DataFrame()
    
    if balance_classes:
        balance_classes(data, col_name='label', random_state=random_state) 
        
    is_random = True if root_dir.parent.name == 'random' else False
        
    #data = data.loc[(data.label != 'sil') & (data.label != 'SIL')]
    for name, group in data.groupby('file_id'):
        feat_file = name + '.npy'
        raw_feats = np.load(root_dir / feat_file) if not is_random else np.zeros((int(group.start.iloc[-1] / 0.02 + 30), 1))
        feat_dim = raw_feats.shape[-1]
        group.dropna(inplace=True)
        if group.start_end_indices.dtype == str:
            group['start_end_indices'] = group.start_end_indices.map(literal_eval)
        elif group.start_end_indices.dtype == object:
            group['start_end_indices'] = group.start_end_indices.map(lambda x: list(x) if type(x) != int else [x])
        group['label'] = group.label.astype(np.float32)
        if group.start_end_indices.dtype == list:
            group = group.loc[group.start_end_indices.map(lambda x: len(x) > 0)]
        if average_duplicates:
            #group['start_end_indices'] = group.start_end_indices.map(lambda x: x[0])
            group = group.groupby(['file_id', 'start_end_indices', 'speaker']).mean().reset_index()
            #group.drop_duplicates(subset=['start_end_indices'], keep='first')
            
        if len(group) > 0:
            if average:
                indices = group.start_end_indices.to_numpy()
                for index_list in indices:
                    int_indices = list(map(int, index_list))
                    feats.append(raw_feats[int_indices, :].mean(axis=0).reshape(1, feat_dim))
                full_group = group
            else:
                full_group = group.explode('start_end_indices')
                indices = full_group.start_end_indices.map(int).to_numpy() 
                feats.append(raw_feats[indices, :])
            
            # Full group should be the appropriate group in either case:
            if regression:
                labels.append(full_group.label.astype(np.float32).to_numpy())
            else:
                labels.append(full_group.label.astype(int).to_numpy())
            if testing:
                test_df = pd.concat([test_df, full_group])
    try:    
        X = np.concatenate(feats, axis=0)    
        y = np.concatenate(labels)
    except ValueError:
        print(group)
        print(labels)
        print(indices)
        print(raw_feats)
        # halt code because error will be raised later
        # as a direct result of this exception
        assert True == False
    if not testing:
        return X, y
    else:
        return X, y, test_df


def train_mlp_regressor(
    train_data, module, log_dir,
    validation=False,
    criterion=torch.nn.MSELoss,
    optim=torch.optim.SGD,
    batch=128,
    random_state=42,
    input_dim=768,
    h_dim=512,
    cross_validation=True,
    best_params=True,
    tune_params=False
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
        iterator_train__num_workers=10,
        iterator_valid__num_workers=10,
        # Scoring callbacks.
        callbacks=callbacks,
        #device='cuda'
        )
    
    params = {
    'lr': [0.01, 0.003, 3e-4],
    'max_epochs': [10, 20],
    'module__hidden_dim': [128, 384],
    }
    X, y = train_data
    
    if cross_validation:
        cv = cross_validate(model, X, y, scoring=['neg_mean_squared_error', 'r2'], cv=5, verbose=2)
        log_cross_validation(cv, log_dir, best_params=None)
        model.fit(X, y)
        return model
    
    elif tune_params:
        ps = PredefinedSplit(validation.index)
        gs = GridSearchCV(model, params, refit='neg_mean_squared_error', scoring=['neg_mean_squared_error', 'r2'], cv=ps, verbose=2)
        gs.fit(X, y.reshape(-1, 1))
        
        log_cross_validation(gs.cv_results_, log_dir, best_params=gs.best_params_)
        return gs.best_estimator_

    else:
        if best_params:
            with open(log_dir.parent / 'best_params.pickle', 'r') as f:
                best_params = pickle.load(f)
            model.set_params(best_params)
        model.fit(X, y.reshape(-1, 1))
        return model


def train_mlp_classifier(train_data, module,
    log_dir,
    validation=False,
    criterion=torch.nn.CrossEntropyLoss,
    optim=torch.optim.SGD,
    batch=128,
    random_state=42,
    input_dim=768,
    h_dim=512,
    out_dim=2,
    cross_validation=True,
    best_params=True,
    tune_params=False
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
        max_epochs=50,
        batch_size=batch,
        # Training loss.
        criterion=criterion,
        #criterion__weight=weights,
        # Optimization parameters.
        optimizer=optim,
        # Parallelization.
        iterator_train__shuffle=True,
        iterator_train__num_workers=10,
        iterator_valid__num_workers=10,
        #dataset
        #dataset = train_dataset,
        # Scoring callbacks.
        callbacks=callbacks,
        #device='cuda'
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
        cv = cross_validate(model, X, y, scoring=scores, cv=5, verbose=2)
        log_cross_validation(cv, log_dir, best_params=None)
        model.fit(X, y)
        return model
    
    elif tune_params:
        ps = PredefinedSplit(validation.index)
        gs = GridSearchCV(model, params, refit='log_loss',
                          scoring=scores, cv=ps, verbose=2,
                          n_jobs=-1
                          )
        gs.fit(train_dataset.X, train_dataset.y)
        log_cross_validation(gs.cv_results_, log_dir, best_params=gs.best_params_)
        return gs.best_estimator_ 
    
    else:
        if best_params:
            with open(log_dir.parent / 'best_params.pickle', 'r') as f:
                best_params = pickle.load(f)
            model.set_params(best_params)
        model.fit(X, y)
        return model
     

def train_regression(train_data, log_dir, validation=False, cross_validation=True):

    X, y = train_data
    linear_model = LinearRegression(n_jobs=-1)
    if cross_validation:
        cv = cross_validate(linear_model, X, y, scoring=['neg_mean_squared_error', 'r2'], cv=5, verbose=2)
        log_cross_validation(cv, log_dir, best_params=None)
        
    linear_model.fit(X, y)
    
    return linear_model


def train_logistic_classifier(train_data, log_dir, binary=False, cross_validation=False, best_params=True, validation=False, tune_params=False):
    
    X, y = train_data
    
    # Define parameters for search and scoring strategy.
    param_grid = [{
        'C': [1, 10, 100, 1000]
    }]
    
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
    
    model = LogisticRegression(penalty='l1', multi_class='auto', solver='saga', n_jobs=-1, max_iter=200)
        
    if cross_validation:
        
        cv = cross_validate(model, X, y, scoring=scores, cv=5, verbose=2)
        log_cross_validation(cv.cv_results_, log_dir)
        
        model.fit(X, y)
        return model
    
    elif tune_params:
        print('Functionality for parameter tuning has not been implemented')
        pass
        # instantiate and fit grid search
        ps = PredefinedSplit(validation.index)
        gs = GridSearchCV(model, param_grid,
                    scoring=scores,
                    refit='log_loss', 
                    cv=ps, verbose=2,
                    n_jobs=-1
                    )
        
        gs.fit(X, y)
        dev_feats, dev_labels = X[validation.index, :], y[validation.index]
        
        dist = gs.best_estimator_.predict_proba(dev_feats)
        threshold_results = list()
        for threshold in [0.25, 0.5, 0.75]:
            
            Y_test_pred = dist.applymap(lambda x: 1 if x>threshold else 0)
            if gs.best_estimator_.coef_.shape[0] > 2:
                score = f1_score(dev_labels, Y_test_pred, average='macro')
            else:
                score = f1_score(dev_labels, Y_test_pred)
            
            threshold_results.append([threshold, score])
    
    else:
        if best_params:
            with open(log_dir.parent / 'best_params.pickle', 'r') as f:
                best_params = pickle.load(f)
            model.set_params(best_params)
        model.fit(X, y)
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
    ),
    parser.add_argument(
        '--balance_classes', action='store_true', default=False, help='Whether to artificially balance class counts'
    )
    parser.add_argument(
        '--mean_pooling',  action='store_true', default=False, help="Whether to average representations for a given labeled interval."
    )
    parser.add_argument(
        '--random_seed', type=int, default=42, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--onset_filtering', type=str, default='all', help="Select whether to use 'all' (the entire syllable),"\
        "'onset' (onsets only) or 'rhyme' syllable rhymes only"
    )
    
    args = parser.parse_args()
    
    """Variables
    layer: layer to probe
    root_dir: directory of probed features
    
    """
    
    mean_pooling = args.mean_pooling if (args.task != 'energy_std') & (args.task != 'f0_std') else True
    balance_classes = args.balance_classes
    seed = args.random_seed
    # Create different log dirs to keep track of conditions
    if mean_pooling and balance_classes:
       log_path = Path(f"logs/mean_balanced/{args.corpus_name}/{args.task}/{args.model}/{args.probe}") 
    elif mean_pooling:
        log_path = Path(f"logs/mean/{args.corpus_name}/{args.task}/{args.model}/{args.probe}") 
    elif balance_classes:
        log_path = Path(f"logs/balanced/{args.corpus_name}/{args.task}/{args.model}/{args.probe}")
    elif args.onset_filtering != 'all':
        log_path = Path(f"logs/{args.corpus_name}/{args.task}_{args.onset_filtering}/{args.model}/{args.probe}")
    else:
        #train_test_split
        log_path = Path(f"logs/{args.corpus_name}/{args.task}/{args.model}/{args.probe}")
    os.makedirs(log_path, exist_ok=True)
    
    csv_data = pd.read_csv(args.labels)
            
    if args.task.startswith('stress'):
        stress_category_mapping = {'p': 1, 'n': 0, 's': 0}
        csv_data['label'] = csv_data.label.map(lambda x: stress_category_mapping.get(x, 'SIL'))
    
    if args.corpus_name == 'yemba':
        if args.task == 'tone':
            yemba_tone_mappings = {'bas': 0, 'moyen': 1, 'haut': 2}
            csv_data['label'] = csv_data.label.map(lambda x: yemba_tone_mappings[x])
            
        csv_data['speaker'] = csv_data.file_id.map(lambda x: x.split('_')[1])
    elif args.corpus_name == 'bu_radio':
        csv_data['speaker'] = csv_data.file_id.map(lambda x: x[:2])
    else:
        csv_data['speaker'] = csv_data.file_id.map(lambda x: x.split('_')[0])
    if (args.task.endswith('f0')) and (args.task != 'tone') and (args.task != 'energy'):
        csv_data = csv_data.loc[(csv_data.label != 'sil') & (csv_data.label != 'SIL')]
    elif args.task == 'stress':
        csv_data = csv_data.loc[(csv_data.label != 'sil') & (csv_data.label != 'SIL')]

    else:
        csv_data = csv_data.loc[(csv_data.label != 0)&(csv_data.label != 'sil')& (csv_data.label != 'SIL')&(csv_data.label.notna())]

    if args.task == 'tone':
        csv_data['label'] = csv_data.label.map(lambda x: int(x)-1)
        
    
    #for layer in range(0, args.layer +1):
    layer = int(args.layer)
    # Only cross validate on 0th, middle and second to last layer.
    if layer in [0, args.layer / 2, args.layer-1]:
        #cross_validation = True
        tune_params = True
    else:
        tune_params = False
        
    ####################
    # In case training
    # needs to be sped up
    # set to False
    cross_validation = True
    # The next three 
    # booleans are for 
    # an undeveloped 
    # functionality
    tune_params = False #True
    validation_split = False
    best_params=False
    ####################      
    
    full_train_speakers, test_speakers = train_test_split(csv_data.speaker.sort_values().unique(), test_size=0.2, random_state=seed) #stratify='file_id', but I want to split file ids
    if validation_split:
        train_speakers, dev_speakers = train_test_split(full_train_speakers.speaker.sort_values().unique(), test_size=0.2, random_state=seed)
        dev = csv_data.loc[csv_data.speaker.isin(dev_speakers)]
    else:
        dev = None
    
    train = csv_data.loc[csv_data.speaker.isin(full_train_speakers)]
    test = csv_data.loc[csv_data.speaker.isin(test_speakers)]
    
    if args.onset_filtering != 'all':
        if args.onset_filtering == 'onset':
            train = filter_syllable(train, args.task, is_onset=True)
        else:
            train = filter_syllable(train, args.task, is_onset=False) 

    neural_dim = args.neural_dim
    
    average_duplicates = False
    if args.task in ['f0', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth', 'f0_std', 'energy_std', 'f0_diff', 'energy_diff']:
        out_dim = 1
        regression = True
        average_duplicates = True
        # No classes in regression
        balance_classes = False
        
    elif args.task == 'tone':
        out_dim = 5
        regression = False
        
    else:
        out_dim = len(csv_data.label.unique())
        regression = False
        
    if args.model in [
        'wav2vec2-large', 'wav2vec2-xls-r-300m',
        'wav2vec2-large-960h', 'wav2vec2-large-xlsr-53',
        'wav2vec2-large-xlsr-53-chinese-zh-cn'
        ]:
        neural_dim = 1024
    hidden_dim = args.hidden_dim
    print(f'Probing all {args.layer} layers of {args.model}')

        
    root_dir = args.root_dir / args.model / f'layer-{layer}'

    train_set = get_full_dataset(
            train, root_dir, 
            regression=regression,
            average_duplicates=average_duplicates,
            average=mean_pooling,
            balance_classes=balance_classes
            )

    cv_log_dir = log_path / f"cross_val_results_{layer}.csv"
    print('\nStarting training loop...')
    
    if args.model == 'random':
        model =  DummyRegressor(strategy='mean') if regression else DummyClassifier(strategy='stratified', random_state=seed)
        binary = True if (args.task != 'tone') and (not regression) else False
        X, y = train_set
        model.fit(X, y)
    elif args.probe == 'mlp':
        if regression:
            model = train_mlp_regressor(train_set, MLPRegressor, cv_log_dir, validation=dev,
                                        h_dim=hidden_dim,
                                        tune_params=tune_params,
                                        cross_validation=cross_validation, batch=args.batch_size,
                                        input_dim=neural_dim, best_params=best_params,
                                        random_state=seed
                                        )
        else:
            binary = True if args.task != 'tone' else False
            model = train_mlp_classifier(train_set, MLPClassifier, cv_log_dir, validation=dev,
                                            h_dim=hidden_dim,
                                            tune_params=tune_params, 
                                            out_dim=out_dim, cross_validation=cross_validation,
                                            batch=args.batch_size, input_dim=neural_dim, best_params=best_params,
                                            random_state=seed
                                            )
    else:
        if regression:
            #LogisticRegression()
            model = train_regression(train_set, cv_log_dir, validation=dev)
        else:
            binary = True if args.task != 'tone' else False
            model = train_logistic_classifier(train_set, cv_log_dir, binary=binary, validation=dev, best_params=best_params, tune_params=tune_params)
    ###################
    # save_model object
    save_model(model, log_path / f'layer_{layer}_model.pickle', model_type=args.probe)
    ###################
    print('Done training!')
    print('Beginning test loop...')
    test_feats, test_labels, test_df = get_full_dataset(
        test, root_dir, regression=regression,
        testing=True, average_duplicates=average_duplicates,
        average=mean_pooling
        )
    y_pred = model.predict(test_feats)
    if args.probe == 'mlp':
        y_pred = y_pred.flatten()

    print()
    print('Testing done, outputting results...')        
    print(f"Results for {args.model} on {args.task} with {args.probe} on layer {layer}")
    if not regression:
        f1_result = f1_score(test_labels, y_pred) if binary else f1_score(test_labels, y_pred, average='micro'),
        precision_result = precision_score(test_labels, y_pred) if binary else precision_score(test_labels, y_pred, average='micro')
        recall_result = recall_score(test_labels, y_pred) if binary else recall_score(test_labels, y_pred, average='micro')
        print("layer\tcorpus\tf1_score\taccuracy\tprecision\trecall")
        print(f"{layer}\t{args.corpus_name}\t{f1_result}\t{accuracy_score(test_labels,y_pred)}\t\
            {precision_result}\t{recall_result}")
        y_pred_proba = model.predict_proba(test_feats)
        classes = model.classes_
        result_dict = {
            'file_id': test_df.file_id, 'neural_index': test_df.start_end_indices,
            'y_true': test_labels, 'y_pred': y_pred,
            #'y_proba': y_pred_proba,
            #'classes': [classes]*len(test_df)
            }
        with open(log_path / f'layer-{layer}_distribution.txt', 'w') as f:
            for c in classes:
                f.write(f"{str(c)}\t")
            f.write('\n')
            for prob in y_pred_proba:
                for p in prob:
                    f.write(f"{str(p)}\t")
                f.write('\n')
                    
        
    else:
        print("layer\tcorpus\tmse\tr2")
        print(f"{layer}\t{args.corpus_name}\t{mean_squared_error(test_labels, y_pred)}\{r2_score(test_labels, y_pred)}")
        result_dict = {
        'file_id': test_df.file_id, 'neural_index': test_df.start_end_indices, 'y_pred': y_pred, 'y_true': test_labels
        }
    print('Outputting result dataframe...')
    print()
    try:
        results = pd.DataFrame(result_dict)

        results.to_csv(f"{log_path}/layer_{layer}.csv")  
    except ValueError:
        print(traceback.format_exc())
        for k, v in result_dict.items():
            print(f"{k}: {len(v)}")
    
if __name__ == '__main__':
    main()   
