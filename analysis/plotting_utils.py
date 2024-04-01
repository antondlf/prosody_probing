import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, log_loss, r2_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import math
import os
from scipy.stats import normaltest
from confidence_intervals import evaluate_with_conf_int


def get_metric(name):
    
    mapping = {
        'f1_score': f1_score,
        'f1_micro': f1_score,
        'f1_macro': f1_score,
        'mean_squared_error': mean_squared_error,
        'roc_auc': roc_auc_score,
        'log_loss': log_loss,
        'pearson': pearsonr,
        'spearman': spearmanr,
        'accuracy': accuracy_score,
        'R2': r2_score,
        'normality': normaltest,
        'recall': recall_score,
        'precision': precision_score,
        'recall_macro': recall_score,
        'precision_macro': precision_score,
        'recall_micro': recall_score,
        'precision_micro': precision_score
    }
    
    return mapping[name]


def bootstrap_eval(data, eval_func, fraction2sample=0.5, n_iterations=1000, average=None):
    
    sample_size = int(len(data) * fraction2sample)
    scores = list()
    for i in range(n_iterations):
        iter_sample = resample(data, n_samples = sample_size)
        if average is not None:
            scores.append(eval_func(iter_sample.y_true, iter_sample.y_pred, average=average))
        else:
            scores.append(eval_func(iter_sample.y_true, iter_sample.y_pred))
    
    return scores, sample_size


def get_confidence_interval(true_score, distribution, N, ci=0.95):
    factor = ci*(np.std(distribution)/math.sqrt(N))
    return (true_score-factor, true_score+factor)


def get_bootstrap_ci(true_score, data, eval_func, fraction2sample=0.5, n_iterations=1000, average=None, ci=0.95):
    
    
    scores, sample_size = bootstrap_eval(data, eval_func, fraction2sample=fraction2sample, n_iterations=n_iterations, average=average)
    
    return get_confidence_interval(true_score, scores, sample_size, ci=ci)


def calculate_metric(df, metric_name, task=None, ci=False):
    
    
    if ci:
        return evaluate_with_conf_int(df.y_pred, metric_name, df.y_true, num_bootstraps=500, alpha=5)
    else:
        metric = get_metric(metric_name)
        if 'y_proba' in df.columns:
            
            if metric_name in ['log_loss', 'roc_auc']:
                probabilities = df.y_proba.map(lambda x: x[1]).to_numpy()
                return metric(df.y_true, probabilities)
            
        else:
            if metric_name.endswith('micro'):
                return metric(df.y_true, df.y_pred, average='micro')
            elif metric_name.endswith('macro'):
                return metric(df.y_true, df.y_pred, average='macro')

            elif metric_name == 'pearson':
                return metric(df.y_true, df.y_pred)[0]
            elif metric_name == 'normality':
                statistic, p_value = metric(df.y_true - df.y_pred, nan_policy='omit')
                return (statistic, p_value)
            elif (task == 'stress') and (metric_name == 'f1_score'):
                target_label = df.y_true.unique().min()
                return metric(df.y_true, df.y_pred, pos_label=target_label)
            else:
                return metric(df.y_true, df.y_pred)
        
        
def loop_model_result_dir(task_dir, layer_number, metric_name):
    
    results = list()
    
    for layer in range(0, layer_number+1):
        results.append(calculate_metric(pd.read_csv(task_dir / f"layer_{layer}.csv"), metric_name))
        
    return list(range(0, layer_number+1)), results


def get_model_comparison(log_root, model_list, corpus, task, probe):
    
    df_list = list()
    
    if task == 'f0':
        metric = 'pearson'
    elif task == 'tone':
        metric = 'f1_micro'
    else:
        metric = 'f1_score'
    
    for model, model_size in model_list:
        
        
        task_log_dir = log_root / model / probe
        layers, results = loop_model_result_dir(task_log_dir, model_size, metric)
        
        df_list.append(pd.DataFrame({'model': [model]*len(layers), 'layer': layers, 'score': results}))
        
    return pd.concat(df_list)


def get_log_loss(result_file, result_df, metric):
    
    binarizer = LabelBinarizer()
    layer_num = result_file.stem.split('_')[1]
    
    dist_path = result_file.parent / f'layer-{layer_num}_distribution.txt'
    distribution_df = pd.read_csv(dist_path, sep='\t')
    # There must be a tab before \n in files because an extra column is being read
    distributions = distribution_df.to_numpy()[:, :-1]
    y_true = binarizer.fit_transform(result_df.y_true)
    return metric(y_true, distributions)#, labels=distribution_df.columns[:-1])


def get_roc_auc(result_file, result_df, multiclass=True):
    
    layer_num = result_file.stem.split('_')[1]
    
    dist_path = result_file.parent / f'layer-{layer_num}_distribution.txt'
    distribution_df = pd.read_csv(dist_path, sep='\t')
    # There must be a tab before \n in files because an extra column is being read
    if not multiclass:
        distributions = distribution_df.to_numpy()[:, 1]
        return roc_auc_score(result_df.y_true, distributions) 

    else:
        distributions = distribution_df.to_numpy()[:, :-1]
        return roc_auc_score(result_df.y_true, distributions, multi_class='ovr')


def get_single_metric(task, tsk, metric_mapping, corp, multiclass=False, ci=False):
    
    metric = get_metric(metric_mapping[tsk])
    row_list = list()
    for model in task.glob('*'):
        if model.name not in ['linear', 'mlp']:
            
            mod = model.stem
            for probe in model.glob('*'):
                
                prob = probe.stem
                print(f'Calculating {metric_mapping[tsk]} for {tsk} on {mod} and {corp} with {prob} probe...')                    
                for layer in tqdm(list(probe.glob('layer_*.csv'))):
                    iter_df = pd.read_csv(layer)
                    if ci:
                        result = calculate_metric(iter_df, metric_mapping[tsk], task=tsk, ci=ci)
                        
                    elif metric_mapping[tsk] == 'f1_micro':
                        result = metric(iter_df.y_true, iter_df.y_pred, average='micro')
                        #confidence_interval = get_bootstrap_ci(result, iter_df, metric, average='micro')  
                    elif metric_mapping[tsk] == 'mean_squared_error':
                        result = metric(iter_df.y_true, iter_df.y_pred, squared=False) 
                        error_distribution = np.absolute(iter_df.y_true.to_numpy() - iter_df.y_pred.to_numpy())
                        #confidence_interval = get_confidence_interval(result, error_distribution, len(error_distribution)) 
                    elif metric_mapping[tsk] == 'f1_macro':
                        result = metric(iter_df.y_true, iter_df.y_pred, average='macro')
                        #confidence_interval = get_bootstrap_ci(result, iter_df, metric, average='macro')
                    elif metric_mapping[tsk] == 'log_loss':
                        result = get_log_loss(layer, iter_df, metric)
                        #confidence_interval = (np.nan, np.nan)#get_bootstrap_ci(result, iter_df, metric) 
                    elif metric_mapping[tsk] == 'roc_auc':
                        result = get_roc_auc(layer, iter_df, multiclass=multiclass)
                    #elif metric_mapping[tsk] == 'normality':
                    #    result = metric(iter_df.y_true - iter_df.y_pred)
                    else: 
                        result = calculate_metric(iter_df, metric_mapping[tsk], task=tsk)
                        #confidence_interval = get_bootstrap_ci(result, iter_df, metric) 
                    #if confidence_interval:
                    #    row_list.append({
                    #        'corpus': corp,
                    #        'model': mod,
                    #        'task': tsk,
                    #        'probe': prob,
                    #        'layer': int(layer.stem.split('_')[1]),
                    #        'score': result,
                            #'ci': confidence_interval
                    #    })
                    #else:
                    if ci:
                        row_list.append({
                            'corpus': corp,
                            'task': tsk,
                            'model': mod,
                            'probe': prob,
                            'layer': int(layer.stem.split('_')[1]),
                            'metric': metric_mapping[tsk].split('_')[0],
                            'score': result[0],
                            'ci': result[1],
                            'ci_high': result[1][0],
                            'ci_low': result[1][1] 
                            })
                    else:
                        row_list.append({
                            'corpus': corp,
                            'task': tsk,
                            'model': mod,
                            'probe': prob,
                            'layer': int(layer.stem.split('_')[1]),
                            'metric': metric_mapping[tsk].split('_')[0],
                            'score': result,
                            }) 
    return row_list


def get_metric_list(task, tsk, metric_mapping, corp, multiclass, ci=False):
    
    metric_list = metric_mapping[tsk]
    row_list = list()
    print(f'Calculating {tsk} for {corp}')
    for model in tqdm(list(task.glob('*'))):
        #if model.name not in ['random', 'fbank', 'mfcc']:
        #    continue
        if model.name not in ['linear', 'mlp']:
            mod = model.stem
            for probe in model.glob('*'):
                prob = probe.stem
                #print(f'Calculating {metric_mapping[tsk]} for {tsk} on {mod} and {corp} with {prob} probe...')                    
                for layer in probe.glob('layer_*.csv'): #tqdm(list(probe.glob('layer_*.csv'))):

                    iter_df = pd.read_csv(layer)
                    for metric_name in metric_list:
                        metric = get_metric(metric_name)
                        if metric_mapping[tsk] == 'f1_micro':
                            result = metric(iter_df.y_true, iter_df.y_pred, average='micro')
                            #confidence_interval = get_bootstrap_ci(result, iter_df, metric, average='micro')  
                        elif metric_mapping[tsk] == 'mean_squared_error':
                            result = metric(iter_df.y_true, iter_df.y_pred, squared=False) 
                            error_distribution = np.absolute(iter_df.y_true.to_numpy() - iter_df.y_pred.to_numpy())
                            #confidence_interval = get_confidence_interval(result, error_distribution, len(error_distribution)) 
                        elif metric_mapping[tsk] == 'f1_macro':
                            result = metric(iter_df.y_true, iter_df.y_pred, average='macro')
                            #confidence_interval = get_bootstrap_ci(result, iter_df, metric, average='macro')
                        elif metric_mapping[tsk] == 'log_loss':
                            result = get_log_loss(layer, iter_df, metric)
                            #confidence_interval = (np.nan, np.nan)#get_bootstrap_ci(result, iter_df, metric) 
                        elif metric_mapping[tsk] == 'roc_auc':
                            result = get_roc_auc(layer, iter_df, multiclass=multiclass)
                        elif metric_mapping[tsk] == 'normality':
                            result = metric(iter_df.y_true - iter_df.y_pred)
                        else:

                            result = calculate_metric(iter_df, metric_name, task=tsk)

                        row_list.append({
                            'corpus': corp,
                            'task': tsk,
                            'model': mod,
                            'probe': prob,
                            'layer': int(layer.stem.split('_')[1]),
                            'metric': metric_name.split('_')[0],
                            'score': result
                            #'ci': (None, None)
                            })
    return row_list


def get_result_df(log_root, metric_mapping, save_path, ci=False):
    """Generates a dataframe of all dfs and saves to
    csv.
    –––––––––––––––––––––––––––––––––––––––––––––––––
    log_root: path to top log directory.
    metrics: dictionary mapping task to metric function.
    """
    
    row_list = list()
    
    for corpus in log_root.glob('*'):

        corp = corpus.stem
        if corp == 'mean':
            continue
        
        for task in corpus.glob('*'):
            tsk = task.stem
            multiclass = True if tsk == 'tone' else False
            if type(metric_mapping[tsk]) == list:
                row_list.extend(get_metric_list(task, tsk, metric_mapping, corp, multiclass, ci=ci))
            else:
                row_list.extend(get_single_metric(task, tsk, metric_mapping, corp, multiclass=multiclass, ci=ci))

                                
    result_df = pd.DataFrame(row_list)
    names_for_plotting = {
        'wav2vec2-base': 'English Base',
        'mandarin-wav2vec2': 'Mandarin Base',
        'mandarin-wav2vec2-aishell1': 'Mandarin Finetuned',
        'wav2vec2-base-100h': 'English Finetuned',
        'wav2vec2-large': 'English Large',
        'wav2vec2-large-960h': 'English Large Finetunted',
        'wav2vec2-xls-r-300m': 'Multilingual Large 128',
        'wav2vec2-large-xlsr-53': 'Multilingual Large 53',
        'wav2vec2-large-xlsr-53-chinese-zh-cn': 'Multilingual Large 53 Finetuned',
        'hubert-base-ls960': 'HuBert',
        'wavlm-base': 'WavLM',
        'fbank': 'Mel-Filterbank',
        'mfcc': 'MFCC',
        'random': 'Random Baseline'
    }
    
    model_language = {
        'wav2vec2-base': 'English',
        'mandarin-wav2vec2': 'Mandarin',
        'mandarin-wav2vec2-aishell1': 'Mandarin',
        'wav2vec2-base-100h': 'English',
        'wav2vec2-large': 'English',
        'wav2vec2-large-960h': 'English',
        'wav2vec2-xls-r-300m': 'Multilingual',
        'wav2vec2-large-xlsr-53': 'Multilingual',
        'wav2vec2-large-xlsr-53-chinese-zh-cn': 'Multilingual Mandarin',
        'hubert-base-ls960': 'English',
        'wavlm-base': 'English',
        'fbank': 'Baseline',
        'mfcc': 'Baseline',
        'random': 'Baseline'
    }
    model_state = {
        'wav2vec2-base': 'Pre-trained',
        'mandarin-wav2vec2': 'Pre-trained',
        'mandarin-wav2vec2-aishell1': 'Fine-tuned',
        'wav2vec2-base-100h': 'Fine-tuned',
        'wav2vec2-large': 'Pre-trained',
        'wav2vec2-large-960h': 'Fine-tuned',
        'wav2vec2-xls-r-300m': 'Pre-trained',
        'wav2vec2-large-xlsr-53': 'Pre-trained',
        'wav2vec2-large-xlsr-53-chinese-zh-cn': 'Fine-tuned',
        'hubert-base-ls960': 'Pre-trained',
        'wavlm-base': 'Pre-trained',
        'fbank': 'Baseline',
        'mfcc': 'Baseline',
        'random': 'Baseline'
    }
    
    model_size = {
        'wav2vec2-base': 'Base',
        'mandarin-wav2vec2': 'Base',
        'mandarin-wav2vec2-aishell1': 'Base',
        'wav2vec2-base-100h': 'Base',
        'wav2vec2-large': 'Large',
        'wav2vec2-large-960h': 'Large',
        'wav2vec2-large-xlsr-53': 'Large',
        'wav2vec2-xls-r-300m': 'Large',
        'wav2vec2-large-xlsr-53-chinese-zh-cn': 'Large',
        'hubert-base-ls960': 'Base',
        'wavlm-base': 'Base',
        'fbank': 'Baseline',
        'mfcc': 'Baseline',
        'random': 'Baseline'
    }
    
    model_type = {
        'wav2vec2-base': 'Wav2Vec2.0',
        'mandarin-wav2vec2': 'Wav2Vec2.0',
        'mandarin-wav2vec2-aishell1': 'Wav2Vec2.0',
        'wav2vec2-base-100h':'Wav2Vec2.0',
        'wav2vec2-large': 'Wav2Vec2.0',
        'wav2vec2-large-960h': 'Wav2Vec2.0',
        'wav2vec2-large-xlsr-53':'Wav2Vec2.0',
        'wav2vec2-xls-r-300m': 'Wav2Vec2.0',
        'wav2vec2-large-xlsr-53-chinese-zh-cn': 'Wav2Vec2.0',
        'hubert-base-ls960': 'HuBert',
        'wavlm-base': 'WavLM',
        'fbank': 'Baseline',
        'mfcc': 'Baseline',
        'random': 'Baseline'
    }
    
    task_names_plotting = {
        'syllables_accents': 'English Pitch Accent',
        'tone': 'Mandarin Tone',
        'phones_accents': 'Phone Level Accents',
        'phonwords_accents': 'Word Level Accents',
        'f0': 'Pitch (Hz)',
        'stress': 'Lexical Stress',
        'energy': 'Root Mean Squared Energy',
        'intensity': 'Intensity in dB SPL',
        'intensity_parselmouth': 'Intensity in dB SPL',
        'stress_polysyllabic': 'Lexical Stress Polysyllabic',
        'syllables_accents_polysyllabic': 'Phrasal Accents Polysyllabic',
        'f0_300': 'Pitch (Hz)',
        'f0_diff': 'F0 Syllable Difference',
        'f0_std': 'F0 Standard Deviation',
        'energy_std': 'Energy Standard Deviation',
        'crepe-f0': 'Pitch (Hz)'
    }
    
    result_df['model_names'] = result_df.model.map(lambda x: names_for_plotting[x])
    result_df['language'] = result_df.model.map(lambda x: model_language[x])
    result_df['model_state'] = result_df.model.map(lambda x: model_state[x])
    result_df['model_size'] = result_df.model.map(lambda x: model_size[x])
    result_df['task_name'] = result_df.task.map(lambda x: task_names_plotting[x])
    result_df['model_type'] = result_df.model.map(lambda x: model_type[x])
    result_df.to_csv(save_path)

    

    
    return result_df


def plot_comparison(log_root, model_list, corpus, task, probe, save_path):
    
    
    plot_df = get_model_comparison(log_root, model_list, corpus, task, probe)
    english = plot_df.loc[plot_df.model == 'wav2vec2-base']
    mandarin = plot_df.loc[plot_df.model == 'mandarin-wav2vec2']
    plt.plot(english.layer, english.score, '--', label='wav2vec2-base')
    plt.plot(mandarin.layer, mandarin.score, '.-', label='mandarin')
    plt.legend()
    #f = sns.lineplot(
    #    data=plot_df,
    #    x='layer',
    #    y='score',
    #    style='model',
    ##    hue='model',
    #    hue_order = model_list,
    #    style_order=model_list
    #).get_figure()
    
    plt.savefig(save_path)
    plt.close()
    
def main():
    os.makedirs('plots', exist_ok=True)
    log_folder = 'logs_01-25-2024/logs'
    mand_english_comparison = [('wav2vec2-base', 12), ('mandarin-wav2vec2', 12)]
    plot_list = [
        ([('wav2vec2-base', 12)], 'switchboard', 'syllables_accents', 'linear', 'plots/english_syl_accents.png'),
        (mand_english_comparison, 'mandarin-timit', 'tone', 'linear', 'plots/base_tone_linear.png'),
        ([('mandarin-wav2vec2', 12)], 'mandarin-timit', 'f0', 'linear', 'plots/base_mandarin_f0_linear.png'),
        ([('wav2vec2-base', 12), ('mandarin-wav2vec2', 12)], 'switchboard', 'phones_accents', 'linear', 'plots/base_switchboard_phone_accents_linear.png'),
        ([('wav2vec2-base', 12)], 'switchboard', 'f0', 'linear', 'plots/base_switchboard_f0_linear.png'),
 
    ]
    
    for model_list, corpus, task, probe, save_path in plot_list:
    
        log_root = Path(f'{log_folder}/{corpus}/{task}')
        
        plot_comparison(log_root, model_list, corpus, task, probe, save_path)
        
        
if __name__ == '__main__':
    #main()
    metric_mapping = {
        'f0': 'R2',
        'tone': 'f1_macro',
        'syllables_accents': 'f1_score',
        'phonwords_accents': 'f1_score',
        'phones_accents': 'f1_score',
        'stress': 'f1_score',
        'energy': 'R2'
    }
    metric_list_mapping = {
        'f0': ['R2', 'mean_squared_error', 'pearson', 'normality'],
        'tone': ['f1_macro', 'accuracy', 'recall_macro', 'precision_macro'],
        'syllables_accents': ['f1_score', 'accuracy', 'recall', 'precision'],
        'phonwords_accents': ['f1_score', 'accuracy', 'recall', 'precision'],
        'phones_accents': ['f1_score', 'accuracy', 'recall', 'precision'],
        'stress': ['f1_score', 'accuracy', 'recall', 'precision'],
        'energy': ['R2', 'mean_squared_error', 'pearson', 'normality'],
        'intensity_parselmouth': ['R2', 'mean_squared_error', 'pearson', 'normality'],
        'intensity': ['R2', 'mean_squared_error', 'pearson', 'normality'],
        'stress_polysyllabic':['f1_score', 'accuracy', 'recall', 'precision'], 
        'f0_300': ['R2', 'mean_squared_error', 'pearson', 'normality'], 
        'f0_diff': ['R2', 'mean_squared_error', 'pearson', 'normality'], 
        'f0_std': ['R2', 'mean_squared_error', 'pearson', 'normality'], 
        'energy_diff': ['R2', 'mean_squared_error', 'pearson', 'normality'], 
        'energy_std': ['R2', 'mean_squared_error', 'pearson', 'normality'], 
        'crepe-f0': ['R2', 'mean_squared_error', 'pearson', 'normality'],  
        'syllables_accents_polysyllabic':['f1_score', 'accuracy', 'recall', 'precision'],  
    }
    save_path = 'results/interspeech_final_results.csv'#'results/multi_metric_results.csv'
    get_result_df(Path('results/logs_03-12-2024/logs'), metric_mapping, save_path, ci=False)#Path('results/logs'), metric_list_mapping, save_path)
    
    
    save_alternative = 'results/full_results_alternative_roc_auc.csv'
    alternative_mapping = {
        'f0': 'mean_squared_error',
        'tone': 'log_loss',
        'syllables_accents': 'log_loss',
        'phonwords_accents': 'log_loss',
        'phones_accents': 'log_loss',
        'stress': 'log_loss',
        'energy': 'mean_squared_error'
    }
    alternative_mapping2 = {
        'f0': 'mean_squared_error',
        'tone': 'accuracy',
        'syllables_accents': 'accuracy',
        'phonwords_accents': 'accuracy',
        'phones_accents': 'accuracy',
        'stress': 'accuracy',
        'energy': 'mean_squared_error'
    }
    
    alternative_mapping3 = {
        'f0': 'mean_squared_error',
        'tone': 'roc_auc',
        'syllables_accents': 'roc_auc',
        'phonwords_accents': 'roc_auc',
        'phones_accents': 'roc_auc',
        'stress': 'roc_auc',
        'energy': 'mean_squared_error'
    } 
    
    #get_result_df(Path('results/logs_02-05-2024/logs'), alternative_mapping3, save_alternative)
    
    
    
        