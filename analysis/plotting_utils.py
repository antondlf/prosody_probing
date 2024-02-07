import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, log_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import math
import os


def get_metric(name):
    
    mapping = {
        'f1_score': f1_score,
        'f1_micro': f1_score,
        'mean_squared_error': mean_squared_error,
        'roc_auc': roc_auc_score,
        'log_loss': log_loss,
        'pearson': pearsonr,
        'spearman': spearmanr
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


def calculate_metric(df, metric_name):
    
    metric = get_metric(metric_name)
    if 'y_proba' in df.columns:
        
        if metric_name in ['log_loss', 'roc_auc']:
            probabilities = df.y_proba.map(lambda x: x[1]).to_numpy()
            return metric(df.y_true, probabilities)
        
    else:
        if metric_name == 'f1_micro':
            return metric(df.y_true, df.y_pred, average='micro')
        elif metric_name == 'pearson':
            return metric(df.y_true, df.y_pred)[0]
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


def get_log_loss(result_file, result_df):
    
    binarizer = LabelBinarizer()
    layer_num = result_file.stem.split('_')[1]
    
    dist_path = result_file.parent / f'layer-{layer_num}_distribution.txt'
    distribution_df = pd.read_csv(dist_path, sep='\t')
    # There must be a tab before \n in files because an extra column is being read
    distributions = distribution_df.to_numpy()[:, :-1]
    y_true = binarizer.fit_transform(result_df.y_true)
    
    return log_loss(y_true, distributions)#, labels=distribution_df.columns[:-1])


def get_result_df(log_root, metric_mapping, save_path):
    """Generates a dataframe of all dfs and saves to
    csv.
    –––––––––––––––––––––––––––––––––––––––––––––––––
    log_root: path to top log directory.
    metrics: dictionary mapping task to metric function.
    """
    
    row_list = list()
    
    for corpus in log_root.glob('*'):
        
        corp = corpus.stem
        
        for task in corpus.glob('*'):
            tsk = task.stem
            metric = get_metric(metric_mapping[tsk])
            for model in task.glob('*'):
                if model.name not in ['linear', 'mlp']:
                    mod = model.stem
                    for probe in model.glob('*'):
                        
                        prob = probe.stem
                        print(f'Calculating {metric_mapping[tsk]} for {tsk} on {mod} and {corp} with {prob} probe...')                    
                        for layer in tqdm(list(probe.glob('layer_*.csv'))):
                            iter_df = pd.read_csv(layer)
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
                                result = get_log_loss(layer, iter_df)
                                #confidence_interval = (np.nan, np.nan)#get_bootstrap_ci(result, iter_df, metric) 
                            else: 
                                result = calculate_metric(iter_df, metric_mapping[tsk])
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
                            row_list.append({
                                'corpus': corp,
                                'task': tsk,
                                'model': mod,
                                'probe': prob,
                                'layer': int(layer.stem.split('_')[1]),
                                'score': result,
                                #'ci': (None, None)
                            })
                                
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
        'fbank': 'Mel-Filterbank',
        'mfcc': 'MFCC'
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
        'fbank': 'Baseline',
        'mfcc': 'Baseline'
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
        'fbank': 'Baseline',
        'mfcc': 'MFCC'
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
        'fbank': 'Baseline',
        'mfcc': 'Baseline'
    }
    
    task_names_plotting = {
        'syllables_accents': 'English Pitch Accent',
        'tone': 'Mandarin Tone',
        'phones_accents': 'Phone Level Accents',
        'phonwords_accents': 'Word Level Accents',
        'f0': 'Pitch (Hz)',
        'stress': 'Lexical Stress'
    }
    
    result_df['model_names'] = result_df.model.map(lambda x: names_for_plotting[x])
    result_df['language'] = result_df.model.map(lambda x: model_language[x])
    result_df['model_state'] = result_df.model.map(lambda x: model_state[x])
    result_df['model_size'] = result_df.model.map(lambda x: model_size[x])
    result_df['task_name'] = result_df.task.map(lambda x: task_names_plotting[x])
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
        'f0': 'pearson',
        'tone': 'f1_micro',
        'syllables_accents': 'f1_score',
        'phonwords_accents': 'f1_score',
        'phones_accents': 'f1_score',
        'stress': 'f1_score'
    }
    save_path = 'results/full_results.csv'
    get_result_df(Path('results/logs_02-05-2024/logs'), metric_mapping, save_path)
    
    
    save_alternative = 'results/full_results_alternative.csv'
    alternative_mapping = {
        'f0': 'mean_squared_error',
        'tone': 'log_loss',
        'syllables_accents': 'log_loss',
        'phonwords_accents': 'log_loss',
        'phones_accents': 'log_loss',
        'stress': 'log_loss'
    }
    
    get_result_df(Path('results/logs_02-05-2024/logs'), alternative_mapping, save_alternative)
    
        