import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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


def calculate_metric(df, metric_name):
    
    metric = get_metric(metric_name)
    if 'y_proba' in df.columns:
        
        if metric_name in ['log_loss', 'roc_auc']:
            probabilities = df.y_proba.map(lambda x: x[1]).to_numpy()
            return metric(df.y_true, probabilities)
        
    else:
        if metric_name == 'f1_micro':
            return metric(df.y_true, df.y_pred, average='micro')
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
        
        (mand_english_comparison, 'mandarin-timit', 'tone', 'linear', 'plots/base_tone_linear.png'),
        #(mand_english_comparison, 'mandarin-timit', 'f0', 'linear', 'plots/base_mandarin_f0_linear.png'),
        ([('wav2vec2-base', 12), ('mandarin-wav2vec2', 12)], 'switchboard', 'phones_accents', 'linear', 'plots/base_switchboard_phone_accents_linear.png'),
        #(mand_english_comparison, 'switchboard', 'f0', 'linear', 'plots/base_switchboard_f0_linear.png'),
 
    ]
    
    for model_list, corpus, task, probe, save_path in plot_list:
    
        log_root = Path(f'{log_folder}/{corpus}/{task}')
        
        plot_comparison(log_root, model_list, corpus, task, probe, save_path)
        
        
if __name__ == '__main__':
    main()
    
        