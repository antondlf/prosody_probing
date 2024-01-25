import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, log_loss


def get_metric(name):
    
    mapping = {
        'f1_score': f1_score,
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
        metric(df.y_true, df.y_pred)
        
        
def loop_model_result_dir(string_format_dir, layer_number, metric_name):
    
    results = list()
    
    for layer in range(0, layer_number+1):
        results.append(calculate_metric, pd.read_csv(string_format_dir.format(layer), metric_name))
        
    return list(range(0, layer_number+1)), results
