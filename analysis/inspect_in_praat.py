from pympi import Praat
import pandas as pd
import soundfile as sf
import librosa
import argparse
from pathlib import Path
import os
import shutil
from tqdm import tqdm


def translate2dotname(file_id, feat):
    return f'{file_id[:6]}{file_id[6]}_{file_id.split("_")[-1]}'


def translate2long(file, wav_file, column='target'):
    
    all_data = pd.read_csv(file)
    data = all_data.loc[all_data.file_id == wav_file.stem]
    
    #for row in ['target', 'pred']:
    tier_data = data.groupby((data[column] != data[column].shift()).cumsum()).agg(
        {column: (column, 'first'), 
        'start' : ('timestamp_ms', 'min'), 
        'end': ('timestamp_ms', 'max')
        }
        ).reset_index(drop=True)
    


def create_textgrid_object(file_dur, file_start=0):
    
    return Praat.TextGrid(xmin=file_start, xmax=file_dur)


def create_tier(tg, feat_name, csv_data, tiertype='IntervalTier'):
    
    tier = tg.add_tier(name=feat_name, tier_type=tiertype)
    if tiertype == 'IntervalTier':
        for row in csv_data.itertuples():
            if row.start < row.end:
                try:
                    tier.add_interval(row.start, row.end, row.label,)
                except AttributeError:
                    tier.add_interval(row.start, row.end, row.stress)
    
    elif tiertype == 'TextTier':
        for row in csv_data.itertuples():
            tier.add_point(row.start, row.label)


def generate_tier(data, tg, tier_name):
    
        if (data.start == data.end).all():
            tiertype = 'TextTier'
        else:
            tiertype = 'IntervalTier'
            
        create_tier(tg, tier_name, data, tiertype=tiertype)   
                 

def generate_textgrid(wav_file, annotation_list, save_dir, interactive=False):
    
    dur = librosa.get_duration(path=wav_file)
    tg = create_textgrid_object(dur)
    for annotation in annotation_list:
        
        feat = annotation.stem
        
        if '_results_' in annotation.stem:
            for column in ['target', 'pred']:
                data = translate2long(annotation, wav_file, column=column)
                generate_tier(data, tg, column)
        else:   
            try:
                data = pd.read_csv(Path(annotation, wav_file.stem).with_suffix('.csv'))
            except FileNotFoundError:
                data = pd.read_csv(Path(annotation, translate2dotname(wav_file.stem, feat)).with_suffix('.csv')) 
            generate_tier(data, tg, feat)
            

    save_file = Path(save_dir, f'{wav_file.stem}.TextGrid')
    tg.to_file(save_file)

    
    
def main():
    
    result_layer = '15'
    result_experiment = 'accent_spreading'
    result_feature = 'phones'
    result_model = 'wav2vec2-large-960h'
    result_probe = 'linear'
    log_date = '25-09-2023'
    result_logs = Path('~/Projects/interpretability/experiments/{result_experiment}/{log_date}_logs/{result_feature}')
    result_filename = f'{result_feature}_results_{result_model}_layer{result_layer}_{result_probe}.csv'
    
    
    corpus = 'switchboard'
    
    annotation_list = [
        Path(f'data/{corpus}/phones'),
        Path(f'data/{corpus}/phonwords'),
        #Path(f'~/Projects/interpretability/data/{corpus}/f0'),
        result_logs / result_filename
    ]
    save_dir = f'analysis/inspection/{corpus}/{result_feature}_{result_layer}_{result_probe}/'
    os.makedirs(save_dir, exist_ok=True)
    ls = [file.stem for file in list(Path(f'data/{corpus}/accent').glob('*.csv'))]
    for file in tqdm(list(Path(f'data/{corpus}/wav').glob('*.wav'))):
        if file.stem in ls:
            generate_textgrid(file, annotation_list, save_dir)
            shutil.copy(file, save_dir / file.name)
            
        
if __name__ == '__main__':
    main()