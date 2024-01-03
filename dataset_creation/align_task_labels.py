import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import shutil

SWITCHBOARD_PATH = Path('data/switchboard')

TASK_SET = {
    'switchboard':
    {#'phone_accents': (SWITCHBOARD_PATH / 'phones', SWITCHBOARD_PATH / 'accents'),
     'word_accents': (SWITCHBOARD_PATH / 'phonwords', SWITCHBOARD_PATH / 'accents'),
     'syllable_accents': (SWITCHBOARD_PATH / 'syllables', SWITCHBOARD_PATH / 'accents'),
     'phone_accents': (SWITCHBOARD_PATH / 'phones', SWITCHBOARD_PATH / 'accents'),
     
     'stress': (SWITCHBOARD_PATH / 'syllables', None),
     'phonemes': (SWITCHBOARD_PATH / 'phones', None),
     
     'f0': (SWITCHBOARD_PATH / 'f0', None)
    },
    
    'mandarin-timit':
        {
            'tone': (Path('data/mandarin-timit/tone'), None),
            'f0': (Path('data/mandarin/f0'), None)
        }

}


def ms2idx(time_s, step_s=0.02):
    time_ms = int(time_s * 1000)
    step_ms = int(step_s * 1000)
    return ((time_ms - (time_ms%step_ms))/ step_ms) / 1000


## Get the entire dataset with indices
def get_neural_indices(annotation_dir, save_dir, accent_dir=None, step=0.02):
    
    experiment_df = pd.DataFrame()
    tmp_dir =  save_dir / 'tmp_save'
    os.makedirs(tmp_dir, exist_ok=True)
    for file in tqdm(list(annotation_dir.glob('*.csv'))):
        iter_df = pd.read_csv(file)
        iter_df['file_id'] = file.stem
        # This format can then be exploded after reading
        #try:
        iter_df['start_end_indices'] = iter_df.apply(
            lambda x: [
                ms2idx(x.start, step_s=step),
                ms2idx(x.end, step_s=step)
                ], axis=1
            ) 
        #except ValueError:
        #    print()
        
        if accent_dir:
            accent_path = annotation_dir.parent / 'accent' / file.name
            accent_file = pd.read_csv(accent_path)
            accent_file['start_end_indices'] = accent_file.start.map(
            lambda x:
                ms2idx(x, step_s=step)
            )  
            
            iter_df['label'] = iter_df.apply(
                lambda x:\
                    accent_file.loc[
                        (x.start <= accent_file.start) & (x.end >= accent_file.end)
                        ][['label']] if \
                        len(accent_file.loc[
                        (x.start <= accent_file.start) & (x.end >= accent_file.end)
                        ]) > 0 else 0, axis=1
            )
        #iter_df['start_end_indices'] = iter_df.start_end_indices.map(lambda x: list(range(int(x[0]), int(x[1]))))
        iter_df.to_csv(tmp_dir / file.name, index=False)
    
    for file in tmp_dir.glob('*.csv'):
        iter_df = pd.read_csv(file)
        experiment_df = pd.concat([experiment_df, iter_df])
        
    experiment_df.to_csv(save_dir/f'{annotation_dir.stem}.csv')   
    shutil.rmtree(tmp_dir)
        
def create_task_datasets(task_set, save_dir):
    
    for key, (feat_dir_value, accent_dir_value) in task_set.items():
        
        has_accent = 'with' if accent_dir_value is not None else 'without'
        print(f'Creating task for {feat_dir_value.stem} {has_accent} accent values...')
        get_neural_indices(feat_dir_value, save_dir, accent_dir=accent_dir_value)


if __name__ == '__main__':
    for corpus, task_set in TASK_SET.items():
        save_dir = Path(f'data/{corpus}/aligned_tasks')
        os.makedirs(save_dir, exist_ok=True)
        create_task_datasets(task_set, save_dir)
    #get_neural_indices(Path('data/switchboard/phones'), accent_dir=Path('data/switchbaord/accents'))
## Figure out how to map so that either it works with a dataloader
## Or I can mask out necessary tokens without recomputing every iteration