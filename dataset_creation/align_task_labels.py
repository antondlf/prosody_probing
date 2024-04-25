from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import os
import shutil
import math
import librosa

SWITCHBOARD_PATH = Path('data/switchboard')

TASK_SET = {
    'switchboard':
    {#'phone_accents': (SWITCHBOARD_PATH / 'phones', SWITCHBOARD_PATH / 'accents', SWITCHBOARD_PATH / 'wav'),
     'word_accents': (SWITCHBOARD_PATH / 'phonwords', SWITCHBOARD_PATH / 'accents', SWITCHBOARD_PATH / 'wav'),
     'syllable_accents': (SWITCHBOARD_PATH / 'syllables', SWITCHBOARD_PATH / 'accents', SWITCHBOARD_PATH / 'wav'),
     #'phone_accents': (SWITCHBOARD_PATH / 'phones', SWITCHBOARD_PATH / 'accents', SWITCHBOARD_PATH / 'wav'),
     
     'stress': (SWITCHBOARD_PATH / 'syllables', None, SWITCHBOARD_PATH / 'wav'),
     'stress_polysyllabic': (SWITCHBOARD_PATH / 'syllables', None, SWITCHBOARD_PATH / 'wav'),
     'syllable_accents_polysyllabic': (SWITCHBOARD_PATH / 'syllables', SWITCHBOARD_PATH / 'accents', SWITCHBOARD_PATH / 'wav'),
     
     'phonemes': (SWITCHBOARD_PATH / 'phones', None, SWITCHBOARD_PATH / 'wav'),
     'f0': (SWITCHBOARD_PATH / 'f0', None, SWITCHBOARD_PATH / 'wav'),
     'f0_std':(SWITCHBOARD_PATH / 'f0_std', None, SWITCHBOARD_PATH / 'wav'),
     'f0_diff':(SWITCHBOARD_PATH / 'f0_diff', None, SWITCHBOARD_PATH / 'wav'),  
     'energy': (SWITCHBOARD_PATH / 'energy', None, SWITCHBOARD_PATH / 'wav'),
     'crepe-f0': (SWITCHBOARD_PATH / 'crepe-f0', None, SWITCHBOARD_PATH / 'wav'),
     'energy_std': (SWITCHBOARD_PATH / 'energy_std', None, SWITCHBOARD_PATH / 'wav'),
    'energy_diff': (SWITCHBOARD_PATH / 'energy_diff', None, SWITCHBOARD_PATH / 'wav'),
    'intensity': (SWITCHBOARD_PATH / 'intensity', None, SWITCHBOARD_PATH / 'wav'),
    'intensity_parselmouth': (SWITCHBOARD_PATH / 'intensity_parselmouth', None, SWITCHBOARD_PATH / 'wav'),
    'f0_300': (SWITCHBOARD_PATH / 'f0_300', None, SWITCHBOARD_PATH / 'wav'),

    },
    
    'mandarin-timit':
        {
            'tone': (Path('data/mandarin-timit/tone'), None, Path('data/mandarin-timit/wav')),
            'tone_rhymes': (Path('data/mandarin-timit/tone_rhymes'), None, Path('data/mandarin-timit/wav')),
            'f0': (Path('data/mandarin-timit/f0'), None, Path('data/mandarin-timit/wav')),
            'f0_std': (Path('data/mandarin-timit/f0_std'), None, Path('data/mandarin-timit/wav')),
            'f0_diff': (Path('data/mandarin-timit/f0_diff'), None, Path('data/mandarin-timit/wav')),
            'energy': (Path('data/mandarin-timit/energy'), None, Path('data/mandarin-timit/wav')),
            'crepe-f0': (Path('data/mandarin-timit/crepe-f0'), None, Path('data/mandarin-timit/wav')), 
            'energy_std': (Path('data/mandarin-timit/energy_std'), None, Path('data/mandarin-timit/wav')),
            'energy_diff': (Path('data/mandarin-timit/energy_diff'), None, Path('data/mandarin-timit/wav')),
            'intensity': (Path('data/mandarin-timit/intensity'), None, Path('data/mandarin-timit/wav')),
            'intensity_parselmouth': (Path('data/mandarin-timit/intensity_parselmouth'), None, Path('data/mandarin-timit/wav')),
            'f0_300': (Path('data/mandarin-timit/f0_300'), None, Path('data/mandarin-timit/wav')),
        },
    'bu_radio':
        {
            'phone_accents': (Path('data/bu_radio/phones_accents'), None, Path('data/bu_radio/wav')),
            'word_accents': (Path('data/bu_radio/word_accents'), None, Path('data/bu_radio/wav')),
            'f0': (Path('data/bu_radio/f0'), None, Path('data/bu_radio/wav')),
            'crepe-f0': (Path('data/bu_radio/crepe-f0'), None, Path('data/bu_radio/wav')),
            'energy': (Path('data/bu_radio/energy'), None, Path('data/bu_radio/wav')),
            'intensity': (Path('data/bu_radio/intensity'), None, Path('data/bu_radio/wav')),
            'intensity_parselmouth': (Path('data/bu_radio/intensity_parselmouth'), None, Path('data/bu_radio/wav')),
            'f0_300': (Path('data/bu_radio/f0_300'), None, Path('data/bu_radio/wav')),
        
        }

}


def calculate_cnn_output_length(input_length, k, s, p=0):
  # Note we set p=0 since w2v2 uses the default padding from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
  # See https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L69C9-L69C9
  return math.floor((input_length + 2*p - k)/s) + 1


def get_frames_from_samples(raw_input_length):
    after_cnn1 = calculate_cnn_output_length(raw_input_length, k=10, s=5)
    after_cnn2 = calculate_cnn_output_length(after_cnn1, k=3, s=2)
    after_cnn3 = calculate_cnn_output_length(after_cnn2, k=3, s=2)
    after_cnn4 = calculate_cnn_output_length(after_cnn3, k=3, s=2)
    after_cnn5 = calculate_cnn_output_length(after_cnn4, k=3, s=2)
    after_cnn6 = calculate_cnn_output_length(after_cnn5, k=2, s=2)
    after_cnn7 = calculate_cnn_output_length(after_cnn6, k=2, s=2)
    
    return after_cnn7


def derive_silences(df):
    offset = 0
    for idx in range(1, len(df)-1):
        i = idx+offset
        if df.iloc[i, 1] != df.iloc[i+1, 0]:
            
            df1 = df.iloc[:i+1]
            df2 = df.iloc[i+1:]
            insert_row = pd.DataFrame({
                        'start': df1.iloc[-1, 1],
                        'end': df2.iloc[0, 0],
                        'label': 'sil',
                        'file_id': df1.iloc[0, -1]
                        }, index=[i])
            df = pd.concat([df1, insert_row, df2])
            offset += 1
    return df.reset_index(drop=True)


def ms2idx(time_s, step_s=0.02):
    time_ms = int(time_s * 1000)
    step_ms = int(step_s * 1000)
    return ((time_ms - (time_ms%step_ms))/ step_ms)


def map_word_label(row, word_df):
    
   try:
       return word_df.loc[(word_df.start <= row.start)&(word_df.end >=row.end)].label.item()
   except ValueError:
       return np.nan


def remove_monosyllables(df, root_dir, filename, feat_name='phonwords'):
    
    word_df = pd.read_csv(root_dir.parent / feat_name / filename)
    
    df['word'] = df.apply(lambda x: map_word_label(x, word_df), axis=1)
    
    
    df['syllable_length'] = df.groupby(df['word'].ne(df['word'].shift()).cumsum()).word.transform(len)
    #df.dropna(inplace=True)
    df = df.loc[(df.label != 'sil')&(df.label != 'SIL')]
    
    return df.loc[df.syllable_length > 1][['file_id', 'start', 'end', 'label']]
    

def find_accent_label(x, accent_file, binary_accent=True):
    # One word "sw4019.B.phonwords.xml#id(ms33B_pw98)" in turn 38 of this conversation
    # "boy" is labeled with two accents. There is no reason (at least according to this author)
    # why this should be the case, so it is being treated as a transcription error.
    if len(accent_file.loc[(x.start <= accent_file.start) & (x.end >= accent_file.end)]) > 0:
        try:
            return accent_file.loc[(x.start <= accent_file.start) & (x.end >= accent_file.end)]['label'].item()
        except ValueError:
            if len(accent_file.loc[(x.start <= accent_file.start) & (x.end >= accent_file.end)]) > 1:
                print(f"{x.file_id} has more than one accent for {x.label}")
                if binary_accent:
                    return accent_file.loc[(x.start <= accent_file.start) & (x.end >= accent_file.end)]['label'].iloc[0]
                else:
                    return accent_file.loc[(x.start <= accent_file.start) & (x.end >= accent_file.end)]['label'].sum()
            else:
                raise ValueError(f"{x.file_id} has raised an error.")
    elif x.label == 'SIL':
        return 'SIL'
    else:
        return 0


## Get the entire dataset with indices
def get_neural_indices(annotation_dir, save_dir, wav_dir, accent_dir=None, remove_monosyllabic=False, binary_accent=True, step=0.02):
    
    experiment_df = pd.DataFrame()
    tmp_dir =  save_dir / 'tmp_save'
    os.makedirs(tmp_dir, exist_ok=True)
    for file in tqdm(list(annotation_dir.glob('*.csv'))):
        iter_df = pd.read_csv(file, index_col=0)
        iter_df['file_id'] = file.name.replace('.csv', '')
        if (annotation_dir.name != 'phones') and (annotation_dir.name not in ['f0', 'f0_300', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']):
            # This is super hacky, better to have consistency over 
            # different tasks but for now it will do
            iter_df = derive_silences(iter_df)
        
        if remove_monosyllabic:
            iter_df = remove_monosyllables(iter_df, annotation_dir, file.name, feat_name='phonwords')
            if len(iter_df) == 0:
                continue
        # This format can then be exploded after reading
        #try:
        if annotation_dir.name in ['f0', 'f0_300', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']:
            iter_df['start_end_indices'] = iter_df.start.map(
                lambda x: \
                    ms2idx(x, step_s=step)
                    
            )
        else:
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
                    find_accent_label(x, accent_file, binary_accent=binary_accent), axis=1
            )
        # Refactor all of these checks into two separate functions.
        if annotation_dir.name not in ['f0', 'f0_300', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']:
            iter_df['start_end_indices'] = iter_df.start_end_indices.map(lambda x: list(range(int(x[0]), int(x[1]))))
        
        # Avoid off by one error by calculating the amount of sample outputs
        # And removing one sample if necessary
        file_sample_length = sf.read(wav_dir / f"{file.stem}.wav")[0].shape[0]
        if annotation_dir.name in ['f0', 'f0_300','crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']:
            last_index = iter_df.iloc[-1, -1]
        elif annotation_dir.name == 'phones_accents':
            iter_df = iter_df.loc[iter_df.start_end_indices.map(lambda x: len(x)>0)]
            last_index = iter_df.iloc[-1, -1][-1]
        else: 
            try:
                last_index = iter_df.iloc[-1, -1][-1]#start_end_indices[-1].item()[-1]
            except IndexError:
                iter_df = iter_df[:-1]
                last_index = iter_df.iloc[-1, -1][-1]
            
        frame_length = get_frames_from_samples(file_sample_length)
        if file.stem == 'SP01_112':
            print()
        
        # Make sure all indices are accounted for at the end
        if last_index != frame_length - 1:
            if last_index > frame_length - 1:
                if last_index == frame_length: 
                    if annotation_dir.name in ['f0', 'f0_300', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']:
                        while frame_length-1 < iter_df.iloc[-1, -1]:
                            iter_df = iter_df.iloc[:-1]
                    else:
                        iter_df.at[iter_df.index[-1], 'start_end_indices'] = iter_df.iloc[-1, -1][:-1]
                elif type(iter_df.iloc[-1, -1]) != list:
                    while frame_length-1 < iter_df.iloc[-1, -1]:
                        iter_df = iter_df.iloc[:-1]
                    
                elif frame_length-1 in iter_df.iloc[-1, -1]:
                    index_list = iter_df.iloc[-1, -1]
                    iter_df.at[iter_df.index[-1], 'start_end_indices'] = index_list[:index_list.index(frame_length-1)+1]
                else:
                    while frame_length-1 not in iter_df.iloc[-1, -1]:
                        iter_df = iter_df.iloc[:-1]
                    index_list = iter_df.iloc[-1, -1]
                    iter_df.at[iter_df.index[-1], 'start_end_indices'] = index_list[:index_list.index(frame_length-1)+1]
                
            elif last_index < frame_length-1:
                if last_index == frame_length:
                    iter_df.iloc[-1, -1].append(iter_df.iloc[-1, -1][-1] + 1)       
                else:
                    if annotation_dir.name not in ['f0', 'f0_300', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']:
                        iter_df.loc[iter_df.index[-1]+1] = {
                            'start': iter_df.iloc[-1, 1],
                            'end': librosa.get_duration(path=wav_dir / f"{file.stem}.wav"),
                            'label': 'sil',
                            'file_id': file.stem,
                            'start_end_indices': list(range(iter_df.iloc[-1, -1][-1]+1, frame_length))
                            }
                    #else:
                    #    iter_df.loc[iter_df.index[-1]+1] = {
                    #        'start': iter_df.iloc[-1, 1],
                    #        #'end': librosa.get_duration(path=wav_dir / f"{file.stem}.wav"),
                    #        'label': 'sil',
                    #        'file_id': file.stem,
                    #        'start_end_indices': iter_df.iloc[-1, -1][-1]+1
                    #        } 
                    
        # and at the beginning
        # We don't care if the regression tasks start at 1
        # or at 0 index, it's okay to throw one away.
        if annotation_dir.name in ['f0', 'f0_300', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']:
            first_index = iter_df.iloc[0, -1]
        else:
            try:
                first_index = iter_df.iloc[0, -1][0]
            except IndexError:
                first_index = iter_df.iloc[1, -1][0]
            if first_index > 0:
                
                iter_df = pd.concat([
                    pd.DataFrame({
                    'start': 0, 'end': iter_df.iloc[0, 0], 'label': 'sil', 
                    'file_id': file.stem, 'start_end_indices': list(range(0, iter_df.iloc[0, -1][0]))
                    }), iter_df])
            
        if annotation_dir.name not in ['f0', 'f0_300', 'crepe-f0', 'energy', 'intensity', 'intensity_parselmouth']:
                
            try:
                if iter_df.iloc[-1, -1][-1] != frame_length -1:
                    raise AssertionError("Somethings up! Neural frame length does not align with last index")  
            except IndexError:
                iter_df = iter_df[:-1]
                if iter_df.iloc[-1, -1][-1] != frame_length -1:
                    raise AssertionError("Somethings up! Neural frame length does not align with last index")
        
        #for i, row in iter_df.iterrows():
            
              
        
        iter_df.to_csv(tmp_dir / file.name, index=False, float_format='%.4f')
    
    for file in tmp_dir.glob('*.csv'):
        iter_df = pd.read_csv(file)
        experiment_df = pd.concat([experiment_df, iter_df])
    accent_labels = '_accents' if accent_dir else ''
    binary_accent = '' if binary_accent else '_multiaccent'
    monosyllables = '' if not remove_monosyllabic else '_polysyllabic'
    annotation_domain = 'stress' if (annotation_dir.stem == 'syllables') and (accent_dir == None) else annotation_dir.stem 
    experiment_df.to_csv(save_dir/f'{annotation_domain}{accent_labels}{binary_accent}{monosyllables}.csv', index=False,  float_format='%.3f')   
    shutil.rmtree(tmp_dir)
        
def create_task_datasets(task_set, save_dir, include=None):
    
    for key, (feat_dir_value, accent_dir_value, wav_dir_value) in task_set.items():
        
        if key.split('_')[-1] == 'polysyllabic':
            remove_monosyllabic = True
        else:
            remove_monosyllabic = False
            
        if include is None:      
            has_accent = 'with' if accent_dir_value is not None else 'without'
            print(f'Creating task for {feat_dir_value.stem} {has_accent} accent values...')
            if accent_dir_value is not None:
                for binary_accent in [True, False]:
                    get_neural_indices(feat_dir_value, save_dir, wav_dir_value, remove_monosyllabic=remove_monosyllabic, accent_dir=accent_dir_value, binary_accent=binary_accent)
            else:
                get_neural_indices(feat_dir_value, save_dir, wav_dir_value, accent_dir=accent_dir_value, remove_monosyllabic=remove_monosyllabic)
        else:
            if key in include:
                has_accent = 'with' if accent_dir_value is not None else 'without'
                print(f'Creating task for {feat_dir_value.stem} {has_accent} accent values...')
                if accent_dir_value is not None:
                    for binary_accent in [True, False]:
                        get_neural_indices(feat_dir_value, save_dir, wav_dir_value, remove_monosyllabic=remove_monosyllabic, accent_dir=accent_dir_value, binary_accent=binary_accent)
                else:
                    get_neural_indices(feat_dir_value, save_dir, wav_dir_value, remove_monosyllabic=remove_monosyllabic, accent_dir=accent_dir_value) 


if __name__ == '__main__':
    include = ['tone_rhymes']
    for corpus, task_set in TASK_SET.items():
        save_dir = Path(f'data/{corpus}/aligned_tasks')
        os.makedirs(save_dir, exist_ok=True)
        create_task_datasets(task_set, save_dir, include=include)
    #get_neural_indices(Path('data/switchboard/phones'), accent_dir=Path('data/switchbaord/accents'))
## Figure out how to map so that either it works with a dataloader
## Or I can mask out necessary tokens without recomputing every iteration