import librosa
import os
import shutil
from pathlib import Path
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
pd.options.mode.chained_assignment = None


feat_labels = {
    'syllables': 'stress',
    'phonwords': 'orth',
    'phrase': 'type',
    'breaks': 'index',
    'accent': None,
    'phones': 'label',
    'turns': 'id'
 }

def load_wav(path):
    return sf.read(path) # upsampling from 8khz


def write_wav(path, audio, sampling_rate):

    print(sampling_rate)
    assert sampling_rate == 8000

    upsampled = librosa.resample(audio, orig_sr=8000, target_sr=16000)
    # check resampling
    print('target sr:', upsampled.shape, 'original_sr:',audio.shape)

    if upsampled.shape[0] > 0:
        assert upsampled.shape != audio.shape
        assert audio.shape[0]*(16000/8000) == upsampled.shape[0]

    sf.write(path, upsampled, 16000)


def extract_chunk(audio_data, sampling_rate, start_time, end_time):
    """
    Takes in an audio file and returns a chunk bounded by start and
    end time.
    -----------------------
    audio_data: array of samples.
    sampling_rate: sampling rate in hz.
    start_time: beginning timestamp in miliseconds.
    end_time: ending timestamp in miliseconds.
    -----------------------
    :returns audio samples for the chunk.
    """

    start_seconds = float(start_time)
    end_seconds = float(end_time)

    start_idx, end_idx = librosa.time_to_samples(
        [start_seconds, end_seconds],
        sr=sampling_rate
    )

    return audio_data[start_idx:end_idx]


def get_turns(audio_id, turn_dir=Path('corpora/nxt_switchboard_ann/csv/turns')):
    
    return pd.read_csv(Path(turn_dir,audio_id).with_suffix('.csv'))


def all_backchannel(root_directory, file_path):
    """Removes backchannel turns and audios shorter than
    40 miliseconds from the data.""" 
    audio_duration = librosa.get_duration(path=file_path)
    phrase_dir = root_directory / 'phrase' / file_path.with_suffix('.csv').name
    try:
        phrase_check = pd.read_csv(phrase_dir)
    except FileNotFoundError:
        if audio_duration >= 0.04:
            return False
        else:
            return True
    
    if len(phrase_check) > 0:
        try:
            return (phrase_check.label == 'backchannel').all(axis=0)

        except AttributeError:
            print('wrong attribute')
            return (phrase_check.type == 'backchannel').all(axis=0)
        
    elif audio_duration >= 0.04:
        return False
    
    else:
        return True


def adjust_annotations(start, end, audio_id, turn_id, annotation_dir, save_dir,feat_mapping=feat_labels):
    
    
    for feat in os.listdir(annotation_dir):
        os.makedirs(Path(save_dir, feat), exist_ok=True)
        annotation_file = pd.read_csv(Path(annotation_dir, feat, audio_id).with_suffix('.csv'))
        
        chunk = annotation_file.loc[(annotation_file['start'] >= start) & (annotation_file['end'] <= end)]

        chunk['start'] = chunk.start - start
        chunk['end'] = chunk.end - start
        if feat_mapping[feat] is not None:
            chunk = chunk[['start', 'end', feat_mapping[feat]]]
            chunk.columns = ['start', 'end', 'label']
        
        else:
            # the case of phones
            chunk['label'] = [1] * len(chunk)
    
        
        chunk.to_csv(Path(save_dir, feat, turn_id).with_suffix('.csv'))


def split_wav(audio_id, audio_dir, save_dir, feat_mapping=feat_labels):

    turns = get_turns(audio_id)
    wav_data, sampling_rate = sf.read(audio_dir)
    os.makedirs(save_dir / 'wav', exist_ok=True)

    for turn in turns.itertuples():
        try:
            start_offset = turn.start
            end_offset = turn.end
        except KeyError:
            continue

        chunk = extract_chunk(wav_data, sampling_rate, start_offset, end_offset)

        save_path = os.path.join(save_dir, 'wav', audio_id + '_' + turn.id + '.wav')

        if sampling_rate != 16000:
            write_wav(save_path, chunk, sampling_rate)
        else:
            sf.write(save_path, chunk, sampling_rate)
        turn_id = audio_id + '_' + turn.id 
        adjust_annotations(start_offset, end_offset, audio_id, turn_id, Path('corpora/nxt_switchboard_ann/csv/'), save_dir, feat_mapping=feat_mapping)


def chunk_switchboard_files(root_directory, save_directory, feat_mapping=feat_labels):

    split_save = Path('corpora/nxt_switchboard_ann/split_data')
    for file in tqdm(list(root_directory.glob('*.wav'))):
        audio_id = file.name.split('.')[0]
        split_wav(audio_id, file, split_save, feat_mapping=feat_mapping) 
        
    print('Removing backchannel phrases')
    split_wav_files = split_save/'wav'
    for file in tqdm(list(split_wav_files.glob('*.wav'))):

        if not all_backchannel(save_directory, file):
            
            shutil.copy(file, save_directory/ 'wav' / file.name)


def cleanup_extra_files(root_directory, wav_directory, new_root):
    master_list = [file.stem for file in wav_directory.glob('*.wav')]
    print('Moving CSVs in Master list to data folder')
    for folder in tqdm(os.listdir(root_directory)):
        os.makedirs(new_root / folder, exist_ok=True)
        feat_folder = root_directory / folder
        for file in feat_folder.glob('*.csv'):
            
            if file.stem not in master_list:
                os.remove(file)
            else:
                print(file, new_root / file.name)
                file.rename(new_root / folder / file.name)


def remove_incomplete_annotations(new_root):
    
    # TODO: There are smarter and faster ways to do this
    # remove files where last turn start times are greater than the
    # last start of any annotation file
    # Do this before transfering files to data dir.
    file_list = os.listdir(new_root)
    empty_list = list()
    for feat in file_list:
        feat_files = new_root / feat
        for file in feat_files.glob('*.csv'):
            
            if len(pd.read_csv(file)) == 0:
                empty_list.append(file.stem)
    
    incomplete_files = set(empty_list)
    
    for file_id in incomplete_files:
        
        for feat in file_list:
            
            rm_file = new_root / feat / file_id
            endpoint = '.wav' if feat == 'wav' else '.csv'
            os.remove(rm_file.with_suffix(endpoint))


def split_move_switchboard():
    
    old_root = Path('corpora/nxt_switchboard_ann')
    new_root = Path('data/switchboard')
    wav_dir = new_root / 'wav'
    os.makedirs(wav_dir, exist_ok=True) 
    chunk_switchboard_files(old_root / 'wav', new_root)
    split_root = old_root / 'split_data'
    cleanup_extra_files(split_root, wav_dir, new_root)
    
    remove_incomplete_annotations(new_root)
    
    
if __name__ == '__main__':
    split_move_switchboard()