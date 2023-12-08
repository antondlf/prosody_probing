import librosa
import os
from pathlib import Path
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm


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
            chunk['label'] = [1]* len(chunk)
    
        
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


def chunk_switchboard_files(dir, feat_mapping=feat_labels):

    for file in tqdm(list(dir.glob('*.wav'))):
        audio_id = file.name.split('.')[0]
        split_wav(audio_id, file, Path('data/switchboard'), feat_mapping=feat_mapping)


if __name__ == '__main__':

    chunk_switchboard_files(Path('corpora/nxt_switchboard_ann/wav'))