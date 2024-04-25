import crepe
import numpy as np
from scipy.io import wavfile
from librosa.feature import rms
from librosa import times_like
import librosa
from lhotse import Fbank, FbankConfig, Mfcc, MfccConfig
import parselmouth as prsl
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
import math


def get_f0(path):
    
    sr, audio = wavfile.read(path)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    
    # Average every two samples to downsample from 0.01 to 0.02
    # If num samples is odd average all but the last sample and then
    # append last sample.
    if frequency.shape[0] % 2 == 0:
        mean_frequency = np.mean(time.reshape(-1, 2), axis=1)
    
    else:
        mean_frequency = np.append(np.mean(frequency[:-1].reshape(-1, 2), axis=1), frequency[-1])
    
    downsample_time = time[0::2]
    
    return downsample_time, mean_frequency


def get_energy(path, match_w2v2=True):

    sr, audio = wavfile.read(path)
    if match_w2v2:
        rms_values = rms(y=audio, frame_length=400, hop_length=320)
        times = times_like(rms_values, sr=sr, hop_length=320)
        return times, rms_values[0]
    else:
        rms_values = rms(y=audio)
        return times_like(rms_values, sr=sr), rms_values[0]
        

def get_fbank(path, nbins=80):

    sr, audio = wavfile.read(path)
    extractor = Fbank(FbankConfig(num_mel_bins=nbins, frame_shift=0.02)) 
    return extractor.extract(audio.astype(np.float32), sr)
    #return np.mean(fbank.reshape(-1, 10), axis=1) 


def get_mfcc(path, nbins=80):
    sr, audio = wavfile.read(path)
    extractor = Mfcc(MfccConfig(num_mel_bins=nbins, frame_shift=0.02)) 
    return extractor.extract(audio.astype(np.float32), sr)   


def get_pitch_parselmouth(path):
    
    snd = prsl.Sound(str(path))
    pitch = snd.to_pitch(pitch_ceiling=300)
    pitch_values = pitch.selected_array['frequency']
    return pitch.xs(), pitch_values


def get_intensity_parselmouth(path):
    
    snd = prsl.Sound(str(path))
    intensity = snd.to_intensity(time_step=0.02)
    return intensity.xs(), intensity.values.flatten()


def get_intensity(path, match_w2v2=True):
    
    sr, audio= wavfile.read(path)
    if match_w2v2:
        x = librosa.util.frame(audio, frame_length=400, hop_length=320)
        time = times_like(x, sr=sr, hop_length=320)
    else:
        x = librosa.util.frame(audio, frame_length=2048, hop_length=512)
        time = times_like(x, sr=sr, hop_length=512)
        
    se = np.mean(librosa.util.abs2(x, dtype=np.float32), axis=-2, keepdims=True)
    #time, energy = get_energy(path)
    return time, 10*np.log10(se/(2e-5**2)).flatten()


def get_pitch_energy(path):
    
    pitch_time, pitch = get_f0(path)
    rms_time, rms = get_energy(path, match_w2v2=True)
    
    return pitch_time, np.vstack((pitch, rms))


def get_energy_std(path, domain_path=Path('data/switchboard/phonwords'), match_w2v2=True):
    
    sr, audio = wavfile.read(path)
    if match_w2v2:
        rms_values = rms(y=audio, frame_length=400, hop_length=320)[0]
        times = times_like(rms_values, sr=sr, hop_length=320)
    else:
        rms_values = rms(y=audio)[0]
        times = times_like(rms_values, sr=sr)
        
    domain = pd.read_csv(domain_path / f'{path.stem}.csv')
    
    energy_std_rows = list()
    for row in domain.itertuples():
        
        rms_for_std = rms_values[np.where(np.logical_and(times>=row.start, times<=row.end))]
        
        deviation = np.std(rms_for_std)
        energy_std_rows.append({
            'start': row.start,
            'end': row.end,
            'file_id': path.stem,
            'label': deviation
        })
        
    return pd.DataFrame(energy_std_rows)


def get_f0_std(path, domain_path=Path('data/switchboard/phonwords'), match_w2v2=True):
    
    time, pitch =get_pitch_parselmouth(str(path))
    
    domain = pd.read_csv(domain_path / f'{path.stem}.csv')
    f0_std_rows = list()
    for row in domain.itertuples():
        
        f0_for_std = pitch[np.where(np.logical_and(time>=row.start, time<=row.end))]
        
        deviation = np.std(f0_for_std)
        f0_std_rows.append({
            'start': row.start,
            'end': row.end,
            'file_id': path.stem,
            'label': deviation
        })
        
    return pd.DataFrame(f0_std_rows)


def get_f0_diff(path, domain_path=Path('data/switchboard/phonwords'), match_w2v2=True):
    
    time, pitch =get_pitch_parselmouth(str(path))
    
    domain = pd.read_csv(domain_path / f'{path.stem}.csv')
    f0_diff_rows = list()
    for row in domain.itertuples():
        
        iter_f0 = pitch[np.where(np.logical_and(time>=row.start, time<=row.end))]
        
        first_half, second_half = np.split(iter_f0, 2)
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        mean_diff = first_mean - second_mean
        
        f0_diff_rows.append({
            'start': row.start,
            'end': row.end,
            'file_id': path.stem,
            'label': mean_diff
        })
        
    return pd.DataFrame(f0_diff_rows)
    

def get_energy_diff(path, domain_path=Path('data/switchboard/phonwords'), match_w2v2=True):
    
    sr, audio = wavfile.read(path)
    if match_w2v2:
        rms_values = rms(y=audio, frame_length=400, hop_length=320)[0]
        times = times_like(rms_values, sr=sr, hop_length=320)
    else:
        rms_values = rms(y=audio)[0]
        times = times_like(rms_values, sr=sr)
        
    domain = pd.read_csv(domain_path / f'{path.stem}.csv')
    
    energy_diff_rows = list()
    for row in domain.itertuples():
        
        iter_rms = rms_values[np.where(np.logical_and(times>=row.start, times<=row.end))]
        
        first_half, second_half = np.split(iter_rms, 2)
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)        

        energy_diff = first_mean - second_mean

        energy_diff_rows.append({
            'start': row.start,
            'end': row.end,
            'file_id': path.stem,
            'label': energy_diff
        })
        
    return pd.DataFrame(energy_diff_rows)    
 

# def get_envelope():

# def get_mean_amplitude(): ????

# def get_relative_pitch(): ????

def main():
    
    for corpus in ['switchboard', 'mandarin-timit', 'bu_radio']:
        wav_path = Path(f'data/{corpus}/wav')
        domain_folder = 'syllables' if corpus == 'switchboard' else 'phones'
        #for file in tqdm(list(wav_path.glob('*.wav'))):
        #    time, energy = get_energy(str(file), match_w2v2=True)
        #    filename = file.name.replace('.wav', '.csv')
        #    os.makedirs(f'data/{corpus}/energy', exist_ok=True)
        #    pd.DataFrame({'start': time, 'file_id': [file.name.replace('.wav', '')]*len(time), 'label': energy}).to_csv(f'data/{corpus}/energy/{filename}')
           
        #for file in tqdm(list(wav_path.glob('*.wav'))):
        #    filename = file.name.replace('.wav', '.csv')
        #    iter_df = get_f0_std(file, domain_path=Path(f'data/{corpus}/{domain_folder}'))
        #    os.makedirs(f'data/{corpus}/f0_std', exist_ok=True)
        #    iter_df.to_csv(f'data/{corpus}/f0_std/{filename}')
        
        #for file in tqdm(list(wav_path.glob('*.wav'))):
        #    for feat in ['f0', 'energy']:
        #        filename = file.name.replace('.wav', '.csv')
        #        iter_df = get_f0_std(file, domain_path=Path(f'data/{corpus}/{domain_folder}'))
        #        os.makedirs(f'data/{corpus}/{feat}_diff', exist_ok=True)
        #        iter_df.to_csv(f'data/{corpus}/{feat}_diff/{filename}')
        
        for file in tqdm(list(wav_path.glob('*.wav'))):
            
            time, value = get_pitch_parselmouth(file)
            filename = file.name.replace('.wav', '.csv')
            #time, value = func(file)
            os.makedirs(f'data/{corpus}/f0_300', exist_ok=True)
            pd.DataFrame({'start': time, 'file_id': [file.stem]*len(time), 'label': value}).to_csv(f'data/{corpus}/f0_300/{filename}')
            
             
if __name__ == '__main__':
    main()
     