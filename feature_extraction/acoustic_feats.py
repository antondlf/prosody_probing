import crepe
import numpy as np
from scipy.io import wavfile
from librosa.feature import rms
from librosa import times_like
from lhotse import Fbank, FbankConfig, Mfcc, MfccConfig
import parselmouth as prsl
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm


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
    
    snd = prsl.Sound(path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    return pitch.xs(), pitch_values


def get_pitch_energy(path):
    
    pitch_time, pitch = get_f0(path)
    rms_time, rms = get_energy(path, match_w2v2=True)
    
    return pitch_time, np.vstack((pitch, rms))
    
    
    

# def get_intensity():

# def get_envelope():

# def get_mean_amplitude(): ????

# def get_relative_pitch(): ????

def main():
    
    for corpus in ['mandarin-timit']: #, switchboard]:
        wav_path = Path(f'data/{corpus}/wav')
        
        for file in tqdm(list(wav_path.glob('*.wav'))):
            time, energy = get_energy(str(file), match_w2v2=True)
            filename = file.name.replace('.wav', '.csv')
            os.makedirs(f'data/{corpus}/energy', exist_ok=True)
            pd.DataFrame({'start': time, 'file_id': [file.name.replace('.wav', '')]*len(time), 'label': energy}).to_csv(f'data/{corpus}/energy/{filename}')
           
           
if __name__ == '__main__':
    main()
     