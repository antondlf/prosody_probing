import crepe
import numpy as np
from scipy.io import wavfile
from librosa.feature import rms
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
        mean_frequency = np.concatenate(np.mean(frequency[:-1].reshape(-1, 2), axis=1), frequency[-1])
    
    downsample_time = time[0::2]
    
    return downsample_time, mean_frequency

def get_energy(path, match_w2v2=True):

    _sr, audio = wavfile.read(path)
    if match_w2v2:
        return rms(y=audio, frame_length=400, hop_length=320)
    else:
        return rms(y=audio)
        

def get_fbank(path, nbins=80):

    sr, audio = wavfile.read(path)
    extractor = Fbank(FbankConfig(num_mel_bins=nbins)) 
    return extractor.extract(audio, sample_rate=sr)


def get_mfcc(path, nbins=80):
    sr, audio = wavfile.read(path)
    extractor = Mfcc(MfccConfig(num_mel_bins=nbins)) 
    return extractor.extract(audio, sample_rate=sr)   


def get_pitch_parselmouth(path):
    
    snd = prsl.Sound(path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    return pitch.xs(), pitch_values
    
    

# def get_intensity():

# def get_envelope():

# def get_mean_amplitude(): ????

# def get_relative_pitch(): ????

def main():
    
    for corpus in ['switchboard', 'mandarin-timit']:
        wav_path = Path(f'data/{corpus}/wav')
        
        for file in tqdm(list(wav_path.glob('*.wav'))):
            time, pitch = get_pitch_parselmouth(str(file))
            filename = file.name.replace('.wav', '.csv')
            os.makedirs(f'data/{corpus}/f0', exist_ok=True)
            pd.DataFrame({'start': time, 'file_id': [file.stem]*len(time), 'label': pitch}).to_csv(f'data/{corpus}/f0/{filename}')
           
           
if __name__ == '__main__':
    main()
     