from praatio import textgrid
from pympi.Praat import TextGrid
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import os
from tqdm import tqdm
import math

def ms2idx(time_s, step_s=0.02):
    time_ms = int(time_s * 1000)
    step_ms = int(step_s * 1000)
    return ((time_ms - (time_ms%step_ms))/ step_ms)


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



def get_yemba_data(root):
    

    data_rows = list()

    for file in tqdm(list(root.glob('**/*.TextGrid'))):
        
        parts = file.parts
        
        speaker = parts[-3]
        group = parts[-2]
        filename = parts[-1]
        statement = f"statement_{file.stem.split('_')[-1]}"
        #grid = TextGrid(file)
        try:
            grid = textgrid.openTextgrid(file, includeEmptyIntervals=False)
        except Exception:
            print(file, 'failed')
            #break
        syllables = grid.getTier('syllable').entries
        tones = grid.getTier('ton').entries
        try:
            
            audio_sample, sr = librosa.load(file.with_suffix('.wav'), sr=16000)
            frame_length = get_frames_from_samples(int(audio_sample.shape[0]))
            
            for syllable, tone in zip(syllables, tones):
                
                index_list = list(range(int(ms2idx(syllable.start)), int(ms2idx(syllable.end))))
                if len(index_list) > 0:
                    if index_list[-1] > frame_length-1:
                        while frame_length-1 < index_list[-1]:
                            index_list = index_list[:-1]
                
                    data_rows.append(
                        {
                            'file_id': file.stem,
                            'speaker': speaker,
                            'group': group,
                            'statement': statement,
                            'start': syllable.start,
                            'end': syllable.end,
                            'syllable': syllable.label,
                            'tone': tone.label,
                            'start_end_indices': index_list
                        }
                    )
        except Exception:
            print(file, 'wav file did not exist')
            continue
            
    return pd.DataFrame(data_rows)


def main():
    
    root = Path('corpora/Yemba_Dataset/audios')
    step = 0.02
    data = get_yemba_data(root) 
    
    
    os.makedirs('data/yemba/wav', exist_ok=True)
    os.makedirs('data/yemba/tone', exist_ok=True)
    os.makedirs('data/yemba/aligned_tasks/', exist_ok=True)

    # There is a reading error where one of the 'bas' labeled tones
    # is labeled 'b'.
    data['tone'] = data.tone.map(lambda x: 'bas' if x == 'b' else x)
    data_save = data[['file_id', 'start', 'end', 'tone', 'start_end_indices']]
    
    data_save.columns = ['file_id', 'start', 'end', 'label', 'start_end_indices'] 
    
    
    for file in tqdm(list(root.glob('**/*.wav'))):
        try:
            audio, sr = librosa.load(path=str(file), sr=16000)
            sf.write(f'data/yemba/wav/{file.stem}.wav', audio, sr)
        except Exception:
            print(file, 'failed')
            data_save = data_save.loc[data_save.file_id != file.stem]
            
    data_save.to_csv('data/yemba/aligned_tasks/tone.csv')
if __name__ == '__main__':
    main()
        