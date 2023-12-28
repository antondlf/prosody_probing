import subprocess
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert2wav(
    original_format, corpus_dir,
    save_dir, corpus_name,
    recursive_iter=False
    ):
    os.makedirs(save_dir, exist_ok=True)
    glob_string = f'*.{original_format}' 
    if recursive_iter:
        iterator = list(corpus_dir.rglob(glob_string))
    else:
        iterator = list(corpus_dir.glob(glob_string))
    accent_file_ids = [
        file.split('.')[0][:2] + '0' + file.split('.')[0][2:]\
            for file in os.listdir('corpora/nxt_switchboard_ann/xml/accent')
    ]
    print(iterator)
    print(f'Converting {corpus_name} to wav...')
    for file in tqdm(iterator):
        if file.stem in accent_file_ids:
            new_file = save_dir / file.name
            
            command = [
                'ffmpeg',
                '-i',
                file,
                new_file.with_suffix('.wav')           
            ]
            subprocess.call(command)
    
    return
        
        
def split_switchboard_channels(wav_dir, save_dir, corpus_name):
    os.makedirs(save_dir, exist_ok=True)
    print(f'Splitting wav files for {corpus_name}')
    for file in tqdm(list(wav_dir.glob('*.wav'))):
        fileout = Path(save_dir, file.stem[:2] + file.stem[3:])
        command1 = [
            'ffmpeg', '-y', '-i', file,
            '-af', "pan=mono|FC=FL",
            f'{str(fileout)}A.wav']
        command2 = [
            'ffmpeg', '-y', '-i', file, '-af', 
            "pan=mono|FC=FR", f'{str(fileout)}B.wav'
            ]
        command_del = ['rm', 'file']
        subprocess.call(command1)
        subprocess.call(command2)
        subprocess.call(command_del)


def resample_audio(audio_dir, save_dir, original_rate=8000, target_rate=16000):
    os.makedirs(save_dir, exist_ok=True)
    for file in audio_dir.glob('*.wav'):
        
        command = [
            'ffmpeg',
            '-i',
            file,
            '-isr',
            str(original_rate),
            '-ar',
            str(target_rate),
            '-osr',
            str(target_rate),
            save_dir / file.name
        ]
        subprocess.call(command)
    
    


def process_switchboard_audio():
    
    corpora = ['switchboard']
    resample_corpora = {'switchboard': 8000}
    
    corpus_args = {
        'switchboard': ('sph',
                        Path('corpora/swb1_LDC97S62'),
                        Path('corpora/nxt_switchboard_ann/raw_wav'),
                        True,
                        Path('corpora/nxt_switchboard_ann/split_wav'),
                        Path('corpora/nxt_switchboard_ann/wav')
        )
    }
   
    for corpus in corpora:
        resample = True if corpus in resample_corpora.keys() else False
        original_format,\
        corpus_dir,\
        save_dir,\
        recursive_iter,\
        split_save,\
            clean_save = corpus_args[corpus]
        convert2wav(
            original_format, corpus_dir, save_dir,
            corpus, recursive_iter=recursive_iter
            )
        if corpus == 'switchboard':
            split_switchboard_channels(save_dir, split_save, corpus)
            shutil.rmtree(save_dir)
            resample_audio(split_save, clean_save, original_rate = resample_corpora[corpus])
            shutil.rmtree(split_save)
            
        elif resample:
            
            resample_audio(save_dir, clean_save, original_rate = resample_corpora[corpus])
            shutil.rmtree(save_dir)
            
        
       
if __name__ == '__main__':
    
    process_switchboard_audio()
    
    
    