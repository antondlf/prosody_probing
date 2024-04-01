import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import shutil
import os
#import pympi.Praat as praat
import sys
import re
from tqdm import tqdm
from pathlib import Path
import pinyin

vowel_inventory = [
    'i', 'iii', 'ang', 'iu', 'e',
    'ing', 'en', 'iang', 'ii', 'ong',
    'v', 'ou', 'uo', 'uan', 'ui', 'eng', 'u', 'an',
    'ian', 'a', 'van', 'ai', 'uang', 've', 
    'un', 'ao', 'er','ei', 'ua', 'ia', 'in', 'ie', 'vn', 'iao',
    'iong', 'uai'
]


def get_tone_indices(x):
    m = re.findall('\d+', x)
    if m != []:
        return ''.join(m)
    else:
        return x
    

def get_pinyin(char):
    return pinyin.get(char, format='numerical')


def yield_next_tone(x):
    
    for tone in x:
        yield tone


def label_onset(row, shifter):
    
    if row.label == 0:
        try:
            return shifter.iloc[row.idx]['label']
        except KeyError:
            return row.label
    
    else:
        return row.label

def map_tone(word_df, phone_df, onset=True):
    
    #phone_and_tone = pd.DataFrame()
    phone_and_tone = list()
    for row in word_df.itertuples():
        tone_iterator = yield_next_tone(row.tone)
        df_subset = phone_df.loc[
            (phone_df.start >= row.start) &\
                (phone_df.end <= row.end)
                    ]
        df_subset['label'] = df_subset.char.map(lambda x: next(tone_iterator) if x in vowel_inventory else 'SIL' if x == 'sil' else 0)
        phone_and_tone.append(df_subset)
    phone_and_tone = pd.concat(phone_and_tone)
    
    if onset:
        # map tone to onset
        shifter = phone_and_tone.shift(-1) 
        phone_and_tone['idx'] = phone_and_tone.index
        phone_and_tone['label'] = phone_and_tone.apply(lambda x: label_onset(x, shifter), axis=1)
        phone_and_tone.drop('idx', axis=1, inplace=True)
    
    return phone_and_tone


def process_mandarin_timit(file_id_list, save_dir):
    
    word_dir = save_dir / 'words'
    phone_dir = save_dir / 'phones'
    tone_dir = save_dir / 'tone_rhymes'
    os.makedirs(word_dir, exist_ok=True)
    os.makedirs(phone_dir, exist_ok=True)
    os.makedirs(tone_dir, exist_ok=True)
    
    
    base_dir = Path('corpora/global_timit_cmn/data/segmentation')
    
    for file_id in tqdm(file_id_list):

        word_filename = f'{file_id}.words'
        phone_filename = f'{file_id}.phones'
        
        word_df = pd.read_csv(base_dir / word_filename, sep=' ', names=['char', 'start', 'end'])
        word_df['pinyin_txt'] = word_df.char.map(lambda x: pinyin.get(x, format='numerical'))
        word_df['tone'] = word_df.pinyin_txt.map(lambda x: get_tone_indices(x)) 
        
        phone_df = pd.read_csv(base_dir / phone_filename, sep=' ', names=['char', 'start', 'end']) 
        onset = False if tone_dir.name.endswith('_rhymes') else True
        tone_df = map_tone(word_df, phone_df, onset=onset)
        
        file_save = file_id + '.csv'
        tone_df.to_csv(tone_dir/ file_save, index=None)
        word_df.columns = ['label', 'start', 'end', 'pinyin', 'tone']
        phone_df.columns = ['label', 'start', 'end']
        word_df.to_csv(word_dir / file_save, index=None)
        phone_df.to_csv(phone_dir / file_save, index=None)
        
        
def main():
    
    file_id_list = [file.stem for file in Path('data/mandarin-timit/wav').glob('*.wav')]#corpora/global_timit_cmn/data/wav').glob('*.wav')]
    
    save_dir = Path('data/mandarin-timit/')
    
    process_mandarin_timit(file_id_list, save_dir)
    #shutil.move('corpora/global_timit_cmn/data/wav', 'data/mandarin-timit/wav')
    

if __name__ == '__main__':
    
    main()