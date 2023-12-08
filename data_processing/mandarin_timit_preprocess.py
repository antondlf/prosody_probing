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


def map_tone(word_df, phone_df):
    
    phone_and_tone = pd.DataFrame()
    for row in word_df.itertuples():
        tone_iterator = yield_next_tone(row.tone)
        df_subset = phone_df.loc[
            (phone_df.start >= row.start) &\
                (phone_df.end <= row.end)
                    ]
        df_subset['tone'] = df_subset.char.map(lambda x: next(tone_iterator) if x in vowel_inventory else 0)
        phone_and_tone = pd.concat([phone_and_tone, df_subset])
    
    return phone_and_tone


def process_mandarin_timit(file_id_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    base_dir = Path('corpora/global_timit_cmn/data/segmentation')
    
    for file_id in tqdm(file_id_list):

        word_filename = f'{file_id}.words'
        phone_filename = f'{file_id}.phones'
        
        word_df = pd.read_csv(base_dir / word_filename, sep=' ', names=['char', 'start', 'end'])
        word_df['pinyin_txt'] = word_df.char.map(lambda x: pinyin.get(x, format='numerical'))
        word_df['tone'] = word_df.pinyin_txt.map(lambda x: get_tone_indices(x)) 
        
        phone_df = pd.read_csv(base_dir / phone_filename, sep=' ', names=['char', 'start', 'end']) 
        
        tone_df = map_tone(word_df, phone_df)
        
        file_save = file_id + '.csv'
        tone_df.to_csv(save_dir / file_save, index=None)
        
        
def main():
    
    file_id_list = [file.stem for file in Path('corpora/global_timit_cmn/data/wav').glob('*.wav')]
    
    save_dir = Path('data/mandarin-timit/tone')
    
    process_mandarin_timit(file_id_list, save_dir)
    shutil.move('corpora/global_timit_cmn/data/wav', 'data/mandarin-timit/wav')
    

if __name__ == '__main__':
    
    main()