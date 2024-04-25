from pathlib import Path
import pandas as pd
import re
import os
from tqdm import tqdm
import shutil


def load_tab_delim(path, col_list):
    
    return pd.read_csv(path, sep='\t', names=col_list)


def load_whitespace_delim(path, col_list):
   
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [re.sub('[ ]+', '\t', re.sub('; .+', '', line).strip()).split('\t') for line in lines[lines.index('#\n')+1:]]
    
    if len(lines) > 1:
        if len(lines[0]) == len(col_list):
            return pd.DataFrame(lines, columns=col_list)
        elif len(lines[1]) == len(col_list):
            return pd.DataFrame(lines[1:], columns=col_list)

    print(lines)
    return None


def save_file(df, save_path):
    
    df.to_csv(save_path)
    
    
def get_word_alignments(df):
    
    final_rows = list()
    segment_rows = list()
    for row in df.iterrows():
    
        if row[1].phone_label.startswith('>'):
            for seg in segment_rows:
                seg['word_label'] = row[1].phone_label
            final_rows.extend(segment_rows)
            segment_rows = list()
        else:
            segment_rows.append(row[1])
        
    return pd.DataFrame(final_rows) 

    
def iterate_bu_corpus():
    
    bu_dir = Path('corpora/bu_radio/data')
    
    for feat in ['tobi', 'word', 'phone']:
        os.makedirs(f'corpora/bu_radio/clean_data/{feat}', exist_ok=True)
    
    # Get tone annotations
    for file in bu_dir.glob('**/*.ton'):
        save_path = f'corpora/bu_radio/clean_data/tobi/{file.stem}.csv'
        file_df = load_whitespace_delim(file, ['timestamp_ms', 'unk', 'label'])
        if file_df is not None:
            save_file(file_df, save_path)
        
    #for file in bu_dir.glob('**/*.wrd'):
    #    save_path = f'corpora/bu_radio/clean_data/word/{file.stem}.csv' 
    #    file_df = load_whitespace_delim(file, ['timestamp_s', 'unk', 'label'])
    #    if file_df is not None:
    #        save_file(file_df, save_path)
        
    for file in bu_dir.glob('**/*.ala'):
        save_path = f'corpora/bu_radio/clean_data/phone/{file.stem}.csv'
        file_df = load_tab_delim(file, col_list=['phone_label', 'timestamp_ms', 'duration_ms'])
        file_df = get_word_alignments(file_df)
        save_file(file_df, save_path)


def get_in_between(start, end, data):
    #print(start, end)
    #print(data.loc[(data.timestamp_ms >= start) & (data.timestamp_ms <= end)&(data.label.str.contains('\*'))])
    
    label_chunk = data.loc[(data.timestamp_ms >= start) & (data.timestamp_ms <= end)&(data.label.str.endswith('*'))]

    if len(label_chunk) > 0:
        return label_chunk.label.item()
    else:
        return '0'    


def reformat_data(binary=True):

    root = Path('corpora/bu_radio/clean_data/phone')
    for domain in ['phones', 'words']:
        os.makedirs(f'data/bu_radio/{domain}_accents', exist_ok=True)
    for file in tqdm(list(root.glob('*.csv'))):
        #print(file)
        try:
            tobi = pd.read_csv(f'corpora/bu_radio/clean_data/tobi/{file.name}')
            phones = pd.read_csv(file)
            phones['start'] = phones.timestamp_ms.map(lambda x: float(x)/1000)
            phones['end'] = phones.apply(lambda x: (int(x.timestamp_ms) + int(x.duration_ms))/1000, axis=1)
            
            phones = phones.loc[phones.phone_label != 'H#']

            phones['label'] = phones.apply(lambda x: get_in_between(x.start, x.end, tobi), axis=1)
            if binary:
                phones['label'] = phones.label.map(str)
                phones['label'] = phones.label.map(lambda x: 1 if x != '0' else 0)
                save_folder = 'phones_accents'
            else:
                save_folder = 'phones_multiaccent'
                
            phones[['start', 'end', 'label']].to_csv(f'data/bu_radio/{save_folder}/{file.stem}.csv')
            #phones.groupby('word').agg('max').reset_index().to_csv(f'data/bu_radio/words_accents/{file.stem}.csv')
            
        except FileNotFoundError:
            pass
        

def copy_audio():
    
    root = Path('corpora/bu_radio/data')
    destination_dir = Path('data/bu_radio/wav')
    os.makedirs(destination_dir, exist_ok=True)
    for file in tqdm(list(root.rglob('*.wav'))):
        
        shutil.copy(file, destination_dir / file.name)
        
    
if __name__ == '__main__':

    iterate_bu_corpus()
    reformat_data(binary=True)
    copy_audio()