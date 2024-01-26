from pathlib import Path
import pandas as pd
import re
import os


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
    
        if row[1].phone.startswith('>'):
            for seg in segment_rows:
                seg['word_label'] = row[1].phone
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
        
    
if __name__ == '__main__':
    iterate_bu_corpus()