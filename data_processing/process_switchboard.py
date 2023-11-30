import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import os
#import pympi.Praat as praat
import sys
import re
from tqdm import tqdm
from pathlib import Path

root_dir = 'annotations'
###################################################################################
###################################################################################
##########  This file includes xml to csv format, extaction of simple   ###########
##########  acoustic features from each wav file into csv format.        ###########
###################################################################################
###################################################################################

tag_mapping = {
   'accent': ['accent', 'pointer'],
   'phones': ['ph', 'None'],
   'breaks': ['break', 'pointer'],
   'phonwords': ['phonword', 'child'],
   'syllables': ['syllable', 'child'],
   'turns': ['turn', 'child'],
   'phrase': ['phrase', 'child']
   }


def get_root(directory):
   return ET.parse(directory).getroot()


def xml2clean_csv(file_id, corpus_dir, feature_name, start_seconds, end_seconds, labels):
    
    save_dir = Path(corpus_dir, feature_name, file_id).with_suffix('.csv')
    
    return pd.DataFrame(
        {'start': start_seconds, 'end': end_seconds, 'label': labels}
        ).to_csv(save_dir)
    


def xml2df(
    file,
    feat
):
    """Turns xml from switchboard to pandas df with
    urls cleaned up."""
    data_list = list()
    root = get_root(file)
    for elem in root:
        if elem.tag == tag_mapping[feat][0]:
            data_list.append(elem.attrib)
        for child in elem:
            tag = child.tag.replace('{http://nite.sourceforge.net/}', '')
            if tag == tag_mapping[feat][-1]:
                data_list[-1].update(child.attrib)               
        
    return pd.DataFrame(data_list)

    
def xml2raw_csv(
    xml_dir, save_dir,
    feature_list, tag_mapping,
    accent_file_ids
    ):
    
    xml_dir = Path('corpora/nxt_switchboard_ann/xml/')
    save_dir = Path('corpora/nxt_switchboard_ann/csv')
    os.makedirs(save_dir, exist_ok=True)
    for feat in feature_list:
        directory = xml_dir / feat
        print(f"Processing {feat}...")
        for file in tqdm(list(directory.glob('*.xml'))):
            save_stem = ''.join(file.name.split('.')[:2])
            root = get_root(file)
            #if save_stem in accent_file_ids:
            data_list = list()
            for elem in root:
                if elem.tag == tag_mapping[feat][0]:
                    data_list.append(elem.attrib)
                for child in elem:
                    tag = child.tag.replace('{http://nite.sourceforge.net/}', '')
                    if tag == tag_mapping[feat][-1]:
                        #print(child)
                        data_list[-1].update(child.attrib)               
            
            df = pd.DataFrame(data_list)

            df.columns = [
                col.replace('{http://nite.sourceforge.net/}', '') for col in df.columns
                ]
            os.makedirs(save_dir / feat, exist_ok=True)
            save_path = save_dir / feat / save_stem
            df.to_csv(save_path.with_suffix('.csv'), index=False)


def preprocess_switchboard():
    
    feature_list = [
    'accent', 'breaks', 'phones', 
    'phonwords', 'syllables', 'turns',
    'phrase'
    ]
    tag_mapping = {
    'accent': ['accent', 'pointer'],
    'phones': ['ph', 'None'],
    'breaks': ['break', 'pointer'],
    'phonwords': ['phonword', 'child'],
    'syllables': ['syllable', 'child'],
    'turns': ['turn', 'child'],
    'phrase': ['phrase', 'child']
    } 
    accent_file_ids = [
        ''.join(file.split('.')[:2])\
            for file in os.listdir('corpora/nxt_switchboard_ann/xml/accent')
    ]
    xml_dir = Path('corpora/nxt_switchboard_ann/xml/')
    save_dir = Path('corpora/nxt_switchboard_ann/csv') 
    xml2raw_csv(
        xml_dir, save_dir,
        feature_list, tag_mapping,
        accent_file_ids
        )
    
    # Create a directory 
    # - Data
    #   - switchboard
    #       - wav
    #       - <annotations>
    
    #
    
if __name__ == '__main__':
    preprocess_switchboard()  