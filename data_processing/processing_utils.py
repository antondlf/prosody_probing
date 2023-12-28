import os
from pathlib import Path


def check_files(wav_path, corpus, check_folder_list):
    
    missing_files = list()
    for wave in wav_path.glob('*.wav'):
        
        for feat in check_folder_list:
            check_path = f'data/{corpus}/{feat}/{wave.stem}.csv'
            if not os.path.isfile(check_path):
                
                missing_files.append(check_path)
                

    return missing_files
        


def check4mising_files(check_folder_lists):
    
    missing_file_dict = dict()
    
    for corpus, folder_list in check_folder_lists.items():
        
        wav_path = Path(f'data/{corpus}/wav')
        missing_file_dict[corpus] = check_files(wav_path, corpus, folder_list)
        
    print('The following files are missing:')
    print() 
    for corpus, missing_list in missing_file_dict.items():
        print(f'For {corpus}')
        print()
        if len(missing_list) > 0:
            print('\t', '\n\t'.join(missing_list))
            print()
        else:
            print('\tNo files are missing')
            print()
    print()  
        
        
if __name__ == '__main__':
    
    check_folder_lists = {
    'switchboard': ['accent', 'breaks', 'phones', 'phonwords', 'phrase', 'syllables'],
    'mandarin-timit': ['tone']
    }
    
    check4mising_files(check_folder_lists)