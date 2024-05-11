from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
from itertools import product
from ast import literal_eval
import pandas as pd
import numpy as np


def get_phone_type(phone):
    vowels = ['ah', 'y', 'ow',  'aa',  'ae', 'ih', 'ay', 
       'eh', 'ao', 'ax',  'uw', 'v', 'f', 'er', 'ga',
        'ey', 'aw', 'iy', 'uh', 'en', 'el', 'oy']
    silence = ['SIL', 'sil']

    consonants = ['b','t', 'dh', 'n', 'd', 'l', 's', 'k', 'r', 'th',
                  'ng', 'w', 'hh', 'm', 'z', 'p', 'g', 'ch', 'jh', 'sh', 'lg','zh']
    if phone in vowels:
        return 'V'
    elif phone in consonants:
        return 'C'
    else:
        return 'sil'   
    
def get_phone_voicing(phone, language='English'):
    
    if language == 'English':
        if phone in {'b', 'dh', 'n', 'd', 'l', 'r',
                      'ng', 'w', 'm', 'z', 'g', 'jh','lg','zh'}:
            return True
        elif phone in {'sil', 'SIL'}:
            return np.nan
        else:
            return False
    else:
        vowel_inventory = [
            'i', 'iii', 'ang', 'iu', 'e',
            'ing', 'en', 'iang', 'ii', 'ong',
            'v', 'ou', 'uo', 'uan', 'ui', 'eng', 'u', 'an',
            'ian', 'a', 'van', 'ai', 'uang', 've', 
            'un', 'ao', 'er','ei', 'ua', 'ia', 'in', 'ie', 'vn', 'iao',
            'iong', 'uai'
        ]

        consonants = {
            'sil','f','q','h','g','n','ch',
            'm','sh','d','b','t','zh','z',
            'k','l','x','j','r','p', 'c','s'
        }
        voiced = {
            'g','n',
            'm','d','b','zh','z',
            'l','x','j','r'
        }

        voiceless = {'c', 'ch', 'f', 'h', 'k', 'p', 'q', 's', 'sh', 't'}
        if phone in voiced:
            return True
        elif phone in voiceless:
            return False
        else:
            return np.nan
        
    
def get_phone_merge():
    
        phone_reference = pd.read_csv('data/switchboard/aligned_tasks/phones.csv')
        
        phone_reference['phone_type'] = phone_reference.label.map(get_phone_type)
        phone_reference['phone_number'] = phone_reference.phone_type.map(lambda x: 1 if x == 'V' else 0)
        phone_reference['voicing'] = phone_reference.label.map(get_phone_voicing)
        phone_reference['start_end_indices'] = phone_reference.start_end_indices.map(literal_eval)
        
        data = pd.read_csv('data/switchboard/aligned_tasks/stress.csv')
        data['start_end_indices'] = data.start_end_indices.map(literal_eval)

        data['syllable_id'] = data.index.to_list()
        data_exploded = data.explode('start_end_indices')
        phone_exploded = phone_reference.explode('start_end_indices')
        
        merged_data = phone_exploded.merge(data_exploded, how='inner', on=['file_id', 'start_end_indices'])
        
        merged_data['onset_number'] = merged_data.groupby(['syllable_id']).phone_number.cumsum()
        merged_data['is_onset'] = merged_data.onset_number.map(lambda x: True if x == 0 else False)

        #imploded_data = merged_data.groupby(['file_id', 'syllable_id', 'is_onset', 'label_x']).agg(
        #    {
        #        'start_end_indices': lambda x: x,
        #        'voicing': lambda x: x.max(),
        #        'label_y': lambda y: y.unique()[0] if len(y.unique()) == 1 else y
        #        
        #        }).reset_index().sort_values(by='syllable_id')
        
        final_data = merged_data[['file_id', 'start_end_indices', 'is_onset', 'voicing', 'label_y']]#.loc[merged_data.is_onset == is_onset]
        final_data.columns = ['file_id', 'neural_index', 'is_onset', 'voicing', 'label'] 
        #test_results = final_data.merge(result_data[['file_id', 'neural_index', 'y_true', 'y_pred']], how='inner', on=['file_id', 'neural_index'])


        return final_data

    
def get_tone_merge():
    

    task_df = pd.read_csv('data/mandarin-timit/aligned_tasks/tone_rhymes.csv')
    phone_df = pd.read_csv('data/mandarin-timit/aligned_tasks/phones.csv')
    task_df = phone_df.merge(task_df, on=['start', 'end', 'file_id', 'start_end_indices'], suffixes=('_phone', ''))
    
    task_df['voicing'] = task_df.label_phone.map(lambda x: get_phone_voicing(x, language='Mandarin'))

    task_df['start_end_indices'] = task_df.start_end_indices.map(literal_eval)
    
    task_df['is_onset'] = task_df.label.map(lambda x: True if x == 0 else False)
    test_explode = task_df.explode('start_end_indices')[['label', 'file_id', 'is_onset', 'start_end_indices', 'voicing']]
    # Rename start_end_indices to neural_index for merge
    test_explode.columns = ['label', 'file_id', 'is_onset', 'neural_index', 'voicing']

    #results = test_explode.merge(results[['file_id', 'neural_index', 'y_true', 'y_pred']], how='inner', on=['file_id', 'neural_index'])
    return test_explode



    
def get_layerwise_scores(score, model_list, log_dir='logs_05-07-2024'):
    
    
    results = list()
    
    mandarin_frame = get_tone_merge()
    english_frame = get_phone_merge()
    
    for model in model_list:
    
        for feat, onset_selection, voicing_selection in tqdm(list(product([
            'tone', 'tone_onset', 'tone_rhyme',
            'stress', 'stress_onset', 'stress_rhyme',
            'accents', 'accents_onset', 'accents_rhyme'
        ], [True, False, 'all'], [True, False, 'all']))):
            
            layer_scores = list()
            corpus = 'mandarin-timit' if feat == 'tone' else 'switchboard'
            feat = f'syllables_{feat}' if feat.startswith('accent') else feat
            for layer in range(0,13):
                result_test = pd.read_csv(
                    f'results/{log_dir}/logs/{corpus}/{feat}/{model}/linear/layer_{layer}.csv',
                                        usecols=['file_id', 'neural_index', 'y_true', 'y_pred']
                    )
                if feat.startswith('tone'):
                    test_results = mandarin_frame.merge(result_test[['file_id', 'neural_index', 'y_true', 'y_pred']], how='right', on=['file_id', 'neural_index'])

                else:
                    #result_test = pd.read_csv(
                    #f'results/{log_dir}/logs/{corpus}/{feat}/{model}/linear/layer_{layer}.csv',
                    #                    usecols=['file_id', 'neural_index', 'y_true', 'y_pred']
                    #)
                    test_results = english_frame.merge(result_test[['file_id', 'neural_index', 'y_true', 'y_pred']], how='right', on=['file_id', 'neural_index'])

                    
                    #test_results = test_set_full.merge(result_test[['file_id', 'neural_index', 'y_true', 'y_pred']], how='inner', on=['file_id', 'neural_index'])

                if type(onset_selection) == bool:
                    scoring_df = test_results.loc[test_results.is_onset == onset_selection]
                else:
                    scoring_df = test_results
                if type(voicing_selection) == bool:
                    scoring_df = scoring_df.loc[scoring_df.voicing == voicing_selection]
                else:
                    scoring_df = scoring_df                    
                    
                #print(corpus, feat, layer, voicing_selection, onset_selection)
                if (feat.startswith('tone')) & (score is not accuracy_score):
                    performance = score(scoring_df.y_true, scoring_df.y_pred, average='macro')
                elif feat.startswith('stress'):
                    #pos_label=scoring_df.y_true.value_counts().index[-1]
                    pos_label = 0
                    performance = score(scoring_df.y_true, scoring_df.y_pred, pos_label=pos_label)
                else:
                    performance = score(scoring_df.y_true, scoring_df.y_pred)
                #if model == 'wav2vec2-base-100h':
                    #print(scoring_df)
                    #print(performance)

                results.append({
                    'feature': feat.split('_')[0],
                    'syllable_part': feat.split('_')[-1] if feat.split('_')[-1] in ['onset', 'rhyme'] else 'full',
                    'layer': layer,
                    'is_onset_eval': onset_selection,
                    'is_voiced_eval': voicing_selection,
                    'model': model,
                    'metric': 'f1' if score is f1_score else 'accuracy' if score is accuracy_score else 'precision' if score is precision_score else 'recall',
                    'score': performance,

                    })
    return pd.DataFrame(results)


def main():

    model_list = ['wav2vec2-base', 'mandarin-wav2vec2'] #, 'wav2vec2-base-100h', 'mandarin-wav2vec2-aishell1', 'hubert-base-ls960', 'wavlm-base']
    log_dir = 'logs_05-07-2024'
    onset_data = get_layerwise_scores(f1_score, model_list, log_dir='logs_05-07-2024')
    onset_data.to_csv('onset_rhyme_results.csv')


if __name__ == '__main__':
    main()
