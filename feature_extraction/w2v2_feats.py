"""some functions adapted from https://github.com/Bartelds/pae_probe_experiments/blob/master/bin/gen_wav2vec_feats_hf.py"""
from transformers.models.wav2vec2 import Wav2Vec2Model
from transformers.models.hubert import HubertModel
from transformers.models.wavlm import WavLMModel
import soundfile as sf
import librosa
import torch
import numpy as np
from acoustic_feats import get_f0, get_energy, get_fbank, get_mfcc, get_pitch_energy, get_pitch_parselmouth
from transformers import logging
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import os


MODEL_PATH_PREPEND = {
    'wav2vec2-large': 'facebook',
    'wav2vec2-large-robust': 'facebook',
    'wav2vec2-large-xlsr-53': 'facebook',
    'wav2vec2-xls-r-300m': 'facebook',
    'mandarin-wav2vec2': 'kehanlu',
    'wav2vec2-base': 'facebook',
    'mms-300m': 'facebook',
    'mandarin-wav2vec2-aishell1': 'kehanlu',
    'wav2vec2-base-100h': 'facebook',
    'wav2vec2-large-xlsr-53-chinese-zh-cn': 'jonatasgrosman',
    'wav2vec2-large-960h': 'facebook',
    'hubert-base-ls960': 'facebook',
    'hubert-large': 'facebook',
    'wavlm-base': 'microsoft',
    'wavlm-large': 'microsoft'
}

MODEL_EXTRACTOR_CLASS = {
    
    'wav2vec2-large': Wav2Vec2Model,
    'wav2vec2-large-robust': Wav2Vec2Model,
    'wav2vec2-large-xlsr-53': Wav2Vec2Model,
    'wav2vec2-xls-r-300m': Wav2Vec2Model,
    'mandarin-wav2vec2': Wav2Vec2Model,
    'wav2vec2-base': Wav2Vec2Model,
    'mms-300m': Wav2Vec2Model,
    'mandarin-wav2vec2-aishell1': Wav2Vec2Model,
    'wav2vec2-base-100h': Wav2Vec2Model,
    'wav2vec2-large-xlsr-53-chinese-zh-cn':Wav2Vec2Model,
    'wav2vec2-large-960h': Wav2Vec2Model,
    'hubert-base-ls960': HubertModel,
    'hubert-large-ls960': HubertModel, 
    'wavlm-base': WavLMModel,
    'wavlm-large': WavLMModel,
}

def get_feature_func(model_or_feat, layer=None):
    
    print(model_or_feat)
    if model_or_feat in MODEL_PATH_PREPEND.keys():
        
        model_name = MODEL_PATH_PREPEND.get(model_or_feat, 'facebook') + '/' + model_or_feat
        
        model_kwargs = {}
        if layer is not None:
            model_kwargs["num_hidden_layers"] = layer if layer > 0 else 0
            
        extractor_class = MODEL_EXTRACTOR_CLASS.get(model_or_feat, Wav2Vec2Model)
        model = extractor_class.from_pretrained(model_name, **model_kwargs)
        
        model.eval()
        if torch.cuda.is_available():
            model.cuda()       
        @torch.no_grad()
        def _featurize(path):
            
            input_values, rate = sf.read(path, dtype=np.float32)
            assert rate == 16_000
            input_values = torch.from_numpy(input_values.astype(np.float32)).unsqueeze(0)
            if torch.cuda.is_available():
                input_values = input_values.cuda()

            if layer is None:
                hidden_states = model(
                    input_values, output_hidden_states=True).hidden_states
                hidden_states = [s.squeeze(0).cpu().numpy() for s in hidden_states]
                return hidden_states

            if layer >= 0:
                hidden_state = model(
                    input_values
                ).last_hidden_state.squeeze(0).cpu().numpy()
            else:
                hidden_state = model.feature_extractor(input_values)
                hidden_state = hidden_state.transpose(1, 2)
                if layer == -1:
                    hidden_state = model.feature_projection(hidden_state)
                hidden_state = hidden_state.squeeze(0).cpu().numpy()

            return hidden_state
        return _featurize 
    
    else:
        if model_or_feat == 'pitch':
            return get_f0 #get_pitch_parselmouth

        elif model_or_feat == 'energy':
            return get_energy
        elif model_or_feat == 'pitch-energy':
            return get_pitch_energy
        elif model_or_feat == 'fbank':
            return get_fbank
        elif model_or_feat == 'mfcc':
            return get_mfcc


def main():
    
    parser = ArgumentParser(
        'Extract features from audio files'
    )       
    parser.add_argument(
        'model', type=str, help='name of model to extract features'
    )
    parser.add_argument(
        'layer', type=str, help='layer to extract features. If "all", all layers, if "none" then baseline.'
    )
    parser.add_argument(
        'wavepath', type=Path, help='Path to wav files to process.'
    )
    parser.add_argument(
        '-s', type=Path, default='data/feats', help="path to save features to."
    )
    parser.add_argument(
        '-c', type=str, default='switchboard', help='corpus name'
    )
    
    args = parser.parse_args()
    
    wav_path = args.wavepath
    layer = int(args.layer) if (args.layer != 'all') and (args.layer != 'None') else args.layer
    feat_save = args.s / args.c
    os.makedirs(feat_save, exist_ok=True)
    
    if layer == 'None':
        featurizer = get_feature_func(args.model, layer=None)
    else:
        featurizer = get_feature_func(args.model, layer=layer)
        
    for file in tqdm(list(wav_path.glob('*.wav'))):
        if layer != 'None':
            if args.model in ['fbank', 'mfcc', 'pitch_energy', 'pitch', 'energy']:
                file_save = feat_save / args.model / f'layer-{0}' / f'{file.stem}.npy'
                os.makedirs(file_save.parent, exist_ok=True)
                if not file_save.exists():
                    np.save(file_save, featurizer(file))
            else:
                file_save = feat_save / args.model / f'layer-{str(layer)}' / f'{file.stem}.npy'
                os.makedirs(file_save.parent, exist_ok=True)
                if not file_save.exists():
                    np.save(file_save, featurizer(file))
        else:
            hidden_states = featurizer(file)
            for i, state in enumerate(hidden_states):
                file_save = feat_save / args.model / f'layer-{str(i)}' / f'{file.stem}.npy'
                os.makedirs(file_save.parent, exist_ok=True)
                if not file_save.exists():
                    np.save(file_save, state) 
                
                
        
if __name__ == '__main__':
    main()