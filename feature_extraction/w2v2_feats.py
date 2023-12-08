"""some functions adapted from https://github.com/Bartelds/pae_probe_experiments/blob/master/bin/gen_wav2vec_feats_hf.py"""
from transformers.models.wav2vec2 import Wav2Vec2Model
import soundfile as sf
import librosa
import torch
import numpy as np
from acoustic_feats import get_f0, get_energy
from transformers import logging


MODEL_PATH_PREPEND = {
    'wav2vec2-large-960h': 'facebook',
    'wav2vec2-large-robust': 'facebook',
    'wav2vec2-large-xlsr-53': 'facebook',
    'wav2vec2-xls-r-300m': 'facebook',
    'mandarin-wav2vec2': 'kehanlu',
    'wav2vec2-base': 'facebook',
    'mms-300m': 'facebook'
}

def get_feature_func(model_or_feat, layer=None):
    
    if model_or_feat in MODEL_PATH_PREPEND.keys():
        
        model_name = MODEL_PATH_PREPEND.get(model_or_feat, 'facebook') + '/' + model_or_feat
        
        model_kwargs = {}
        if layer is not None:
            model_kwargs["num_hidden_layers"] = layer if layer > 0 else 0
            
        model = Wav2Vec2Model(model_name, **model_kwargs)
        
        model.eval()
        if torch.cuda.is_available():
            model.cuda()       
            
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
            
    
    else:
        if model_or_feat == 'f0':
            pass
        elif model_or_feat == 'rms':
            pass
        elif model_or_feat == 'pitch-energy':
            pass
        elif model_or_feat == 'fbank':
            pass
        elif model_or_feat == 'mfcc':
            pass
        
    
    