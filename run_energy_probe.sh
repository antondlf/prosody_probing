#!/bin/bash

set -e  # Exit on error


current_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo $current_path
DATA_DIR="$current_path/../../data/"
FEATS_DIR="$DATA_DIR/feats/"
#MODEL_NAMES="wav2vec2-large-robust"
MODEL_NAMES="wav2vec2-base mandarin-wav2vec2 wav2vec2-base-100h mandarin-wav2vec2-aishell1 mfcc fbank pitch energy pitch-energy"
LAYER="all"
PROBES=$2
CORPORA="mandarin-timit switchboard"
FEATURES="stress syllables_accents f0 energy energy_std f0_std"
##############################################################################
# Configuration
##############################################################################
nj=-1   # Number of parallel jobs for CPU operations.
stage=$1
gpu=4

mkdir -p logs/

##############################################################################
# Extract features
##############################################################################
if [ $stage -le 0 ]; then
  for corpus in $CORPORA; do
	for model in $MODEL_NAMES; do
      python3 feature_extraction/w2v2_feats.py $model 0 data/$corpus/wav -c $corpus 
	done
    done

fi

#if [ $stage -le 1 ]; then
#	for model in $MODEL_NAMES; do
#		python3 $current_path/../../bin/featurize_corpus.py -i $DATA_DIR/switchboard/wav/ -o $FEATS_DIR/{model}/layer-{layer}/ -m "mandarin-wav2vec2-aishell1" -l "base"
#	done
#fi

##############################################################################
# Run classification tasks.
##############################################################################
if [ $stage -le 1 ]; then
    for probe in $PROBES; do
    for model in $MODEL_NAMES; do
        if [ $model == 'wav2vec2-large' ] || [ $model == "wav2vec2-xls-r-300m" ] ||\
         [ $model == "wav2vec2-large-xlsr-53" ] || [ $model == "wav2vec2-large-xlsr-53-chinese-zh-cn" ] ||\
          [ $model == "wav2vec2-large-960h" ]; then
          layer=24
        elif [ $model == "fbank" ] || [ $model == "mfcc" ] || [ $model == "pitch" ]; then
          layer=0
        else
          layer=12
        fi
    for corpus in $CORPORA; do
        if [ $corpus == 'mandarin-timit' ]; then #&& [ [ $model == 'wav2vec2-large' ] || [ $model == 'wav2vec2-xls-r-300m' ] ]; then
            FEATURES="crepe-f0" #intensity" #energy"
        
        elif [ $corpus == 'switchboard' ]; then
            FEATURES="crepe-f0" #intensity" #energy"

        fi
        for feature in $FEATURES; do
          echo "Processing $feature from $model for $corpus"

          echo "$0: Running classification experiments..."
        #for l in $(seq 1 1); do
          echo "Probing $model with $probe for layer $layer" >> logs/${model}_${feature}.stdout
              python3 run_probes.py $model 0 -l data/$corpus/aligned_tasks/${feature}.csv \
             -d data/feats/$corpus -c $corpus -t $feature -r True -p $probe #--gpu_count #$gpu #$3
                  >> logs/${model}_${feature}_${corpus}_${probe}.stdout \
                  2>> logs/${model}_${feature}_${corpus}_${probe}.stderr &
    	  wait
        #done
    	  done
    done
    done
  done
fi