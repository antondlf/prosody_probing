#!/bin/bash

set -e  # Exit on error


current_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo $current_path
DATA_DIR="$current_path/../../data/"
FEATS_DIR="$DATA_DIR/feats/"
#MODEL_NAMES="wav2vec2-large-robust"
MODEL_NAMES="wav2vec2-base mandarin-wav2vec2"
LAYER="all"
PROBES="mlp linear"
CORPORA="switchboard mandarin-timit"
FEATURES="f0 phones_accents phonwords_accents syllables_accents"
##############################################################################
# Configuration
##############################################################################
nj=-1   # Number of parallel jobs for CPU operations.
stage=1
gpu=4

mkdir -p logs/


##############################################################################
# Extract features
##############################################################################
if [ $stage -le 0 ]; then
  for corpus in $CORPORA; do
	for model in $MODEL_NAMES; do
		python3 feature_extraction/w2v2_feats.py $model 'None' data/$corpus/wav -c $corpus
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
  for corpus in $CORPORA; do
    if [ $corpus == 'mandarin-timit' ]; then
      FEATURES="f0 tone"
    fi
    for model in $MODEL_NAMES; do
        for feature in $FEATURES; do
        for probe in $PROBES; do
        for i in $(seq 0 1 13); do
          echo "Processing $feature from $model layer $i"

          echo "$0: Running classification experiments..."

          echo "$model $probe layer $i regression" >> logs/${model}_${feature}.stdout
              python3 run_probes.py $model i -l data/$corpus/aligned_tasks/$feature \
             -d data/feats/$corpus -c $corpus -t $feature -r True -p $probe -g $gpu
                  >> logs/${model}_${feature}_${probe}.stdout \
                  2>> logs/${model}_${feature}_${probe}.stderr &
    	  wait
    	  done
    	  done
    done
    done
  done
fi