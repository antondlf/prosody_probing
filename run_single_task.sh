#!/bin/bash

LARGE_MODELS=("wav2vec2-large" "wav2vec2-large-960h" "wav2vec2-xls-r-300m" "wav2vec2-large-xlsr-53" "wav2vec2-large-xlsr-53-chinese-zh-cn")
ACOUSTIC_BASELINES=("fbank" "mfcc" "pitch" "energy" "pitch_energy")
model=$1
feature=$2
probe=$3
corpus=$4
extra_args=$5


if [[ ${LARGE_MODELS[@]} =~ $MODEL ]]
then
    layer=24
    layer_extraction="None"
elif [[ ${ACOUSTIC_BASELINES[@]} =~ $MODEL ]]
then
    layer=0
    layer_extraction=0
else
    layer=12
    layer_extraction="None"
fi
mkdir -p logs/
echo "Extracting features for $model on $corpus"
python3 feature_extraction/w2v2_feats.py $model $layer_extraction data/$corpus/wav -c $corpus 
wait

echo "Processing $feature from $model for $corpus"

echo "$SLURM_NODENAME: Running classification experiments..."

echo "Probing $model with $probe for $layer layers" >> logs/${model}_${feature}.stdout
    python3 run_probes.py $model $layer -l data/$corpus/aligned_tasks/${feature}.csv \
    -d data/feats/$corpus -c $corpus -t $feature $5
        >> logs/${model}_${feature}_${corpus}_${probe}.stdout \
        2>> logs/${model}_${feature}_${corpus}_${probe}.stderr 