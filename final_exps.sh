#!/usr/bin/env bash

# Check if API keys are set
if [[ -z ${OPENAI_API_KEY} ]]; then
    echo "Set OPENAI_API_KEY environment variable before running the experiment"
    exit 1
fi

if [[ -z ${GMAPS_API_KEY} ]]; then
    echo "Set GMAPS_API_KEY environment variable before running the experiment"
    exit 1
fi

DATA_DIR=Data/
if [ ! -d ${DATA_DIR} ]; then
    echo "Ensure data directory ${DATA_DIR} exists before running the experiment"
    exit 1
fi

# 20 percent original run, 1994 seed, all defaults etc 
dataset_name=$1
output_dir=final_results/20percent

# Run baselines
python final_exps.py \
    --output_dir ${output_dir} \
    --dataset_name ${dataset_name} \
    --percent_labeled 20 \
    --seed 1994 \
    --run_knn \
    --run_clip_zero \
    --run_clip_supervised \

# Run AiSciVision ablations
python final_exps.py \
    --output_dir ${output_dir} \
    --dataset_name ${dataset_name} \
    --percent_labeled 20 \
    --seed 1994 \
    --run_gpt_alone \
    --run_gpt_tools \
    --run_gpt_visrag \
    --run_gpt_visrag_tools
