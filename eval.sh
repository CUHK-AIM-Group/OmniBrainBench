#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data_hdd/syliu/workspace/Med-VLM/hf-checkpoints
export HF_DATASETS_CACHE=/data_hdd/syliu/workspace/Med-VLM/Datasets

# if use close-set evaluation, set your openai api key and base url here
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_BASE_URL="your_openai_base_url"

#if close-set evaluation: OmniBrainBench, if open-set evaluation: OmniBrainBench-Open
EVAL_DATASETS="OmniBrainBench-Open"
DATASETS_PATH="/data_hdd/syliu/workspace/Med-VLM/Datasets/"

#For open-source models, you can choose from the following models:
#TestModel,Qwen3-VL, Qwen3-VL-Moe, Qwen2-VL,Qwen2.5-VL,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr,Hulumed-qwen3, Hulumed-qwen2.
#For commercial models, please set MODEL_NAME=GPT_Openai
MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct" # For commercial models, this is Model_name
OUTPUT_PATH="eval_results/Qwen2.5-VL-3B"
CUDA_VISIBLE_DEVICES="4"

TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"

#Eval setting 
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="True"
# gpt api model name
GPT_MODEL="gpt-5-mini-2025-08-07"
OPENAI_API_KEY="your_openai_api_key"

# pass hyperparameters and run python sccript
python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_gpt_model "$GPT_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES" 
