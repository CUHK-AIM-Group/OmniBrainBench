import os
import torch
import random
import numpy as np
from tqdm import tqdm  
import json
from argparse import ArgumentParser
from LLMs import init_llm
import gc
import multiprocessing
import sys
import traceback

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_eval_datasets(datasets_str):
    return datasets_str.split(',')

def main():
    parser = ArgumentParser()
    parser.add_argument('--eval_datasets', type=parse_eval_datasets, default=['MMMU'],
                    help='name of eval dataset')
    parser.add_argument('--datasets_path', type=str, default="benchmarks",
                    help='path of eval dataset')
    parser.add_argument('--output_path', type=str, default='eval_results/Qwen2-VL-7B-Instruct',
                        help='name of saved json')
    parser.add_argument('--model_name', type=str, default='Qwen2-VL-7B-Instruct',
                        help='name of model')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_visible_devices', type=str, default=None)
    parser.add_argument('--tensor_parallel_size', type=str, default="1")
    parser.add_argument('--use_vllm', type=str, default="True")
    parser.add_argument('--reasoning', type=str, default="False")

    parser.add_argument('--num_chunks', type=str, default="1")
    parser.add_argument('--chunk_idx', type=str, default="0")

    parser.add_argument('--max_image_num', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--repetition_penalty', type=float, default=1)

    parser.add_argument('--test_times', type=int, default=1)

    parser.add_argument('--use_llm_judge', type=str, default="False")
    parser.add_argument('--judge_gpt_model', type=str,default = "None")

    parser.add_argument('--openai_api_key', type=str,default = "None")


    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"

    # llm judge setting
    if args.openai_api_key == "None" and args.use_llm_judge == "True":
        raise ValueError("If you want to use llm judge, please set the openai api key")
    

    os.environ["judge_gpt_model"] = args.judge_gpt_model
    os.environ["use_llm_judge"] = args.use_llm_judge
    os.environ["openai_api_key"] = args.openai_api_key

    # eval modle setting
    os.environ["REASONING"] = args.reasoning
    os.environ["use_vllm"] = args.use_vllm

    os.environ["max_image_num"] = str(args.max_image_num)


    # vllm and torch setting
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.use_vllm == "True":
        os.environ["tensor_parallel_size"] = args.tensor_parallel_size
        if int(args.tensor_parallel_size) > 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        torch.multiprocessing.set_start_method('spawn')
        os.environ["num_chunks"] = args.num_chunks
        os.environ["chunk_idx"] = args.chunk_idx


    os.makedirs(args.output_path, exist_ok=True)

    print('initializing LLM...')
    try:
        model = init_llm(args)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Failed to initialize LLM")
        print(f"{'='*80}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        sys.exit(1)
    
    total_results_path = os.path.join(args.output_path,'total_results.json')

    from benchmarks import prepare_benchmark

    for eval_dataset in tqdm(args.eval_datasets):

        set_seed(args.seed)
        print(f'evaluating on {eval_dataset}...')

        eval_dataset_path = os.path.join(args.datasets_path,eval_dataset) if args.datasets_path != "hf" else None
        eval_output_path = os.path.join(args.output_path,eval_dataset)
        os.makedirs(eval_output_path, exist_ok=True)
        
        try:
            benchmark = prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path)
            benchmark.load_data()
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Failed to prepare benchmark for {eval_dataset}")
            print(f"{'='*80}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            print(f"{'='*80}\n")
            continue
        
        try:
            final_results = benchmark.eval() if benchmark else {}
            print(f'final results on {eval_dataset}: {final_results}')
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Failed during evaluation of {eval_dataset}")
            print(f"{'='*80}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            print(f"{'='*80}\n")
            final_results = None
            
        if final_results is not None:
            if os.path.exists(total_results_path):
                with open (total_results_path,"r") as f:
                    total_results = json.load(f)
            else:
                total_results = {}
            total_results[eval_dataset] = final_results

        with open(os.path.join(args.output_path,'total_results.json'),'w') as f:
            json.dump(total_results,f,indent=4)
        gc.collect()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n{'='*80}")
        print("Program interrupted by user")
        print(f"{'='*80}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"FATAL ERROR: Program crashed")
        print(f"{'='*80}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        sys.exit(1)
