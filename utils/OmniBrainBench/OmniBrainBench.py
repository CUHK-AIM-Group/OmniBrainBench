import torch
import os
import json
import gc
import logging
import numpy as np

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

# Evaluation metrics related imports
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from ..utils import save_json,extract,judge_multi_choice,judge_open_end_vqa,get_compare_messages,judger
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt, get_open_ended_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class OmniBrainBench(BaseDataset):
    def __init__(self,model,dataset_path,output_path, openset=False):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.openset = openset
        
        # Initialize evaluators (only in openset mode)
        self.evaluators = {}
        if self.openset:
            self._initialize_evaluators()
    
    def _initialize_evaluators(self):
        """Initialize various evaluators"""
        # ROUGE evaluator
        try:
            self.evaluators['rouge'] = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            logger.info("ROUGE scorer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ROUGE: {e}")
        
        # BERTScore evaluator
        try:
            self.evaluators['bertscore'] = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en', rescale_with_baseline=True)
            logger.info("BERTScore scorer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize BERTScore: {e}")
    
    def _compute_rouge_metrics(self, reference, hypothesis):
        """Calculate ROUGE1 and ROUGEL metrics"""
        if 'rouge' not in self.evaluators:
            return {'rouge1': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.evaluators['rouge'].score(reference, hypothesis)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            return {'rouge1': 0.0, 'rougeL': 0.0}
    
    def _compute_bleu_metrics(self, reference, hypothesis):
        """Calculate BLEU metrics - improved version"""
        try:
            ref_tokens = [reference.split()]
            hyp_tokens = hypothesis.split()
            
            if len(hyp_tokens) == 0 or len(ref_tokens[0]) == 0:
                return {'bleu': 0.0}
            
            # Choose appropriate weights based on text length
            if len(hyp_tokens) <= 2:
                weights = (1.0, 0, 0, 0)  # Use only 1-gram
            elif len(hyp_tokens) <= 4:
                weights = (0.5, 0.5, 0, 0)  # Use 1-gram and 2-gram
            else:
                weights = (0.25, 0.25, 0.25, 0.25)  # Standard 4-gram
            
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu(ref_tokens, hyp_tokens, 
                                      weights=weights, 
                                      smoothing_function=smoothie)
            
            return {'bleu': min(bleu_score, 1.0)}  # Ensure not exceeding 1.0
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return {'bleu': 0.0}
    
    def _compute_bertscore_metrics(self, reference, hypothesis):
        """Calculate BERTScore metrics"""
        if 'bertscore' not in self.evaluators:
            return {'bertscore': 0.0}
        
        try:
            P, R, F1 = self.evaluators['bertscore'].score([hypothesis], [reference])
            return {'bertscore': F1.item()}  # Use F1 score
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return {'bertscore': 0.0}
    
    def _compute_exact_match(self, reference, hypothesis):
        """Calculate exact match"""
        return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0
    
    def _comprehensive_text_evaluation(self, reference, hypothesis):
        """Comprehensive text evaluation - 4 metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['exact_match'] = self._compute_exact_match(reference, hypothesis)
        
        # 4 required evaluation metrics
        # 1-2. ROUGE metrics
        rouge_metrics = self._compute_rouge_metrics(reference, hypothesis)
        metrics.update(rouge_metrics)
        
        # 3. BLEU metrics
        bleu_metrics = self._compute_bleu_metrics(reference, hypothesis)
        metrics.update(bleu_metrics)
        
        # 4. BERTScore metrics
        bertscore_metrics = self._compute_bertscore_metrics(reference, hypothesis)
        metrics.update(bertscore_metrics)
        
        return metrics
    
    def _compute_statistics(self, scores):
        """Calculate statistics: max, min, mean, standard deviation"""
        if not scores:
            return {
                'max': 0.0,
                'min': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'count': 0
            }
        
        scores_array = np.array(scores)
        return {
            'max': float(np.max(scores_array)),
            'min': float(np.min(scores_array)),
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'count': len(scores)
        }
    
    def resize_image_if_needed(self, img, max_size=1280):
        """
        Resize image if width or height exceeds max_size, maintaining aspect ratio
        
        Args:
            img: PIL Image object
            max_size: Maximum size limit, default 1280

        Returns:
            Resized PIL Image object
        """
        width, height = img.size
        
        # Check if resize is needed
        if width <= max_size and height <= max_size:
            return img
        
        # Calculate scale ratio, maintaining aspect ratio
        if width > height:
            scale = max_size / width
        else:
            scale = max_size / height
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use high quality resize method
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        print(f"[INFO] Resized image from ({width}, {height}) to ({new_width}, {new_height})")
        
        return resized_img
        
    def run(self, samples, model, batch_size=2000):
        """Override BaseDataset's run method, add image resize functionality, process one by one to avoid image accumulation"""
        out_samples = []
        print(f"[INFO] Running model on {len(samples)} samples")
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(samples, desc="Processing")):
                messages = sample["messages"]
                
                # 处理单个样本的图像
                if "images" in messages:
                    image_list = []
                    image_paths = messages["images"]
                    
                    # Ensure image_paths is a list
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    
                    for img_path in image_paths:
                        try:
                            # Ensure img_path is a string
                            if img_path is None:
                                print(f"[WARNING] Skipping None image path in sample {idx}")
                                continue
                            
                            img_path = str(img_path).strip()
                            if not img_path:
                                print(f"[WARNING] Skipping empty image path in sample {idx}")
                                continue
                            
                            with Image.open(img_path) as img:
                                resized_img = self.resize_image_if_needed(img)
                                image_list.append(resized_img.copy())
                        except Exception as e:
                            print(f"[WARNING] Error loading image {img_path}: {e}")
                            # Don't append None, just skip the image
                    
                    # Only add to messages if we have valid images
                    if image_list:
                        processed_message = {"prompt": messages["prompt"], "images": image_list}
                    else:
                        # If no valid images, use text-only message
                        print(f"[WARNING] No valid images for sample {idx}, using text-only mode")
                        processed_message = {"prompt": messages["prompt"]}
                else:
                    processed_message = messages
                
                try:
                    # 单个样本处理
                    outputs = model.generate_outputs([processed_message])
                    response = outputs[0] if outputs else "[ERROR] No output"
                    
                    if "messages" in sample:
                        del sample["messages"]
                    sample["response"] = response
                    out_samples.append(sample)
                    
                except Exception as e:
                    print(f"[ERROR] Error processing sample {idx}: {e}")
                    if "messages" in sample:
                        del sample["messages"]
                    sample["response"] = f"[ERROR] Generation failed: {str(e)}"
                    out_samples.append(sample)
        
        return out_samples

    def load_data(self):
        dataset_path = self.dataset_path
        datasets = []

        if self.openset:
            json_path = os.path.join(dataset_path, "open-ended-qa_2704.json")
            image_path = os.path.join(dataset_path,"open-ended-qa_2704")
            print(json_path)
        else:
            json_path = os.path.join(dataset_path, "closed-ended-qa_6823.json")
            image_path = os.path.join(dataset_path,"closed-ended-qa_6823")
        with open(json_path,"r") as f:
            datas = json.load(f)
            
        for data in datas:
            # if data['image_path'] is List:
            if isinstance(data['image_path'], list):
                image_paths = []
                for img_p in data['image_path']:
                    image_paths.append(os.path.join(image_path, img_p))
                data['image_path'] = image_paths
            else:
                data['image_path'] = [os.path.join(image_path, data['image_path'])]
            datasets.append(data)
            
        for idx,sample in tqdm(enumerate(datasets)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples


    def construct_messages(self,sample):

        question = sample["question"]
        answer = sample["answer"]

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        if self.openset:
            prompt = get_open_ended_prompt(question,is_reasoning)
        else:
            choices = []
            opt_title = ['A','B','C','D','E']
            labels = sample['options']
            assert len(labels) == len(opt_title), f"options length {len(labels)} not equal to 5"
    
            
            for opt,lbl in zip(opt_title, labels):
                choice = f"{opt}. {lbl}"
                choices.append(choice)
                    
            prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
        
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt)
        
        if "image_path" in sample:
            image_paths = sample["image_path"]
            
            # Ensure image_paths is a list
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            elif not isinstance(image_paths, list):
                image_paths = [image_paths]
            
            # Filter out None values from image paths
            image_paths = [p for p in image_paths if p is not None]
            
            # 这里只存储路径，不立即加载图像
            if image_paths:
                messages = {"prompt":prompt,"images":image_paths}
            else:
                messages = {"prompt":prompt}
            
            del sample["image_path"]
        else:
            messages = {"prompt":prompt}
        
        sample["messages"] = messages
        if not self.openset:
            sample["choices"] = choices
        sample["answer"] = answer
        
        # 清理不需要的键
        if "options" in sample:
            del sample["options"]
        if "source_file" in sample:
            del sample["source_file"]
        return sample


    def cal_metrics(self,out_samples):
        total_task_type = defaultdict(int)
        right_task_type = defaultdict(int)
        total_clinical_phase_type = defaultdict(int)
        right_clinical_phase_type = defaultdict(int)
        
        # Differentiate between multiple choice and open-ended questions
        if self.openset:
            # Use format consistent with recompute_open_metrics.py
            logger.info(f"Computing metrics for {len(out_samples)} samples with 4 evaluation metrics")
            
            # Initialize metrics structure (consistent with recompute_open_metrics.py)
            metrics = {
                "total_samples": len(out_samples),
                "accuracy": 0.0,
                "metrics_summary": {
                    "rouge1": {},
                    "rougeL": {},
                    "bleu": {},
                    "bertscore": {}
                },
                "five_metrics": {
                    "rouge1": 0.0,
                    "rougeL": 0.0,
                    "bleu": 0.0,
                    "bertscore": 0.0
                },
                "sample_details": [],
                # Keep original task type and clinical phase statistics
                "total metrics": {
                    "total": 0,
                    "right": 0
                }
            }
            
            # Collect scores from all samples
            rouge1_scores = []
            rougeL_scores = []
            bleu_scores = []
            bertscore_scores = []
            exact_match_scores = []
            
            # Open-ended question evaluation
            messages_list = []
            open_id = []
            
            for i, sample in enumerate(tqdm(out_samples, desc="Computing 4 metrics")):
                response = sample["response"]
                response = extract(response, "answer")
                answer = sample["answer"]
                question = sample["question"]
                task_type = sample["task_label"]
                clinical_phase_type = sample["clinical_phase"]
                
                total_task_type[task_type] += 1
                total_clinical_phase_type[clinical_phase_type] += 1
                metrics["total metrics"]["total"] += 1
                
                # Use new 4-metric evaluation
                text_metrics = self._comprehensive_text_evaluation(answer, response)
                
                # Save sample-level metrics (consistent with recompute_open_metrics.py)
                sample_detail = {
                    "id": sample.get("id", i),
                    "response": response,
                    "answer": answer,
                    "metrics": text_metrics
                }
                metrics["sample_details"].append(sample_detail)
                
                # Save to original sample (backward compatible)
                out_samples[i]["correct"] = text_metrics["exact_match"] == 1.0
                out_samples[i]["metrics"] = text_metrics
                
                # Collect metric scores
                if 'rouge1' in text_metrics:
                    rouge1_scores.append(text_metrics['rouge1'])
                if 'rougeL' in text_metrics:
                    rougeL_scores.append(text_metrics['rougeL'])
                if 'bleu' in text_metrics:
                    bleu_scores.append(text_metrics['bleu'])
                if 'bertscore' in text_metrics:
                    bertscore_scores.append(text_metrics['bertscore'])
                if 'exact_match' in text_metrics:
                    exact_match_scores.append(text_metrics['exact_match'])
                
                # Calculate accuracy based on EM metric (backward compatible)
                if text_metrics["exact_match"] == 1.0:
                    metrics["total metrics"]["right"] += 1
                    right_task_type[task_type] += 1
                    right_clinical_phase_type[clinical_phase_type] += 1
                
                # If using LLM judge, prepare message list
                if os.environ.get("use_llm_judge", "False") == "True":
                    messages = get_compare_messages(question, response, answer)
                    messages_list.append(messages)
                    open_id.append(i)
            
            # Use LLM judge (if enabled)
            if os.environ.get("use_llm_judge", "False") == "True":
                logger.info("Using LLM judge for evaluation")
                metrics["total metrics"]["right"] = 0
                # Reset right counts for task_type and clinical_phase_type
                right_task_type = defaultdict(int)
                right_clinical_phase_type = defaultdict(int)
                
                llm = judger
                results = llm.generate_outputs(messages_list)
                for i, result in zip(open_id, results):
                    result = extract(result, "judge")
                    result = True if result == "0" else False
                    out_samples[i]["correct"] = result
                    out_samples[i]["llm_judge_result"] = result
                    
                    if result:
                        metrics["total metrics"]["right"] += 1
                        task_type = out_samples[i]["task_label"]
                        clinical_phase_type = out_samples[i]["clinical_phase"]
                        right_task_type[task_type] += 1
                        right_clinical_phase_type[clinical_phase_type] += 1
            
            # Calculate overall accuracy (consistent with recompute_open_metrics.py)
            metrics["accuracy"] = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0
            metrics["total metrics"]["acc"] = metrics["accuracy"]
            
            # Calculate statistics for each metric (consistent with recompute_open_metrics.py)
            metrics["metrics_summary"]["rouge1"] = self._compute_statistics(rouge1_scores)
            metrics["metrics_summary"]["rougeL"] = self._compute_statistics(rougeL_scores)
            metrics["metrics_summary"]["bleu"] = self._compute_statistics(bleu_scores)
            metrics["metrics_summary"]["bertscore"] = self._compute_statistics(bertscore_scores)
            
            # Calculate mean values for five_metrics (consistent with recompute_open_metrics.py)
            metrics["five_metrics"]["rouge1"] = metrics["metrics_summary"]["rouge1"]["mean"]
            metrics["five_metrics"]["rougeL"] = metrics["metrics_summary"]["rougeL"]["mean"]
            metrics["five_metrics"]["bleu"] = metrics["metrics_summary"]["bleu"]["mean"]
            metrics["five_metrics"]["bertscore"] = metrics["metrics_summary"]["bertscore"]["mean"]
            
            # New: Calculate comprehensive average of all metrics (excluding gpt_judge)
            valid_metrics = [v for k, v in metrics["five_metrics"].items() if k != "gpt_judge"]
            metrics["five_metrics"]["average_metric"] = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0.0
            
            # Print summary
            logger.info(f"Total Accuracy: {metrics['accuracy']:.4f}")
            logger.info("4 Evaluation Metrics Mean Values:")
            for metric_name, metric_value in metrics["five_metrics"].items():
                logger.info(f"  - {metric_name}: {metric_value:.4f}")
            
            logger.info("Detailed Statistics:")
            for metric_name, stats in metrics["metrics_summary"].items():
                logger.info(f"  - {metric_name}:")
                logger.info(f"      Max: {stats['max']:.4f}")
                logger.info(f"      Min: {stats['min']:.4f}")
                logger.info(f"      Mean: {stats['mean']:.4f}")
                logger.info(f"      Std: {stats['std']:.4f}")
                logger.info(f"      Count: {stats['count']}")
        
        else:
            # Multiple choice evaluation (original logic)
            metrics = {
                "total metrics": {
                    "total": 0,
                    "right": 0
                }
            }
            
            for i, sample in enumerate(out_samples):
                response = sample["response"]
                response = extract(response, "answer")
                
                choices = sample["choices"]
                answer = sample["answer"]
                task_type = sample["task_label"]
                clinical_phase_type = sample["clinical_phase"]
                total_task_type[task_type] += 1
                total_clinical_phase_type[clinical_phase_type] += 1
                metrics["total metrics"]["total"] += 1

                correct = judge_multi_choice(choices, answer, response)
                out_samples[i]["correct"] = correct
                if correct:
                    right_task_type[task_type] += 1
                    right_clinical_phase_type[clinical_phase_type] += 1
                    metrics["total metrics"]["right"] += 1
            
            metrics["total metrics"]["acc"] = metrics["total metrics"]["right"] / metrics["total metrics"]["total"] if metrics["total metrics"]["total"] > 0 else 0
        
        # Calculate task type and clinical phase metrics (needed for both modes)
        task_type_metrics = {}
        for key, value in total_task_type.items():
            right_cnt = right_task_type[key]
            task_type_metrics[key] = {"total": value, "right": right_cnt, "acc": right_cnt / value if value > 0 else 0}
        
        clinical_phase_type_metrics = {}
        for key, value in total_clinical_phase_type.items():
            right_cnt = right_clinical_phase_type[key]
            clinical_phase_type_metrics[key] = {"total": value, "right": right_cnt, "acc": right_cnt / value if value > 0 else 0}
        
        metrics["task type metrics"] = task_type_metrics
        metrics["clinical phase type metrics"] = clinical_phase_type_metrics
        
        return metrics, out_samples
    
    def eval(self):
        """Override parent eval method, save additional open_metrics.json in openset mode"""
        model = self.model
        dataset_path = self.dataset_path
        output_path = self.output_path
        num_chunks = self.num_chunks
        chunk_idx = self.chunk_idx
        
        if num_chunks == 1:
            results_path = os.path.join(output_path, "results.json")
            metric_path = os.path.join(output_path, "metrics.json")
            
            out_samples = self.run(self.samples, model)
            save_json(results_path, out_samples)

            metrics, out_samples = self.cal_metrics(out_samples)
            save_json(metric_path, metrics)
            save_json(results_path, out_samples)
            
            # If in openset mode, additionally save open_metrics.json (consistent with recompute_open_metrics.py format)
            if self.openset:
                open_metrics_path = os.path.join(output_path, "open_metrics.json")
                # Extract format consistent with recompute_open_metrics.py
                open_metrics = {
                    "total_samples": metrics.get("total_samples", 0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "metrics_summary": metrics.get("metrics_summary", {}),
                    "five_metrics": metrics.get("five_metrics", {}),
                    "sample_details": metrics.get("sample_details", [])
                }
                save_json(open_metrics_path, open_metrics)
                logger.info(f"Saved open_metrics.json to {open_metrics_path}")
            
            return metrics

        elif num_chunks > 1:
            results_path = os.path.join(output_path, f"results_{chunk_idx}.json")
            final_results_path = os.path.join(output_path, "results.json")
            out_samples = self.run(self.samples, model)
            save_json(results_path, out_samples)

            total_results_path = os.listdir(output_path)
            total_results_path = [result for result in total_results_path if result.startswith("results_")]
            
            if len(total_results_path) == num_chunks:
                total_results = []
                for result in total_results_path:
                    results_path = os.path.join(output_path, result)
                    with open(results_path, "r") as f:
                        total_results.extend(json.load(f))

                save_json(final_results_path, total_results)
                metrics, out_samples = self.cal_metrics(total_results)
                metric_path = os.path.join(output_path, "metrics.json")
                save_json(metric_path, metrics)
                save_json(final_results_path, out_samples)
                
                # If in openset mode, additionally save open_metrics.json (consistent with recompute_open_metrics.py format)
                if self.openset:
                    open_metrics_path = os.path.join(output_path, "open_metrics.json")
                    # Extract format consistent with recompute_open_metrics.py
                    open_metrics = {
                        "total_samples": metrics.get("total_samples", 0),
                        "accuracy": metrics.get("accuracy", 0.0),
                        "metrics_summary": metrics.get("metrics_summary", {}),
                        "five_metrics": metrics.get("five_metrics", {}),
                        "sample_details": metrics.get("sample_details", [])
                    }
                    save_json(open_metrics_path, open_metrics)
                    logger.info(f"Saved open_metrics.json to {open_metrics_path}")
                
                return metrics
            else:
                return None
        else:
            raise ValueError("num_chunks must be greater than 0")
