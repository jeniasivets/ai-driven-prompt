import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import t2v_metrics
from time import time
from tifa.tifascore import get_question_and_answers,\
                        filter_question_and_answers,\
                        UnifiedQAModel,\
                        tifa_score_single,\
                        VQAModel

import os
os.environ["OPENAI_API_KEY"] = ''

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result for a single image-prompt pair."""
    prompt: str
    image_path: str
    clip_score: float
    tifa_score: float
    generation_time: float
    combined_score: float
    individual_scores: Dict[str, float]


class EnhancedEvaluator:
    def __init__(self, 
                 device: str = "cuda",
                 clip_model: str = "openai:ViT-L-14-336",
                 qa_model: str = "allenai/unifiedqa-v2-t5-large-1363200",
                 vqa_model: str = "mplug-large",
                 debug: bool = False):
        """Initialize the enhanced evaluator with multiple validation strategies."""
        self.device = device
        self.debug = debug
        
        # Initialize different validators
        self.clip_validator = t2v_metrics.CLIPScore(model=clip_model, device=self.device)
        self.vqa_model = VQAModel(vqa_model)
        self.qa_model = UnifiedQAModel(qa_model)
        
        # Default weights for different metrics
        self.weights = {
            'clip': 0.5,    # Semantic alignment
            'tifa': 0.5     # Detailed scene understanding & Visual question answering
        }
        
    def evaluate_single(self, 
                       image_path: str, 
                       prompt: str,
                       weights: Optional[Dict[str, float]] = None) -> EvaluationResult:
        """Evaluate a single image-prompt pair using all metrics."""
        if weights is not None:
            self.weights = weights
            
        # Compute individual scores
        scores = {}
        inference_times = {}
        
        # CLIP Score
        t1 = time()
        with torch.no_grad(), torch.amp.autocast(self.device):
            clip_score = self.clip_validator(images=image_path, texts=prompt).item()
        inference_times['clip'] = time() - t1
        scores['clip'] = clip_score

        # TIFA Score (using existing implementation)
        t1 = time()
        tifa_score = self.compute_tifa_score(image_path, prompt)
        inference_times['tifa'] = time() - t1
        scores['tifa'] = tifa_score
        
        # Compute weighted combined score
        combined_score = sum(score * self.weights[metric] 
                           for metric, score in scores.items())
        
        if self.debug:
            print(f"Individual scores: {scores}")
            print(f"Inference times: {inference_times}")
            print(f"Combined score: {combined_score}")
        
        return EvaluationResult(
            prompt=prompt,
            image_path=image_path,
            clip_score=clip_score,
            tifa_score=tifa_score,
            generation_time=sum(inference_times.values()),
            combined_score=combined_score,
            individual_scores=scores
        )
    
    def evaluate_batch(self, 
                      image_paths: List[str], 
                      prompts: List[str],
                      weights: Optional[Dict[str, float]] = None) -> List[EvaluationResult]:
        """Evaluate multiple image-prompt pairs."""
        return [self.evaluate_single(img_path, prompt, weights) 
                for img_path, prompt in zip(image_paths, prompts)]
    
    def compute_tifa_score(self, image_path: str, prompt: str) -> float:
        """Compute TIFA score using the existing implementation."""
        # This uses existing TIFA score implementation

        # Generate questions with GPT-3.5-turbo
        gpt3_questions = get_question_and_answers(prompt)

        # Filter questions with UnifiedQA
        filtered_questions = filter_question_and_answers(self.qa_model, gpt3_questions)

        # See the questions
        print(filtered_questions)

        # calculate TIFA score
        result = tifa_score_single(self.vqa_model, filtered_questions, image_path)
        print(f"TIFA score is {result['tifa_score']}")
        print(result)

        return result['tifa_score']
    
    def analyze_results(self, results: List[EvaluationResult]) -> Dict:
        """Analyze evaluation results across multiple samples."""
        analysis = {
            'overall': {
                'average_combined_score': np.mean([r.combined_score for r in results]),
                'average_generation_time': np.mean([r.generation_time for r in results])
            },
            'by_metric': {}
        }
        
        # Analyze individual metrics
        for metric in self.weights.keys():
            scores = [getattr(r, f"{metric}_score") for r in results]
            analysis['by_metric'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return analysis
