import os
from src.prompt_refiner import PromptEnhancer, BatchGenerator
from diffusers import StableDiffusionXLPipeline
import json
import subprocess
import sys


if __name__ == "__main__":

    repo_dir = './prompt-to-prompt-with-sdxl'
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/RoyiRa/prompt-to-prompt-with-sdxl"])
    else:
        pass
    sys.path.append(repo_dir)
    from prompt_to_prompt_pipeline import Prompt2PromptPipeline

    model_path = './ckpt/sd_xl_base_1.0.safetensors'

    benchmark_file = './data/benchmark_prompts.json'
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)

    batch_generator = BatchGenerator(
        model_path=model_path,
        pipe=StableDiffusionXLPipeline,
        num_seeds=1
    )

    attn_batch_generator = BatchGenerator(
        model_path=model_path,
        pipe=Prompt2PromptPipeline,
        num_seeds=1
    )

    key = 'API_KEY'

    model_capabilities = """
        Stable Diffusion XL model capabilities:
        - Superior image quality and detail compared to previous models
        - Excellent handling of complex scenes and multiple subjects
        - Strong performance with natural lighting and composition
        - Advanced understanding of photographic concepts
        - Robust prompt following capabilities
        - High-quality animal and pet generation
        - Detailed fur and texture rendering
        - Professional photography style support
        - Multiple subject composition
        - Environmental detail preservation
        """

    enhancer = PromptEnhancer(
        openai_api_key=key,
        model_capabilities=model_capabilities,
    )

    for difficulty, prompts in benchmark_data.items():
        for prompt in prompts:

            # Generate orig prompt w/o enhancer
            batch_generator.generate_image(
                prompt,
                prompt_type='orig_prompt',
                difficulty=difficulty,
                cross_attention_kwargs=None
                )

            # LLM enhancer
            refined_prompt, _ = enhancer.enhance_prompt(
                original_prompt=prompt
            )

            batch_generator.generate_image(
                refined_prompt,
                prompt_type='refined_prompt',
                difficulty=difficulty,
                cross_attention_kwargs=None,
                original_prompt=prompt
            )

            # Attention enhancer
            analysis = enhancer.analyze_prompt(prompt)

            refined_prompt, cross_attention_kwargs = enhancer.enhance_prompt(
              original_prompt=prompt,
              key_words=analysis['key_words'],
              strengths=analysis['strengths']

            )

            attn_batch_generator.generate_image(
              refined_prompt,
              prompt_type='attn_refined_prompt',
              difficulty=difficulty,
              cross_attention_kwargs=cross_attention_kwargs,
              original_prompt=prompt
              )
