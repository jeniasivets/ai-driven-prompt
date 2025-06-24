import os
import sys
import json
import argparse
import subprocess
from diffusers import StableDiffusionXLPipeline


def ensure_repos():
    """Ensure required external repos are cloned and add to sys.path for dynamic import."""
    repos = [
        ("./prompt-to-prompt-with-sdxl", "https://github.com/RoyiRa/prompt-to-prompt-with-sdxl"),
        ('.', '')
    ]
    for repo_dir, repo_url in repos:
        if repo_url and not os.path.exists(repo_dir):
            print(f"Cloning {repo_url}...")
            subprocess.run(["git", "clone", repo_url])
        else:
            if repo_url:
                print(f"Repo {repo_dir} already exists.")
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.append(abs_repo_dir)


def setup_pipelines(config):
    from prompt_to_prompt_pipeline import Prompt2PromptPipeline
    from src.prompt_refiner import BatchGenerator
    model_path = config['model_path']
    device = config.get('device', 'cuda')
    batch_generator = attn_batch_generator = None
    if config['enable_original_prompt'] or config['enable_llm_refined_prompt']:
        batch_generator = BatchGenerator(model_path=model_path, pipe=StableDiffusionXLPipeline,
                                         device=device, num_seeds=1)
    if config['enable_attn_refined_prompt']:
        attn_batch_generator = BatchGenerator(model_path=model_path, pipe=Prompt2PromptPipeline,
                                              device=device, num_seeds=1)
    return batch_generator, attn_batch_generator


def generate_original_prompt(batch_generator, prompt, difficulty, config):
    if not config['enable_original_prompt'] or batch_generator is None:
        return
    batch_generator.generate_image(prompt,
                                   prompt_type='orig_prompt',
                                   difficulty=difficulty,
                                   cross_attention_kwargs=None,
                                   original_prompt=prompt,
                                   output_dir=config['output_dir'])


def generate_llm_refined_prompt(batch_generator, enhancer, prompt, difficulty, config):
    if not config['enable_llm_refined_prompt'] or batch_generator is None:
        return
    refined_prompt, _ = enhancer.enhance_prompt(original_prompt=prompt)
    batch_generator.generate_image(refined_prompt,
                                   prompt_type='refined_prompt',
                                   difficulty=difficulty,
                                   cross_attention_kwargs=None,
                                   original_prompt=prompt,
                                   output_dir=config['output_dir'])


def generate_attn_refined_prompt(attn_batch_generator, enhancer, prompt, difficulty, config):
    if not config['enable_attn_refined_prompt'] or attn_batch_generator is None:
        return
    analysis = enhancer.analyze_prompt(prompt)
    refined_prompt, cross_attention_kwargs = enhancer.enhance_prompt(original_prompt=prompt,
                                                                     key_words=analysis['key_words'],
                                                                     strengths=analysis['strengths'])
    attn_batch_generator.generate_image(refined_prompt,
                                        prompt_type='attn_refined_prompt',
                                        difficulty=difficulty,
                                        cross_attention_kwargs=cross_attention_kwargs,
                                        original_prompt=prompt,
                                        output_dir=config['output_dir'])


def process_single_prompt(batch_generator, attn_batch_generator, enhancer, prompt, difficulty, config):
    generate_original_prompt(batch_generator, prompt, difficulty, config)
    generate_llm_refined_prompt(batch_generator, enhancer, prompt, difficulty, config)
    generate_attn_refined_prompt(attn_batch_generator, enhancer, prompt, difficulty, config)


def get_model_capabilities():
    """Return model capabilities description for LLM prompt refinement."""
    return """
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


def main(config):
    ensure_repos()
    batch_generator, attn_batch_generator = setup_pipelines(config)
    with open('./data/benchmark_prompts.json', 'r') as f:
        benchmark_data = json.load(f)
    from src.prompt_refiner import PromptEnhancer
    enhancer = PromptEnhancer(openai_api_key=config['api_key'], model_capabilities=get_model_capabilities())

    for difficulty, prompts in benchmark_data.items():
        for prompt in prompts:
            process_single_prompt(batch_generator, attn_batch_generator, enhancer, prompt, difficulty, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='./config/pipeline_config.json',
                        help='Path to configuration file')
    parser.add_argument('--model-path', type=str,
                        help='Path to the model safetensors file')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for generated images')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps'],
                        help='Device to use for generation')
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument('--disable-original', action='store_true',
                        help='Disable original prompt generation')
    parser.add_argument('--disable-llm-refined', action='store_true',
                        help='Disable llm_refined prompt generation')
    parser.add_argument('--disable-attn-refined', action='store_true',
                        help='Disable attn_refined prompt generation')
    args = parser.parse_args()

    # Load config from file
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file {args.config_file} not found.")

    # CLI overrides
    if args.model_path:
        config['model_path'] = args.model_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.device:
        config['device'] = args.device
    if args.api_key:
        config['api_key'] = args.api_key
    if args.disable_original:
        config['enable_original_prompt'] = False
    if args.disable_llm_refined:
        config['enable_llm_refined_prompt'] = False
    if args.disable_attn_refined:
        config['enable_attn_refined_prompt'] = False

    if 'api_key' not in config or not config['api_key']:
        raise ValueError("api_key must be set in the config file or provided as --api-key.")

    main(config)
