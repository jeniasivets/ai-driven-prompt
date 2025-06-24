import os
import sys
import json
import subprocess
import argparse


def get_sorted_image_files_and_prompts(directory):
    """Retrieve image file paths sorted by original prompt from metadata."""
    image_files = []
    prompts = []
    for prompt in os.listdir(directory):
        if not prompt.startswith('.'):
            prompt_path = os.path.join(directory, prompt)
            for filename in os.listdir(prompt_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')) and not filename.startswith('metadata'):
                    full_path = os.path.join(prompt_path, filename)
                    metadata_path = os.path.join(prompt_path, 'metadata', 'metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            original_prompt = metadata.get('original_prompt', '')
                            image_files.append((original_prompt, full_path))
                        prompts.append(metadata['generation_prompt'])
    return [file[1] for file in sorted(image_files, key=lambda x: x[0])], prompts


def ensure_repos():
    """Ensure required external repos are cloned and add to sys.path for dynamic import."""
    repos = [
        ("./t2v_metrics", "https://github.com/linzhiqiu/t2v_metrics"),
        ("./tifa", "https://github.com/Yushi-Hu/tifa.git"),
        ('.', '')
    ]
    for repo_dir, repo_url in repos:
        if not os.path.exists(repo_dir):
            print(f"Cloning {repo_url}...")
            subprocess.run(["git", "clone", repo_url])
        else:
            print(f"Repo {repo_dir} already exists.")
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.append(abs_repo_dir)


def main(config):
    ensure_repos()
    if 'OPENAI_API_KEY' not in os.environ:
        os.environ["OPENAI_API_KEY"] = config['api_key']
    from src.evaluator import EnhancedEvaluator
    device = config.get('device', 'cuda')
    evaluator = EnhancedEvaluator(device=device)
    base_path = config.get('output_dir', './results')
    difficulties = ['low', 'medium', 'high']
    prompt_types = ['orig_prompt', 'refined_prompt', 'attn_refined_prompt']
    all_results = {}
    for diff_level in difficulties:
        all_results[diff_level] = {}
        for prompt_type in prompt_types:
            dir_path = os.path.join(base_path, diff_level, prompt_type)
            img_paths, prompts = get_sorted_image_files_and_prompts(dir_path)
            print(f"Evaluating {len(img_paths)} images in {dir_path}...")
            results = evaluator.evaluate_batch(image_paths=img_paths, prompts=prompts)
            analysis = evaluator.analyze_results(results)
            all_results[diff_level][prompt_type] = analysis
            print(f"Summary for {diff_level}/{prompt_type}: {json.dumps(analysis, indent=2)}\n")
    print("\n=== Evaluation Complete ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='./config/pipeline_config.json',
                        help='Path to configuration file')
    parser.add_argument('--results-dir', type=str, help='Directory containing generated results')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps'],
                        help='Device to use for evaluation')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    args = parser.parse_args()

    # Load config from file
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file {args.config_file} not found.")

    # CLI overrides
    if args.results_dir:
        config['results_dir'] = args.results_dir
    if args.device:
        config['device'] = args.device
    if args.api_key:
        config['api_key'] = args.api_key

    if 'api_key' not in config or not config['api_key']:
        raise ValueError("api_key must be set in the config file or provided as --api_key.")

    main(config)
