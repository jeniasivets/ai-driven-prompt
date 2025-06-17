from src.evaluator import EnhancedEvaluator
import os
import json
import subprocess
import sys


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


if __name__ == "__main__":
    repo_dir = './t2v_metrics'
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/linzhiqiu/t2v_metrics"])
    else:
        pass
    sys.path.append(repo_dir)
    evaluator = EnhancedEvaluator()

    base_path = './results'
    paths = {}
    for diff_level in ['low', 'medium', 'high']:
        paths[diff_level] = [
            os.path.join(base_path, diff_level, 'orig_prompt'),
            os.path.join(base_path, diff_level, 'refined_prompt'),
            os.path.join(base_path, diff_level, 'attn_refined_prompt'),
        ]

    for k, v in paths.items():
        for i in range(len(paths[k])):
            img_paths, prompts = get_sorted_image_files_and_prompts(paths[k][i])
            result = evaluator.evaluate_batch(image_paths=img_paths,
                                              prompts=prompts)
            evaluator.analyze_results(result)
