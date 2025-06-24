import openai
import json
from typing import List, Dict, Tuple, Optional, Literal
import torch
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline
from prompt_to_prompt_pipeline import Prompt2PromptPipeline
import os


class PromptRefinementStrategy:
    def __init__(self, openai_api_key: str, model_capabilities: str):
        """Initialize the prompt refinement strategy."""
        openai.api_key = openai_api_key
        self.model_capabilities = model_capabilities
        
    def refine_with_llm(self, original_prompt: str) -> str:
        """Refine a prompt using GPT-4 to better match model capabilities."""
        system_prompt = f"""You are an expert prompt engineer for Stable Diffusion XL image generation. 
        Your task is to refine prompts to better match the following model capabilities: {self.model_capabilities}

        Guidelines for refinement:
        1. Add SDXL-specific quality enhancers (e.g., "highly detailed", "masterpiece", "professional photography")
        2. Include specific lighting and composition details that SDXL excels at
        3. Specify important visual elements with precise descriptors
        4. Maintain the original intent while enhancing clarity
        5. Add style descriptors that work well with SDXL (e.g., "photorealistic", "cinematic lighting")
        6. Include camera and lens specifications for realistic results
        7. Add environmental details that SDXL can render well

        Return only the refined prompt, no explanations."""
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this prompt: {original_prompt}"}
            ]
        )
        return response.choices[0].message.content.strip()


class AttentionMapManipulator:
    def __init__(self, max_strength: float = 5.0):
        self.max_strength = max_strength

    def enhance_prompt(self,
                       prompt: str,
                       key_words: List[str],
                       strengths: List[float]) -> Tuple[List[str], Dict]:
        """Enhance a prompt using attention map manipulation."""
        max_strength_idx = strengths.index(max(strengths))
        strongest_word = [key_words[max_strength_idx]]
        strongest_strength = [self.max_strength]
        # Prepare cross attention parameters
        cross_attention_kwargs = {
            "edit_type": "reweight",
            "equalizer_words": strongest_word,
            "equalizer_strengths": strongest_strength
        }
        # For reweight type, we need to duplicate the prompt
        prompts = [prompt] * 2

        return prompts, cross_attention_kwargs


class BatchGenerator:
    def __init__(self,
                 pipe: StableDiffusionXLPipeline | Prompt2PromptPipeline,
                 model_path: str,
                 device: Literal["cuda", "mps"] = "cuda",
                 num_seeds: int = 1):
        """Initialize the batch generator."""
        self.num_seeds = num_seeds
        self.device = device
        self.pipe = pipe.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
            add_watermarker=False
        ).to(self.device)

    def __del__(self):
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()

    def generate_image(self,
                       prompt: str | List[str],
                       difficulty: str | None,
                       prompt_type: str = 'orig_prompt',
                       cross_attention_kwargs: Optional[Dict] = None,
                       num_inference_steps: int = 20,
                       output_dir: str = './results',
                       original_prompt: Optional[str] = None,
                       ) -> List[torch.Tensor]:
        """Generate multiple images with different seeds."""
        images = []
        for seed in range(self.num_seeds):
            generator = torch.Generator().manual_seed(seed)
            if cross_attention_kwargs:
                result = self.pipe(
                    prompt,
                    cross_attention_kwargs=cross_attention_kwargs,
                    generator=generator,
                    num_inference_steps=num_inference_steps
                )
            else:
                result = self.pipe(
                    prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps
                )
            image = result.images[0]

            dir_prompt = original_prompt if original_prompt else prompt

            # Save image
            image_path = os.path.join(output_dir, difficulty, prompt_type, f"{dir_prompt[:30]}")

            if not os.path.exists(image_path):
                os.makedirs(image_path)
            else:
                counter = 1
                new_image_path = image_path + str(counter)
                while os.path.exists(new_image_path):
                    counter += 1
                    new_image_path = image_path + str(counter)
                image_path = new_image_path
                os.makedirs(image_path)

            image.save(os.path.join(image_path, f"img_{seed}.png"))
            images.append(image)

            # Save metadata
            metadata_path = os.path.join(image_path, "metadata")
            if not os.path.exists(metadata_path):
                os.makedirs(metadata_path)
            meta_prompt = prompt[0] if cross_attention_kwargs else prompt
            metadata = {
                "original_prompt": original_prompt if original_prompt else meta_prompt,
                "generation_prompt": meta_prompt,
                "difficulty": difficulty,
                "prompt_type": prompt_type,
                "cross_attention_kwargs": cross_attention_kwargs if cross_attention_kwargs else None,
                "seed": seed,
                "inference_steps": num_inference_steps,
            }
            json_file_path = os.path.join(metadata_path, 'metadata.json')
            with open(json_file_path, 'w') as json_file:
                json.dump(metadata, json_file, indent=4)

        return images

    def generate_batch(self,
                       prompts: List[str] | List[List[str]],
                       difficulty: str | None,
                       prompt_type: str | None,
                       cross_attention_kwargs: Optional[Dict] = None,
                       num_inference_steps: int = 20) -> List[torch.Tensor]:
        images = []
        for p in prompts:
            curr_image = self.generate_image(p, difficulty, prompt_type, cross_attention_kwargs, num_inference_steps)
            images.append(curr_image)
        return images


class PromptEnhancer:
    def __init__(self,
                 openai_api_key: str,
                 model_capabilities: str,
                 ):
        """Initialize the prompt enhancer with all strategies."""
        self.refinement_strategy = PromptRefinementStrategy(openai_api_key, model_capabilities)
        self.attention_manipulator = AttentionMapManipulator()

    def enhance_prompt(self, 
                      original_prompt: str,
                      key_words: Optional[List[str]] = None,
                      strengths: Optional[List[float]] = None) -> Tuple[str, Dict | None]:
        """Enhance a prompt using all available strategies."""
        
        # Apply attention map manipulation if keywords are provided
        if key_words and strengths:
            prompts, cross_attention_kwargs = self.attention_manipulator.enhance_prompt(
                original_prompt,
                key_words,
                strengths
            )
        # Refine prompt using LLM
        else:
            refined_prompt = self.refinement_strategy.refine_with_llm(original_prompt)
            prompts = refined_prompt
            cross_attention_kwargs = None

        return prompts, cross_attention_kwargs

    def analyze_prompt(self, prompt: str, max_retries: int = 3) -> Dict:
        """Analyze a prompt to suggest key words for attention manipulation."""
        for attempt in range(max_retries):
            try:
                client = OpenAI(api_key=openai.api_key)
                system_prompt = """You are a prompt analysis expert. Analyze the given prompt and identify:
                        1. Key action words that should be emphasized
                        2. Important visual elements
                        3. Suggested attention strengths for each element

                        Return a JSON with the following structure:
                        {
                            "key_words": ["word1", "word2", ...],
                            "strengths": [float1, float2, ...],
                            "analysis": "brief explanation"
                        }"""
                response = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this prompt: {prompt}"}
                    ]
                )
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                print(f"Error analyzing prompt: {prompt}. Attempt {attempt + 1} of {max_retries}. Error: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to analyze prompt after {max_retries} attempts.")
