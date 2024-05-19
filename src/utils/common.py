import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, PreTrainedModel, PreTrainedTokenizer
import torch
from typing import List, Optional
from pydantic import BaseModel

class InferenceConfig(BaseModel):
    max_new_tokens: int = 128
    num_beams: int = 1
    temperature: float = 1.0
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0

def load_model(model_name: str, device: str) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

def build_pipeline(model_name: str, tokenizer: PreTrainedTokenizer) -> TextGenerationPipeline:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return TextGenerationPipeline(model=model, tokenizer=tokenizer)

def generate_text(generator: TextGenerationPipeline, prompt_list: List[str], config: InferenceConfig) -> List[str]:
    generator_params = config.dict()
    generator_params["pad_token_id"] = generator.tokenizer.eos_token_id
    
    generated_texts = []
    for prompt in prompt_list:
        output = generator(prompt, **generator_params)
        generated_texts.append(output[0]["generated_text"])
    
    return generated_texts

def read_prompts_from_folder(folder_path: str) -> List[str]:
    prompt_list = []
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    prompt = file.read().strip()
                    prompt_list.append(prompt)
    except Exception as e:
        print(f"Error reading prompts from folder: {e}")
        raise
    return prompt_list