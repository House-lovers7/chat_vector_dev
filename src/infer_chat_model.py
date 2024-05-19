from utils.common import build_pipeline, generate_text, read_prompts_from_folder
from transformers import AutoTokenizer
from config.settings import CHAT_MODEL_CONFIG, PROMPT_FOLDER

def main() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_CONFIG["tokenizer_name"])
        generator = build_pipeline(CHAT_MODEL_CONFIG["model_name"], tokenizer)
        
        prompt_list = read_prompts_from_folder(PROMPT_FOLDER)
        inst_prompt_list = [f"<s>[INST] {prompt} [/INST]" for prompt in prompt_list]
        generated_texts = generate_text(generator, inst_prompt_list, CHAT_MODEL_CONFIG["inference_config"])
        
        for text in generated_texts:
            print("-----" * 10)
            print(text)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()