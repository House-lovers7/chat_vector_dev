from utils.common import InferenceConfig

CHAT_MODEL_CONFIG = {
    "model_name": "./data/models/chat_model",
    "tokenizer_name": "tokyotech-llm/Swallow-MS-7b-v0.1",
    "inference_config": InferenceConfig(max_new_tokens=256),
}

MISTRAL_INSTRUCT_CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "inference_config": InferenceConfig(),
}

SWALLOW_MS_CONFIG = {
    "model_name": "tokyotech-llm/Swallow-MS-7b-v0.1",
    "inference_config": InferenceConfig(),
}

PROMPT_FOLDER = "./data/prompt_list"