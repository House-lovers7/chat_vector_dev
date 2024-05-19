
from utils.common import load_model
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch

def create_chat_model(base_model: PreTrainedModel, inst_model: PreTrainedModel, cp_model: PreTrainedModel) -> None:
    skip_layers = ["model.embed_tokens.weight", "lm_head.weight"]
    for key, value in cp_model.state_dict().items():
        if any(skip_layer in key for skip_layer in skip_layers) or "layernorm" in key:
            continue
        chat_vector = inst_model.state_dict()[key] - base_model.state_dict()[key]
        new_value = value + chat_vector.to(value.device)
        value.copy_(new_value)
    cp_model.save_pretrained("./data/models/chat_model")

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = load_model("mistralai/Mistral-7B-v0.1", device)
    inst_model = load_model("mistralai/Mistral-7B-Instruct-v0.2", device)
    cp_model = load_model("tokyotech-llm/Swallow-MS-7b-v0.1", device)
    create_chat_model(base_model, inst_model, cp_model)

if __name__ == "__main__":
    main()