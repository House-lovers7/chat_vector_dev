from pydantic import BaseModel

class InferenceConfig(BaseModel):
    # max_length: int = 128
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.99
    top_p: float = 0.95
    truncation: bool = True