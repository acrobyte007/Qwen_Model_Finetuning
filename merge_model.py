import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./adapter"
OUTPUT_PATH = "./qwen-finetuned"


def merge_adapter():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(OUTPUT_PATH)


if __name__ == "__main__":
    merge_adapter()