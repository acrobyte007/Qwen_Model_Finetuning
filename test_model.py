import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "./qwen-finetuned"


if __name__ == "__main__":

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32),
        device_map="auto",
    )

    model.eval()

    print("Model loaded successfully!")

    question = "What are the symptoms of diabetes?"

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    print("\nGenerating answer...\n")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(response)