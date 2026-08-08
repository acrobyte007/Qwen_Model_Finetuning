import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "./qwen-finetuned"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    ),
        device_map="auto",
    )

    model.eval()

    return tokenizer, model


def generate_response(model, tokenizer, question):

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    start_time = time.perf_counter()

    try:

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=300,
                do_sample=False,
            )

        end_time = time.perf_counter()

        response = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        latency = end_time - start_time

        if not response:
            return {
                "success": False,
                "failure_type": "empty_output",
                "latency": latency,
                "response": "",
            }

        return {
            "success": True,
            "failure_type": None,
            "latency": latency,
            "response": response,
        }

    except torch.cuda.OutOfMemoryError:

        return {
            "success": False,
            "failure_type": "cuda_oom",
            "latency": None,
            "response": "",
        }

    except Exception as e:

        return {
            "success": False,
            "failure_type": type(e).__name__,
            "latency": None,
            "response": "",
        }


def benchmark(model, tokenizer, questions):

    results = []

    for i, question in enumerate(questions):

        result = generate_response(
            model,
            tokenizer,
            question,
        )

        results.append(result)

        print(
            f"{i + 1}/{len(questions)} | "
            f"success={result['success']} | "
            f"latency={result['latency']}"
        )

    return results


def calculate_metrics(results):

    total = len(results)

    successful = [
        r for r in results
        if r["success"]
    ]

    failed = [
        r for r in results
        if not r["success"]
    ]

    latencies = [
        r["latency"]
        for r in successful
        if r["latency"] is not None
    ]

    failure_rate = (
        len(failed) / total * 100
        if total > 0
        else 0
    )

    mean_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    return {
        "total": total,
        "successful": len(successful),
        "failed": len(failed),
        "failure_rate": failure_rate,
        "mean_latency": mean_latency,
    }


if __name__ == "__main__":

    tokenizer, model = load_model()

    questions = [
        "What are the symptoms of diabetes?",
        "What causes high blood pressure?",
        "What are the symptoms of asthma?",
        "What is anemia?",
        "What are common symptoms of migraine?",
    ]

    results = benchmark(
        model,
        tokenizer,
        questions,
    )

    metrics = calculate_metrics(results)

    print("\n=== Benchmark Results ===")

    print(f"Total requests: {metrics['total']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Failed: {metrics['failed']}")
    print(f"Failure rate: {metrics['failure_rate']:.2f}%")
    print(f"Mean latency: {metrics['mean_latency']:.3f} sec")