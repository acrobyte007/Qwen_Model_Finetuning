from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import model, generate_answer


app = FastAPI(
    title="MedQuAD Qwen API",
    description="Qwen2.5-1.5B fine-tuned on MedQuAD",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    question: str
    max_new_tokens: int = 300


class GenerateResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model": "Qwen2.5-1.5B fine-tuned on MedQuAD",
        "device": str(model.device),
    }


@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(request: GenerateRequest):

    try:

        answer = generate_answer(
            question=request.question,
            max_new_tokens=request.max_new_tokens,
        )

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Model returned an empty response",
            )

        return GenerateResponse(
            question=request.question,
            answer=answer,
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )