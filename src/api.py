from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.query import search, generate_answer

app = FastAPI(title="Employee Assistant API")


# this defines what a valid request looks like - fastapi uses it to validate automatically
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to ask about employees")


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        matches = search(request.question)

        if not matches:
            raise HTTPException(status_code=404, detail="No relevant employee records found")

        answer = generate_answer(request.question, matches)

        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=matches
        )

    except HTTPException:
        # let our own 404 above pass through as-is
        raise
    except Exception as e:
        # anything unexpected such asollama down chromadb issue becomes a clean 500 instead of a crash
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")


@app.get("/")
def root():
    return {"message": "Employee Assistant API is running. POST to /ask with a question."}