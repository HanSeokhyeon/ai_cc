from fastapi import FastAPI, Body
from pydantic import BaseModel

from search_and_answer import search, generate_answer

app = FastAPI()


class RequestBody(BaseModel):
    question: str


@app.post("/answer")
async def answer(body: RequestBody = Body()):
    qa = search(body.question)
    return generate_answer(qa, body.question)
