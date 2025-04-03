#!/usr/bin/env python3

from ragas.experimental.prompt import PydanticPrompt
from pydantic import BaseModel, Field



class MyInput(BaseModel):

    question: str = Field(description="The question to answer")


class MyOutput(BaseModel):

    answer: str = Field(description="The answer to the question")


class MyPrompt(PydanticPrompt[MyInput,MyInput]):

    instruction = "Answer the given question"
    input_model = MyInput
    output_model = MyOutput
    examples = [
        (
            MyInput(question="Who's building the opensource standard for LLM app evals?"),
            MyOutput(answer="Ragas")
        )
    ]
