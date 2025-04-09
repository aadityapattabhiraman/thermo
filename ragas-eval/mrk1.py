#!/home/akugyo/Programs/Python/ragas/bin/python

import os
import pandas as pd
import warnings
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic
from ragas.llms import LangchainLLMWrapper
from langchain_openai import AzureChatOpenAI



def get_df():

    # Read excel file
    df = pd.read_excel("thermo.xlsx", sheet_name=1)
    
    # Rename columns
    df.columns = df.iloc[2]
    df.columns.name = None

    # Reset index
    df = df[3:][:]
    df = df.reset_index(drop=True)

    return df



def process_df(df):

    print(df.iloc[0])
    useful_col = [
        "General Feedback",
        "Other Feedback",
        "Conversation",
    ]

    df = df[useful_col]
    print()
    print(df.iloc[0])

    return df.iloc[0]["Conversation"], df.iloc[146].Conversation


def evaluate(row):

    splits = row.split("\n")
    conversation = []
    for _ in splits:
        if "user:" in _.lower():
            conversation.append(_)
            continue

        elif "assistant:" in _.lower():
            conversation.append(_)
            continue

        conversation[-1] += "\n" + _
        
    test_data = {
        "user_input": conversation[0],
        "resonse": conversation[1]
    }

    metric = AspectCritic(
        name="response_quality",
        llm=evaluator_llm,
        definition="Evaluate the quality of response."
    )

    test_data = SingleTurnSample(**test_data)
    print(conversation[0])
    print(conversation[1])
    print("Response:", metric.single_turn_score(test_data))
    print()
    

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    evaluator_llm = LangchainLLMWrapper(model)

    df = get_df()
    row = process_df(df)
    for _ in row:
        evaluate(_)
        pass
