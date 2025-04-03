#!/home/akugyo/Programs/Python/ragas/bin/python3

import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas import evaluate



model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

evaluator_llm = LangchainLLMWrapper(model)

embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="VAF_TEXT_EMBEDDER",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

evaluator_embedding = LangchainEmbeddingsWrapper(embedding)

eval_dataset = load_dataset("explodinggradients/earning_report_summary",split="train")
eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)

metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
results = evaluate(eval_dataset, metrics=[metric])

print(results)
