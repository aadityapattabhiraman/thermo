#!/home/akugyo/Programs/Python/ragas/bin/python3

import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset import TestsetGenerator



path = "Docs/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

generator_llm = LangchainLLMWrapper(model)

embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="VAF_TEXT_EMBEDDER",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

generator_embedding = LangchainEmbeddingsWrapper(embedding)

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embedding)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
pd = dataset.to_pandas()
pd.to_csv("test_generated.csv")
