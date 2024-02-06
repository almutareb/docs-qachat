
# logging
import logging

# access .env file
import os
from dotenv import load_dotenv

import time

#boto3 for S3 access
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# HF libraries
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
# vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
import zipfile

# retrieval chain
from langchain.chains import RetrievalQAWithSourcesChain
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# github issues
#from langchain.document_loaders import GitHubIssuesLoader
# debugging
from langchain.globals import set_verbose
# caching
from langchain.globals import set_llm_cache
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache


# template for prompt
from prompt import template



set_verbose(True)

# set up logging for the chain
logging.basicConfig()
logging.getLogger("langchain.retrievers").setLevel(logging.INFO)    
logging.getLogger("langchain.chains.qa_with_sources").setLevel(logging.INFO)    

# load .env variables
config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
AWS_S3_LOCATION=os.getenv('AWS_S3_LOCATION')
AWS_S3_FILE=os.getenv('AWS_S3_FILE')
VS_DESTINATION=os.getenv('VS_DESTINATION')

# remove old vectorstore
if os.path.exists(VS_DESTINATION):
    os.remove(VS_DESTINATION)

# remove old sqlite cache
if os.path.exists('.langchain.sqlite'):
    os.remove('.langchain.sqlite')



# initialize Model config
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# changed named to model_id to llm as is common
llm = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={
    # "temperature":0.1, 
    "max_new_tokens":1024, 
    "repetition_penalty":1.2, 
#    "streaming": True, 
#    "return_full_text":True
    })

# initialize Embedding config
embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embeddings = HuggingFaceHubEmbeddings(repo_id=embedding_model_name)

set_llm_cache(SQLiteCache(database_path=".langchain.sqlite"))

# retrieve vectorsrore
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

## download vectorstore from S3
s3.download_file(AWS_S3_LOCATION, AWS_S3_FILE, VS_DESTINATION)
with zipfile.ZipFile(VS_DESTINATION, 'r') as zip_ref:
    zip_ref.extractall('./vectorstore/')

FAISS_INDEX_PATH='./vectorstore/lc-faiss-multi-qa-mpnet'
db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

# use the cached embeddings instead of embeddings to speed up re-retrival
# db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
# db.get()

retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.25})

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
memory = ConversationBufferMemory(memory_key="history", input_key="question")

qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, verbose=True, chain_type_kwargs={
    "verbose": True,
    "memory": memory,
    "prompt": prompt,
    "document_variable_name": "context"
}
    )
