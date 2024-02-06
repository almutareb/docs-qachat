# import for typing
from langchain.chains import RetrievalQAWithSourcesChain

# gradio
import gradio as gr
#import random
import time
#boto3 for S3 access
import boto3
from botocore import UNSIGNED
from botocore.client import Config
# access .env file
import os
from dotenv import load_dotenv
#from bs4 import BeautifulSoup
# HF libraries
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
# vectorestore
from langchain.vectorstores import Chroma
#from langchain.vectorstores import FAISS
# retrieval chain
from langchain.chains import RetrievalQA
# from langchain.chains import RetrievalQAWithSourcesChain
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# logging
import logging
#import zipfile
# improve results with retriever
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain.retrievers.document_compressors import EmbeddingsFilter
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# reorder retrived documents
# from langchain.document_transformers import LongContextReorder
# github issues
#from langchain.document_loaders import GitHubIssuesLoader
# debugging
from langchain.globals import set_verbose
# caching
from langchain.globals import set_llm_cache
#from langchain.cache import InMemoryCache
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
#set_llm_cache(InMemoryCache())

set_verbose(True)

# load .env variables
config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
AWS_S3_LOCATION=os.getenv('AWS_S3_LOCATION')
AWS_S3_FILE=os.getenv('AWS_S3_FILE')
VS_DESTINATION=os.getenv('VS_DESTINATION')

# initialize Model config
# HuggingFaceH4/zephyr-7b-beta
# mistralai/Mistral-7B-Instruct-v0.1
model_id = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={
    "temperature":0.1, 
    "max_new_tokens":1024, 
    "repetition_penalty":1.2, 
#    "streaming": True, 
#    "return_full_text":True
    })

#model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceHubEmbeddings(repo_id=model_name)

# remove old vectorstore
if os.path.exists(VS_DESTINATION):
    os.remove(VS_DESTINATION)

# remove old sqlite cache
if os.path.exists('.langchain.sqlite'):
    os.remove('.langchain.sqlite')

set_llm_cache(SQLiteCache(database_path=".langchain.sqlite"))

# retrieve vectorsrore
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

## Chroma DB
s3.download_file(AWS_S3_LOCATION, AWS_S3_FILE, VS_DESTINATION)
# use the cached embeddings instead of embeddings to speed up re-retrival
db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
db.get()

## FAISS DB
# s3.download_file('rad-rag-demos', 'vectorstores/faiss_db_ray.zip', './chroma_db/faiss_db_ray.zip')
# with zipfile.ZipFile('./chroma_db/faiss_db_ray.zip', 'r') as zip_ref:
#     zip_ref.extractall('./chroma_db/')

# FAISS_INDEX_PATH='./chroma_db/faiss_db_ray'
# db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

# initialize the bm25 retriever and chroma/faiss retriever
# bm25_retriever = BM25Retriever.
# bm25_retriever.k = 2

# Retrieve more documents with higher diversity useful if your dataset has many similar documents
retriever = db.as_retriever(search_type="mmr")#, search_kwargs={'k': 3, 'lambda_mult': 0.25})
# Above a certain threshold
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={'score_threshold': 0.6}
# )

# # asks LLM to create 3 alternatives baed on user query
# multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=model_id)

# # asks LLM to extract relevant parts from retrieved documents
# compressor = LLMChainExtractor.from_llm(model_id)
# compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=multi_retriever)

global qa 
template = """
You are the friendly documentation AI buddy Arti, who helps the Human in using RAY, the open-source unified framework for scaling AI and Python applications.
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question :
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
memory = ConversationBufferMemory(memory_key="history", input_key="question")

# logging for the chain
logging.basicConfig()
logging.getLogger("langchain.retrievers").setLevel(logging.INFO)    
logging.getLogger("langchain.chains").setLevel(logging.INFO)    



qa = RetrievalQA.from_chain_type(llm=model_id, retriever=retriever, return_source_documents=True, verbose=True, chain_type_kwargs={
    "verbose": True,
    "memory": memory,
    "prompt": prompt
}
    )
# qa = RetrievalQAWithSourcesChain.from_chain_type(llm=model_id, retriever=retriever, return_source_documents=True, verbose=True, chain_type_kwargs={
#     "verbose": True,
#     "memory": memory,
#     "prompt": prompt,
#     "document_variable_name": "context"
# }
#     )


#####
#
# Gradio fns
####

def create_gradio_interface(qa:RetrievalQAWithSourcesChain):
    """
    Create a gradio interface for the QA model
    """
    def add_text(history, text):
        history = history + [(text, None)]
        return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    sources = [doc.metadata.get("source") for doc in response['source_documents']]
    src_list = '\n'.join(sources)
    print_this = response['result'] + "\n\n\n Sources: \n\n\n" + src_list

    history[-1][1] = ""
    for character in response['result']: #print_this: # 
        #print_this:
        history[-1][1] += character
        time.sleep(0.01)
        yield history
    # history[-1][1] = print_this #response['answer']
    # return history

    def infer(question, history):
        query =  question
        result = qa({"query": query, "history": history, "question": question})
        return result

css="""
#col-container {min-width: 800px; max-width: 1920px; margin-left: auto; margin-right: auto;}
"""

    title = """
    <div style="text-align: center;max-width: 1920px;">
        <h1>Chat with your Documentation</h1>
        <p style="text-align: center;">This is a privately hosten Docs AI Buddy, <br />
        It will help you with any question regarding the documentation of Ray ;)</p>
    </div>
    """



    with gr.Blocks(css=css) as demo:
        with gr.Column(min_width=900, elem_id="col-container"):
            gr.HTML(title)      
            chatbot = gr.Chatbot([], elem_id="chatbot")
            #with gr.Row():
            #    clear = gr.Button("Clear")

            with gr.Row():
                question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
            with gr.Row():
                clear = gr.ClearButton([chatbot, question])

        question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot
        )
        #clear.click(lambda: None, None, chatbot, queue=False)
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface(qa)
    demo.queue().launch()
