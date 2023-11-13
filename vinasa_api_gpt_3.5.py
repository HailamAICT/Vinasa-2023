# -*- coding: utf-8 -*

#Import nessesary library
import os
import openai
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
import nest_asyncio
nest_asyncio.apply()
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
import os
import openai
from llama_index import Document, ListIndex,KeywordTableIndex
from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index.retrievers import RecursiveRetriever
from llama_index.schema import IndexNode
from llama_index.llms import OpenAI
from pathlib import Path
from typing import List
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore
from IPython.display import Markdown, display
from langchain.chat_models import ChatOpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.prompts.prompts import QuestionAnswerPrompt
import ctypes
from ctypes.util import find_library
find_library("".join(("gsdll", str(ctypes.sizeof(ctypes.c_voidp) * 8), ".dll")))
import camelot
import cv2
from langchain.chat_models import ChatOpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
import time
import tiktoken
from llama_index.text_splitter import SentenceSplitter
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index.schema import IndexNode
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
import logging
import sys
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager, WandbCallbackHandler
from llama_index import load_index_from_storage
from llama_index.memory import ChatMemoryBuffer
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from multiprocessing.pool import ThreadPool
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
# !pip install docx2txt

# datadoc = SimpleDirectoryReader("/content/drive/MyDrive/hocpyt/ptit_response/Data 29-10-2023").load_data()

# text_splitter = SentenceSplitter(
#   separator=" ",
#   chunk_size=2048,
#   chunk_overlap=0,
#   paragraph_separator="\n\n\n",
#   secondary_chunking_regex="[^,.;。]+[,.;。]?",
#   tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
# )
# node = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
# data = node.get_nodes_from_documents(datadoc)

# text_splitternho = SentenceSplitter(
#   separator=" ",
#   chunk_size=350,
#   chunk_overlap=15,
#   paragraph_separator="\n\n\n",
#   secondary_chunking_regex="[^,.;。]+[,.;。]?",
#   tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
# )
# nodenho = SimpleNodeParser.from_defaults(text_splitter=text_splitternho)
# datanho = nodenho.get_nodes_from_documents(datadoc)

# index350=VectorStoreIndex(data+datanho,show_progress=True)

# PROMPT_TEMPLATE = (
#     "Tôi là Vinades một trợ lý giúp bạn có thể hiểu hơn về thông tư, quy định và quy chế của luật đấu thầu"
#     "Dưới đây là câu hỏi của khách hàng về luật liên quan đến đấu thầu, hãy phân tích ngữ cảnh thật rõ vì đây là lĩnh vực liên quan đến luật pháp để trả lời câu hỏi cuối"
#     "\n -----------------------------\n"
#     "Thôn tin về ngữ cảnh câu hỏi liên quan đến luật pháp và đấu thầu{context_str}"
#     "\n -----------------------------\n"
#     "Trả lời câu hỏi bằng tiếng Việt Nam: {query_str}?"
# )
# QA_PROMPT = QuestionAnswerPrompt(PROMPT_TEMPLATE)
PROMPT_TEMPLATE = (
    "Bạn là : Chatbot sẽ giúp các doanh nghiệp được giải đáp các vấn đề pháp lý về luật đấu thầu mới cũng như nhiều thông tư, nghị định liên quan ở Việt Nam"
    "dưới đây là câu hỏi của người dùng"
    "\n -----------------------------\n"
    "Thông tin về pháp luật đấu thầu của Việt Nam: {context_str}"
    "\n -----------------------------\n"
    "Trả lời câu hỏi bằng tiếng Việt Nam: {query_str}?"
    # "nếu có câu trả lời hãy đưa cho người dùng tên văn bản mà bạn truy xuất thông tin đó"
    # "nếu người dùng yêu cầu lời khuyên thì dựa vào những kiến thức của bạn hãy đưa ra lời khuyên cho người dùng"
    "Nếu không có câu trả lời chính xác thì không được bịa ra câu trả lời mà phải trả lời như sau : (cảm ơn bạn đã đưa ra câu hỏi, nhưng tiếc quá mình chưa có dữ liệu về câu hỏi của bạn nhưng mình có thể giúp bạn thông qua trang web này nhé : https://dauthau.asia/van-ban-dau-thau/)"

)
# index350.storage_context.persist("/content/drive/MyDrive/hocpyt/ptit_chatbot")
storage_context = StorageContext.from_defaults(persist_dir="Vinasa1")

index = load_index_from_storage(storage_context)

QA_PROMPT = QuestionAnswerPrompt(PROMPT_TEMPLATE)

llm = OpenAI(model='ggpt-3.5-turbo-16k' ,temperature=0)
embed_model = OpenAIEmbedding()

prompt_helper = PromptHelper(
  context_window=4096,
  num_output=-1,
  chunk_overlap_ratio=0.1,
  chunk_size_limit=None
)
text_splitterllm = SentenceSplitter(
  separator=" ",
  chunk_size=512,
  chunk_overlap=20,
  paragraph_separator="\n\n\n",
  secondary_chunking_regex="[^,.;。]+[,.;。]?",
  tokenizer=tiktoken.encoding_for_model("ggpt-3.5-turbo-16k").encode)
node_parserllm = SimpleNodeParser.from_defaults(text_splitter=text_splitterllm)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parserllm,
  prompt_helper=prompt_helper
)


from langchain.chains.conversation.memory import ConversationBufferMemory

query_engine = index.as_query_engine( similarity_top_k=2,
                                         temperature=0,
                                             text_qa_template=QA_PROMPT,
                                             service_context=service_context,
                                             verbose=False,
                                             top_p=0.97,
                                             memory=ConversationBufferMemory(token_limit=1500),
                                             max_tokens=-1)



app = FastAPI()

class QueryRequest(BaseModel):
    question: str

def query_ans(query, query_engine):
    response = query_engine.query(query)
    return response

@app.post("/")
def query_handler(request: QueryRequest):
    query = request.question
    pool = ThreadPool(processes=10)
    response_ans = pool.apply_async(query_ans, (query, query_engine))
    chatbot = response_ans.get()
    source = "Những văn bản luật mà chúng tôi đã tìm kiếm là: "

    k = [chatbot.source_nodes[0].metadata['file_path'].split("/")[-1].split(".")[0]]
    for i in range(1,2):
        h = chatbot.source_nodes[i].metadata['file_path'].split("/")[-1].split(".")[0]
        if h != chatbot.source_nodes[i-1].metadata['file_path'].split("/")[-1].split(".")[0]:
            k.append(h)

    for i in k:
        i = i.replace("]thong-tu-21-2022-tt-bkhdt-bo-ke-hoach-","")
        source += i.lower() + ", "
    source = source[:-2] + "."

    now = time.localtime()
    formatted_time = time.strftime("%d/%m/%Y %H:%M:%S", now)
    with open("request.txt", "a",encoding="utf-8") as f:
          f.write(formatted_time+"\n"+query+"\n"+str(chatbot)+ "\n"+"\n"+"\n")
    f.close()
    print(chatbot)
    chatbot.response += "\n" +  source
    print(chatbot)
    return chatbot

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5023)

