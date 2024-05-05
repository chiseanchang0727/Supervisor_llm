from langchain import hub
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import chainlit as cl
from utils import read_json


configs = read_json('configs.json')
DATA_PATH = configs["DATA_PATH"]
DB_PATH = configs['DB_PATH']

# modify the RAG prompt here
#QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

QA_CHAIN_PROMPT = ChatPromptTemplate.from_messages(
  ("human", """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
Question: {question} 
Context: {context} 
Answer: [/INST]""")
)

def load_llm():
    llm = Ollama(
        model = 'llama3',
        verbose = 'True',
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    )
    
    return llm


def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    
    return qa_chain


def qa_bot():
    llm = load_llm()
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings())
    qa = retrieval_qa_chain(llm, vectorstore)
    
    return qa


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content= "Hi, welcome to research info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)
    
    
@cl.on_message
async def main(message):
    chain=cl.user_session.get('chain')
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=['FINAL', 'ANSWER']
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    answer = res['result']
    answer = answer.replace('.', '.\n')
    sources = res['source_documents']
    
    if sources:
        answer+=f"\nSources: "+str(str(sources))
    else:
        answer+=f"\nNo Sources found"
        
    await cl.Message(content=answer).send()