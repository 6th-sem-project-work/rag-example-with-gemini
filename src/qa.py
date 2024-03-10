from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from util import get_absolute_path
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.runnables.base import RunnableEach
from typing import List, Dict
import pprint
from classes import MessageRole, MessageListItem, Message, Model
from langchain.globals import set_debug
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import json
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import requests
from bs4 import BeautifulSoup
import enum
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import LLMChainExtractor

# set_debug(True)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

chatbot_with_history_prompt = """
Summary: {summary}

Context: {context}

Question: {question}

System: provided the summary, context and a question, answer it using the given context. you may offer medical advice. do not deviate from the given context. when context is not related to the question, just say that the context does not have the answer to the question.
"""
chatbot_with_history_promt_template = PromptTemplate(
    template=chatbot_with_history_prompt,
    input_variables=["context", "question", "summary"],
)

chatbot_prompt = """
Context: {context}

Question: {question}

System: provided the context and a question, answer it using the given context. you may offer medical advice. do not deviate from the given context. when context is not related to the question, just say that the context does not have the answer to the question.
"""
chatbot_promt_template = PromptTemplate(
    template=chatbot_prompt, input_variables=["context", "question"]
)

summarization_prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        SystemMessage(
            content="Summarise the chat. include as much detail as much details in the least words possible such that it is easy to understand the context."
        ),
    ]
)

question_rephrase_prompt = """
System: you are an AI. your job is to rephrase the question given below with given context (if required) such that it is possible to answer it without any context. do not change the meaning of question. write the question as if it was written by the user. DO NOT answer the question. just rephrase it.

Context: user and ai are talking about mountains.
Question: what is it?
Rephrased Question: What is a mountain?

Context: {summary}
Question: {question}
Rephrased Question: 
"""
question_rephrase_prompt_template = PromptTemplate(
    template=question_rephrase_prompt, input_variables=["summary", "question"]
)

search_query_prompt = """
System: you are an AI. your job is to generate 3 search queries that are most likely to get the most relevent results from search engines. do not change the meaning of query. DO NOT answer the query. you must output it as a list of strings in json format.

Context: user and ai are talking about mountains.
Question: what is the tallest one called?
Rephrased Question: ["tallest mountains in the world", "mountain heights", "highest peaks in the world"]

Context: {summary}
Question: {question}
Rephrased Question: 
"""
search_query_prompt_template = PromptTemplate(
    template=search_query_prompt, input_variables=["summary", "question"]
)

generic_chatbot_prompt = """
Context: {context}

Question: {question}

System: you are a helpful and smart AI chatbot. provided the context and a question, answer it using the given context. do not deviate from the given context. give detailed and helpful answers.

AI:
"""
generic_chatbot_promt_template = PromptTemplate(
    template=generic_chatbot_prompt, input_variables=["context", "question"]
)

def printer_print(x):
    print()
    pprint.pprint(x)
    print()
    return x
printer = RunnableLambda(printer_print)


class QaService:
    def get_model(self, model: Model, temperature=0.5):
        if model == Model.gemini_pro:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                convert_system_message_to_human=True,
                temperature=temperature,
            )
        elif model == Model.llama2 or model == Model.llama2_uncensored:
            llm = Ollama(model=model.value + ":vram-34", temperature=temperature)
        else:
            raise RuntimeError("unknown llm")

        return llm

    def get_embeddings(self, model: Model):
        if model == Model.gemini_pro:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        elif model == Model.llama2 or model == Model.llama2_uncensored:
            embeddings = OllamaEmbeddings(model=model.value + ":vram-34")
        else:
            raise RuntimeError("unknown embedding model")

        return embeddings

    def summary_chain(self, llm):
        summarization_chain = (
            RunnableLambda(
                lambda x: PromptTemplate(
                    template=summarization_prompt_template.invoke(x).to_string(),
                    input_variables=[],
                ).invoke({})
            )
            | llm
            | StrOutputParser()
        )

        def get_summary(x):
            def get_history_object(mesg):
                if mesg.role == MessageRole.assistant:
                    return AIMessage(mesg.content)
                else:
                    return HumanMessage(mesg.content)

            history = x["history"]

            if len(history) > 1:
                return summarization_chain.invoke(
                    {"history": list(map(get_history_object, history))}
                )
            else:
                return "None"

        return RunnableLambda(get_summary)


class VectorDbQaService(QaService):
    def __init__(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.db = FAISS.load_local(
            get_absolute_path("vectorstore/db_faiss"), embeddings
        )

    def qa_chain(self, model: Model):
        llm = self.get_model(model, 0)

        return (
            RunnableParallel(
                question=lambda x: x["question"],
                summary=self.summary_chain(llm),
            )
            | printer
            | RunnableParallel(
                question=(question_rephrase_prompt_template | llm | StrOutputParser()),
            )
            | printer
            | RunnableParallel(
                question=lambda x: x["question"],
                context=lambda x: self.db.as_retriever().invoke(x["question"]),
            )
            | printer
            | RunnableParallel(
                response=(chatbot_promt_template | llm | StrOutputParser()),
                context=lambda x: x["context"],
            )
            | printer
        )

    def get_response(
        self, question: str, model: Model, history: List[MessageListItem] = []
    ):
        bot = self.qa_chain(model)
        response = bot.invoke({"question": question, "history": history})
        return response

class InternetQaService(QaService):
    def __init__(self):
        self.session = None

    class UrlExtractionMethod(enum.Enum):
        CHROME = "chrome"
        REQUESTS = "requests"

    class AllDocsRetriever(BaseRetriever):
        docs: List[Document] = []

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return self.docs

    def url_content_extraction_chain(self, method: UrlExtractionMethod):
        tags = ["p", "h1", "h2", "h3", "h4", "h5", "span"]

        def chrome_loader(urls):
            loader = AsyncChromiumLoader(urls)
            html = loader.load()
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(
                html, tags_to_extract=tags
            )
            docs = []
            for doc in docs_transformed:
                doc = {
                    "page_content": doc.page_content,
                    "source": doc.metadata["source"],
                }
                docs.append(doc)
            return docs

        def requests_loader(urls):
            if self.session is None:
                self.session = requests.Session()

            docs = []
            for url in urls:
                response = self.session.get(url, timeout=4)
                soup = BeautifulSoup(
                    response.content, "lxml", from_encoding=response.encoding
                )
                text = ""
                for element in soup.find_all(tags):
                    text += element.text + "\n"
                doc = {
                    "page_content": text,
                    "source": url,
                }
                docs.append(doc)
            return docs

        if method == InternetQaService.UrlExtractionMethod.CHROME:
            loader = chrome_loader
        elif method == InternetQaService.UrlExtractionMethod.REQUESTS:
            loader = requests_loader
        else:
            raise RuntimeError("unknown url extraction method")

        return RunnableLambda(lambda x: loader(x["urls"]))

    def generate_questions_chain(self, llm):
        return (
            RunnableParallel(
                questions=(
                    search_query_prompt_template
                    | llm
                    | StrOutputParser()
                    | RunnableLambda(lambda x: json.loads(x))
                ),
                question=lambda x: x["question"],
            )
            | printer
            | RunnableLambda(lambda x: [x["question"]] + x["questions"])
            | printer
        )

    def search_engine_chain(self):
        ddg = DuckDuckGoSearchAPIWrapper()

        def search(query):
            res = ddg.results(query, 3)
            # snippet title link
            return list(map(lambda x: x["link"], res))

        return RunnableLambda(search)

    def web_context_chain(self, llm, embeddings):
        def retriever(pages):
            # - [Contextual compression | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/)
            # - [Retrievers | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/)

            docs = [
                Document(
                    page_content=d["page_content"], metadata={"source": d["source"]}
                )
                for d in pages
            ]
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )

            # retriever = InternetQaService.AllDocsRetriever(docs=docs)
            # filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
            # compressor = DocumentCompressorPipeline(
            #     transformers=[splitter, filter]
            # )
            # # compressor = LLMChainExtractor.from_llm(llm)
            # contextual_retriever = ContextualCompressionRetriever(
            #     base_compressor=compressor, base_retriever=retriever
            # )
            # return contextual_retriever

            faiss = FAISS.from_documents(splitter.split_documents(docs), embeddings)
            retriever = faiss.as_retriever(search_kwargs={"k": 5})
            return retriever

        return (
            RunnableParallel(
                question=lambda x: x["question"],
                pages=(
                    self.generate_questions_chain(llm)
                    | RunnableEach(
                        bound=(
                            RunnableParallel(
                                question=lambda x: x,
                                urls=self.search_engine_chain(),
                            )
                            | RunnableParallel(
                                question=lambda x: x["question"],
                                pages=self.url_content_extraction_chain(
                                    InternetQaService.UrlExtractionMethod.CHROME
                                ),
                            )
                        )
                    )
                    | printer
                    | RunnableLambda(lambda x: [d for docs in x for d in docs["pages"]])
                ),
            )
            | printer
            | RunnableLambda(lambda x: retriever(x["pages"]).invoke(x["question"]))
            | RunnableLambda(
                lambda x: [
                    Document(page_content=d.page_content, metadata=d.metadata)
                    for d in x
                ]
            )
            | printer
        )

    def qa_chain(self, model: Model):
        llm = self.get_model(model)
        embeddings = self.get_embeddings(model)

        return (
            RunnableParallel(
                question=lambda x: x["question"],
                summary=self.summary_chain(llm),
            )
            | printer
            | RunnableParallel(
                question=(
                    question_rephrase_prompt_template
                    | printer
                    | llm
                    | StrOutputParser()
                ),
                summary=lambda x: x["summary"],
            )
            | printer
            | RunnableParallel(
                question=lambda x: x["question"],
                summary=lambda x: x["summary"],
                context=self.web_context_chain(llm, embeddings),
            )
            | printer
            | generic_chatbot_promt_template
            | printer
            | llm
            | StrOutputParser()
        )

    def get_response(self, question: str, model: Model, history: List[MessageListItem] = []):
        bot = self.qa_chain(model)
        resp = bot.invoke({"question": question, "history": history})
        return resp
