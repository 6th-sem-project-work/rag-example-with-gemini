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
from typing import List
import pprint
from classes import MessageRole, MessageListItem, Message, Model
from langchain.globals import set_debug
from langchain_community.llms import Ollama

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
    template=chatbot_with_history_prompt, input_variables=["context", "question", "summary"]
)

chatbot_prompt = '''
Context: {context}

Question: {question}

System: provided the context and a question, answer it using the given context. you may offer medical advice. do not deviate from the given context. when context is not related to the question, just say that the context does not have the answer to the question.
'''
chatbot_promt_template = PromptTemplate(
    template=chatbot_prompt, input_variables=["context", "question"]
)

summarization_prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        SystemMessage(
            content="Summarise the previous chat. include as much detail as much details in fewest words possible such that it is easy for LLMs to understand and use for context."
        ),
    ]
)

question_rephrase_prompt = """
Summary: {summary}

Question: {question}

System: given a summary of chat between an assistant and a person, rephrase the question with any context that might make sense to include. make the question self sufficient such that it is possible to answer it without providing any more context. Note: write the question as if it was written by the user and nothing else in the output.

Example Question: i am having a headache. what should i do?

Question: 
"""
question_rephrase_prompt_template = PromptTemplate(
    template=question_rephrase_prompt, input_variables=["summary", "question"]
)

def printer_print(x):
    print()
    pprint.pprint(x)
    print()
    return x
printer = RunnableLambda(printer_print)


class QaService:
    def __init__(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.db = FAISS.load_local(
            get_absolute_path("vectorstore/db_faiss"), embeddings
        )

    def retrival_qa_chain(self, model: Model):
        if model == Model.gemini_pro:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.5,
                convert_system_message_to_human=True,
            )
        elif model == Model.llama2 or model == Model.llama2_uncensored:
            llm = Ollama(model=model.value + ":vram-34")
        else:
            raise RuntimeError("unknown llm")

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

        return (
            RunnableParallel(
                question=lambda x: x["question"],
                summary=get_summary,
            )
            | printer
            | RunnableParallel(
                question=(
                    question_rephrase_prompt_template | llm | StrOutputParser()
                ),
            )
            | printer
            | RunnableParallel(
                question=lambda x: x['question'],
                context=lambda x: self.db.as_retriever().invoke(x["question"]),
            )
            | printer
            | RunnableParallel(
                response=(chatbot_promt_template | llm | StrOutputParser()),
                context=lambda x: x["context"],
            )
        )

    def get_response(self, question, model, history: List[MessageListItem] = []):
        bot = self.retrival_qa_chain(model)
        response = bot.invoke({"question": question, "history": history})
        return response
