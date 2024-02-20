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
from classes import MessageRole, MessageListItem, Message
from langchain.globals import set_debug

set_debug(True)


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




def print_prompt(x):
    print()
    # pprint.pprint(x)
    print(x)
    print()
    return x


class QAResponse:
    def __init__(self):
        # document why this method is empty
        pass

    def get_custom_prompt(self):
        prompt_template = """
      As an advanced and reliable medical chatbot, your foremost priority is to furnish the user with precise, evidence-based health insights and guidance. It is of utmost importance that you strictly adhere to the context provided, without introducing assumptions or extrapolations beyond the given information. Your responses must be deeply rooted in verified medical knowledge and practices. Additionally, you are to underscore the necessity for users to seek direct consultation from healthcare professionals for personalized advice.

      In crafting your response, it is crucial to:
      - Confine your analysis and advice strictly within the parameters of the context provided by the user. Do not deviate or infer details not explicitly mentioned.
      - Identify the key medical facts or principles pertinent to the user's inquiry, applying them directly to the specifics of the provided context.
      - Offer general health information or clarifications that directly respond to the user's concerns, based solely on the context.
      - Discuss recognized medical guidelines or treatment options relevant to the question, always within the scope of general advice and clearly bounded by the context given.
      - Emphasize the critical importance of professional medical consultation for diagnoses or treatment plans, urging the user to consult a healthcare provider.
      - Where applicable, provide actionable health tips or preventive measures that are directly applicable to the context and analysis provided, clarifying these are not substitutes for professional advice.

      Your aim is to deliver a response that is not only informative and specific to the user's question but also responsibly framed within the limitations of non-personalized medical advice. Ensure accuracy, clarity, and a strong directive for the user to seek out professional medical evaluation and consultation. Through this approach, you will assist in enhancing the user's health literacy and decision-making capabilities, always within the context provided and without overstepping the boundaries of general medical guidance.

      Summary: {summary}
      
      Context: {context}
      
      Question: {question}

      """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "summary"]
        )
        return prompt

    def retrival_qa_chain(self):
        printer = RunnableLambda(print_prompt)

        prompt = self.get_custom_prompt()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(get_absolute_path("vectorstore/db_faiss"), embeddings)
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", temperature=0.5, convert_system_message_to_human=True
        )

        contextualize_q_system_prompt = """"""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="Summarise the previous chat. include as much detail as much details in fewest words possible such that it is easy for LLMs to understand and use for context."),
            ]
        )
        contextualize_q_chain = (
            contextualize_q_prompt | printer | llm | printer | StrOutputParser()
        )
        # print(contextualize_q_prompt.format(chat_history = [HumanMessage(content="What does LLM stand for?"),
        #          AIMessage(content="Large language model"),], question="fsfdf" ))
        # contextualize_q_chain.invoke({"history": [HumanMessage(content="What does LLM stand for?"),
        #          AIMessage(content="Large language model"),]})

        def get_summary(x):
            history = x["history"]
            if len(history) > 1:
                return contextualize_q_chain.invoke({"history": history})
            else:
                return "None"

        return (
            RunnableParallel(
                question=lambda x: x["question"],
                context=lambda x: db.as_retriever().invoke(x["question"]),
                summary=get_summary,
            )
            | prompt
            | printer
            | llm
            | printer
            | StrOutputParser()
        )

    def get_response(self, question, history: List[MessageListItem] = []):
        def get_history_object(mesg):
            if mesg.role == MessageRole.assistant:
                return AIMessage(mesg.content)
            else:
                return HumanMessage(mesg.content)

        bot = self.retrival_qa_chain()
        response = bot.invoke(
            {"question": question, "history": list(map(get_history_object, history))}
        )
        return response

