from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from util import get_absolute_path
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain.globals import set_debug

set_debug(True)

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))



def get_custom_prompt():
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

    
    Context: {context}
    
    Question: {question}

    """
   prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])
   return prompt


def retrival_qa_chain():
   prompt=get_custom_prompt()
   embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
   db = FAISS.load_local(get_absolute_path("vectorstore/db_faiss"), embeddings)
   llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5,convert_system_message_to_human=True)
   qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type='stuff',
      retriever=db.as_retriever(search_kwargs={'k':5}),
      return_source_documents=True,
      chain_type_kwargs={'prompt':prompt}
   )
   return qa_chain



def get_response(query):
   bot = retrival_qa_chain()
   response = bot.invoke(query)
   print(response)
   return response
