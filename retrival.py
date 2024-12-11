from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if _name_ == "_main_":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
    query = input("Ask anything about batman: ")

    prompt_template = PromptTemplate(input_variables=[], template=query)
 
    vectorstore = PineconeVectorStore(index_name="batman-rag-project", embedding=embeddings)

    retriver_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    combined_docs_chain = create_stuff_documents_chain(llm, retriver_qa_chat_prompt)

    retriever_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combined_docs_chain)

    result = retriever_chain.invoke({"input": query})

    print(result)