import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

VECTORSTORE_DIR = "rag_pipeline/faiss_index"

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # allow_dangerous_deserialization=True is needed because FAISS saves a pickle file
    # — safe here since WE created this file ourselves in the previous step.
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def format_docs(docs):
    """Combine the retrieved scheme documents into one text block for the prompt."""
    formatted = []
    for doc in docs:
        formatted.append(f"{doc.page_content}\n(Source: {doc.metadata.get('scheme_name')})")
    return "\n\n---\n\n".join(formatted)

def build_chain():
    vectorstore = load_vectorstore()

    # Retriever: fetches the top 3 most relevant scheme documents for a given question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # gpt-4o-mini: OpenAI's lightweight, low-cost chat model — fits your "less weight" requirement
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for Tamil Nadu farmers, answering questions about
government Agriculture schemes. Use ONLY the context below to answer the question.
If the answer isn't in the context, say you don't have that information and
suggest the farmer contact their local Agriculture Officer.

Context:
{context}

Question: {question}

Answer in a simple, clear way a farmer can easily understand:
""")

    # LCEL chain: pipes data through each step in order
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_chain()

    print("🌾 Tamil Nadu Agriculture Scheme Chatbot (type 'exit' to quit)\n")
    while True:
        question = input("Farmer's question: ")
        if question.lower() == "exit":
            break
        answer = chain.invoke(question)
        print(f"\n🤖 Answer: {answer}\n")