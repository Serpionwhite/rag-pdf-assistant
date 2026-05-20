"""QA chain: wires a retriever to an LLM to answer questions."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI


def _format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vectorstore: Chroma, k: int = 4) -> Runnable:
    """Build a retrieval-augmented QA chain from a vectorstore.

    The chain works in two steps at query time:
      1. Retrieve: find the top-k most relevant chunks from the vectorstore.
      2. Generate: pass those chunks + the question to the LLM via a prompt.

    Args:
        vectorstore: A ``Chroma`` instance from ``build_vectorstore()``
            or ``load_vectorstore()``.
        k: Number of document chunks to retrieve per question.

    Returns:
        A LangChain Runnable. Invoke it with a plain string question:
            answer = chain.invoke("What is this document about?")
            print(answer)  # returns a string directly
    """


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
             ("system", "Answer using only the context below.\n\nContext: {context}"),
             ("human", "{input}"),
         ])
    
    chain = (
             {"context": retriever | _format_docs, "input": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser()
         )
    

    return chain
