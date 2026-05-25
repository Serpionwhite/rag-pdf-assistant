"""QA chain: wires a retriever to an LLM to answer questions."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI


def _format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(retriever: BaseRetriever) -> Runnable:
    """Build a retrieval-augmented QA chain from a retriever.

    The chain works in two steps at query time:
      1. Retrieve: run the retriever to fetch the most relevant chunks.
      2. Generate: pass those chunks + the question to the LLM via a prompt.

    The retriever is built externally (by ``build_retriever()``) so this
    function is agnostic to the retrieval strategy — similarity, MMR, or rerank
    all produce the same interface.

    Args:
        retriever: Any ``BaseRetriever`` from ``build_retriever()``.

    Returns:
        A LangChain Runnable. Invoke it with a plain string question:
            answer = chain.invoke("What is this document about?")
            print(answer)  # returns a string directly
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using only the context below.\n\nContext: {context}"),
            ("human", "{input}"),
        ]
    )

    chain = (
        {"context": retriever | _format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
