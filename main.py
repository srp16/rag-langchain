import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COLLECTION_NAME = "adolescentes_redes"
DEFAULT_PDF = Path("docs") / "LOS ADOLESCENTES Y EL USO DE LAS REDES SOCIALES.pdf"
DEFAULT_QUESTION = "¿En qué bachillerato se realiza el estudio?"


def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Falta {name}. Cópiala en .env a partir de .env.example.")
    return value


def build_store() -> PGVector:
    return PGVector(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=COLLECTION_NAME,
        connection=env("DATABASE_URL"),
        use_jsonb=True,
    )


def index_pdf(store: PGVector, pdf_path: Path) -> int:
    if not pdf_path.is_file():
        raise SystemExit(f"No está el PDF: {pdf_path}")

    pages = PyPDFLoader(str(pdf_path)).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(pages)
    store.add_documents(chunks)
    return len(chunks)


def join_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def ask(store: PGVector, question: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Responde en español usando solo el contexto.
Si la respuesta no está en el contexto, dilo.

Contexto:
{context}

Pregunta: {question}"""
    )
    chain = (
        {
            "context": store.as_retriever(search_kwargs={"k": 4}) | join_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    return chain.invoke(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG de consola sobre un PDF")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Parte el PDF y guarda los embeddings en pgvector",
    )
    parser.add_argument("question", nargs="?", help="Pregunta sobre el documento")
    args = parser.parse_args()

    store = build_store()
    if args.index:
        pdf_path = Path(os.getenv("PDF_PATH", DEFAULT_PDF))
        count = index_pdf(store, pdf_path)
        print(f"Indexados {count} chunks en '{COLLECTION_NAME}'.")

    if args.question or not args.index:
        print(ask(store, args.question or DEFAULT_QUESTION))


if __name__ == "__main__":
    main()
