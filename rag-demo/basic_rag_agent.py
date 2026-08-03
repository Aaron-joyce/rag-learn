import os
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use cosine distance so similarity calculation (1 - distance) works properly
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine", "description": "Document store for RAG agent"}
)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_break = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("\n"))
            if last_break > chunk_size // 2:
                end = start + last_break + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


def load_and_index_document(docs_path: str):
    docs_dir = Path(docs_path)

    if not docs_dir.exists():
        print(f"Creating Docs Directory: {docs_path}")
        docs_dir.mkdir(parents=True)
        return

    indexed_count = 0

    for file_path in docs_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in [".txt", ".md", ".py", ".js", ".ts"]:
            try:
                content = file_path.read_text(encoding="utf-8")

                if len(content) < 50:
                    continue

                chunks = chunk_text(content)
                embeddings = embedding_model.encode(chunks).tolist()

                ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]
                metadatas = [
                    {"source": str(file_path), "chunk_index": i}
                    for i in range(len(chunks))
                ]

                # Use upsert to prevent duplicates on repeated runs
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas
                )

                indexed_count += len(chunks)
                print(f"Indexed: {file_path.name} ({len(chunks)} chunks)")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"\nTotal Chunks indexed: {indexed_count}")


def search_documents(query: str, n_results: int = 5) -> list[dict]:
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            formatted.append(
                {
                    "content": results["documents"][0][i],
                    "source": results['metadatas'][0][i]["source"],
                    "relevance_score": 1 - results['distances'][0][i]
                }
            )

    return formatted


# Python Function Tools (Gemini automatically parses docstrings and type hints)
def search_knowledge_base(query: str, num_results: int = 5) -> str:
    """Search the indexed documents for relevant information using semantic similarity.

    Args:
        query: The natural language query to search for in the documents.
        num_results: Number of document sections to return (max 10).
    """
    num_results = min(num_results, 10)
    results = search_documents(query, n_results=num_results)

    if not results:
        return "No relevant results found. Try another query."

    formatted = f"Found {len(results)} relevant document sections:\n\n"
    for i, r in enumerate(results, 1):
        formatted += f"---- Result {i} (relevance: {r['relevance_score']:.2f}) ---\n"
        formatted += f"source: {r['source']}\n"
        formatted += f"content:\n{r['content']}\n\n"

    return formatted


def list_available_documents() -> str:
    """List all document filenames that have been indexed in the knowledge base."""
    all_data = collection.get(include=['metadatas'])

    if not all_data["metadatas"]:
        return "No documents have been indexed yet."

    sources = {metadata["source"] for metadata in all_data["metadatas"]}

    result = f"Indexed Documents ({len(sources)} files):\n\n"
    for source in sorted(sources):
        result += f" - {source}\n"

    return result


def run_rag_agent(question: str) -> str:
    print("\n" + "=" * 30)
    print("RAG AGENT")
    print("=" * 30)
    print(f"\nQuestion: {question}\n")
    print("-" * 60)

    system_prompt = """You are a knowledgeable assistant with access to a document knowledge base.

When answering questions:
1. SEARCH FIRST - Always search the knowledge base before answering questions.
2. BE THOROUGH - If initial results are weak, try alternate search terms.
3. CITE SOURCES - Reference which documents your information comes from.
4. BE HONEST - State clearly if the requested information is absent.

Format your response as:
**Answer:** [Your comprehensive answer]
**Sources:**
- [List source documents used]"""

    # Automatic Function Calling handles tool execution natively
    response = client.models.generate_content(
        model="gemini-3.5 -flash",
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[search_knowledge_base, list_available_documents],
            temperature=0.2,
        ),
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(f"\n{response.text}\n")

    return response.text


if __name__ == "__main__":

    ## check for working gemini models
    # for m in client.models.list():
    #     for action in m.supported_actions:
    #         if action == "generateContent":
    #             print(m.name)
    print("Loading and indexing documents...")
    load_and_index_document("docs/basic-rag-agent")

    questions = [
        "How do I install this project?",
        "What authentication methods are available?",
        "How do I update a user profile via the API?",
    ]

    for q in questions:
        run_rag_agent(q)
        print("\n" + "#" * 80 + "\n")
