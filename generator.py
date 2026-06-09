"""Generate grounded answers from retrieved chunks using Groq.

Loads the API key from .env, sends the retrieved context to
llama-3.3-70b-versatile with a strict no-outside-knowledge system prompt,
and returns the answer alongside its source documents.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Answer using only the information in the provided documents. "
    "If the answer is not in the provided text, say I don't have enough "
    "information on that. Do not draw on outside knowledge."
)

_client = None


def get_client():
    """Return a shared Groq client, created on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add "
                "your key from https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def format_context(chunks):
    """Render retrieved chunks into a numbered, source-attributed context block."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        blocks.append(f"[Document {i}] (source: {source})\n{text}")
    return "\n\n".join(blocks)


def generate_response(query, chunks):
    """Answer a query using only the supplied chunks.

    Returns a dict with:
      - answer: the model's grounded answer string
      - sources: list of unique source filenames used as context
    """
    context = format_context(chunks)
    user_message = (
        f"Context documents:\n\n{context}\n\n"
        f"Question: {query}"
    )

    completion = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    answer = completion.choices[0].message.content

    # Preserve order while de-duplicating source filenames.
    sources = list(dict.fromkeys(c.get("source", "unknown") for c in chunks))

    return {"answer": answer, "sources": sources}


def main():
    from retriever import retrieve

    query = "What do students say about Professor Abiodun Robert?"
    chunks = retrieve(query, top_k=4)
    result = generate_response(query, chunks)

    print(f"Query: {query}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
