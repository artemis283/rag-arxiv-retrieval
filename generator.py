import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_cited_answer(query: str, chunks: list[dict]) -> dict:
    """Generate an answer with citations from retrieved chunks."""

    # Build context with citation labels
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[{i}] (Paper: {chunk['arxiv_id']}, Section: {chunk['section']})\n"
        context += f"{chunk['content']}\n\n"

    prompt = f"""You are a research assistant. Answer the user's question using ONLY the provided sources.

Rules:
- Cite every claim using [1], [2], etc. matching the source numbers.
- If the sources don't contain enough information, say so.
- Be concise and precise.
- Do not make up information not in the sources.

Sources:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )

    return {
        "answer": response.choices[0].message.content,
        "model": "gpt-4o-mini",
        "tokens_used": response.usage.total_tokens,
    }