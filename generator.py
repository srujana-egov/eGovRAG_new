import os
from dotenv import load_dotenv
load_dotenv()
import openai

def chat_with_assistant(query, docs, model="gpt-5"):
    """
    Calls OpenAI (or compatible) chat completion model with query and supporting docs.
    Returns the generated answer as a string.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment variables.")
    client = openai.OpenAI(api_key=api_key)
    context = "\n\n".join(docs)
    prompt = f"""You are a precise assistant that answers questions based strictly on the provided context.
Rules:
1. Use ONLY information from the context
2. Keep exact terminology and steps from the source
3. If multiple sources have different information, specify which source you're using
4. If information isn't in the context, say "I don't have enough information"
5. For procedures, list exact steps in order
6. Include specific buttons, links, and UI elements mentioned in the source.

Verbosity level: high
Reasoning effort: high

Context:
{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=800  # allow longer answers for recall coverage
        # temperature=0.3 removed for model compatibility
    )
    return response.choices[0].message.content.strip()

def generate_rag_answer(query, hybrid_retrieve_pg, top_k=5, model="gpt-5"):
    docs_and_meta = hybrid_retrieve_pg(query, top_k)
    docs = [doc for doc, meta in docs_and_meta]
    return chat_with_assistant(query, docs, model=model)

if __name__ == "__main__":
    from .retrieval import hybrid_retrieve_pg
    q = input("Enter user query: ")
    answer = generate_rag_answer(q, hybrid_retrieve_pg, model="gpt-5")
    print("\nRAG Answer:\n", answer)
