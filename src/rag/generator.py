# Copyright 2026 Sudarshan A.
# Licensed under the Apache License, Version 2.0.
"""
In this module, we implement the `Generator` class, which combines the previous modules to provide a complete answer based on the retrieval of relevant documents and the generation of a response. The `Generator` class utilizes the `Retriever` to fetch relevant information and then uses a language model to generate a coherent and contextually appropriate answer based on the retrieved data.
"""

from ollama import chat

from src.rag.retriever import RetrievedChunk


class Generator:
    def __init__(self, model_name: str,) -> None:
        self.model_name = model_name

    def generate(self, query: str, chunks: list[RetrievedChunk],) -> str:

        # Parameter checks
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not chunks:
            raise ValueError("At least one retrieved chunk is required")

        # Context for the model to answer the user question based on retrieved chunks
        context = "\n\n".join(
            f"[Source: {chunk.source}, Section: {chunk.section_id}]\n"
            f"{chunk.text}"
            for chunk in chunks
        )

        # Detailed prompt for the model descibing its scope, behavior and context to answer the user question
        prompt = f"""You are an helpful assistant for Northstar Analytics that works in the domain of enterprise technology and data analytics. A user can ask you questions about HR, compliance, IT, and operational policies of the company. Answer the user's question using ONLY the provided policy context.

DO's:
- Use only the information provided in the policy context to answer the question.
- If the context does not contain enough information to answer the question, say that the available policy information is insufficient.
- Always cite the source of the information you provide in your answer, including the policy name and section number.

DONT's:
- DO NOT provide information that is not present in the policy context.
- DO NOT quote the policies exactly as they are written. Instead, summarize the relevant information in your own words.
- DO NOT invent policies, rules, numbers, dates, or exceptions.
- DO NOT comply to providing sensitive information be it PII or other confidential data.

Policy context:
{context}

User question:
{query}

Answer:"""

        # Answer the user question using the model and the prompt
        response = chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]