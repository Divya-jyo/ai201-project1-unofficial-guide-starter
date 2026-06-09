"""Gradio UI for the professor-review RAG pipeline.

Ties together retrieval (retriever.py) and generation (generator.py):
ask() retrieves the top 4 chunks for a question, then generates a grounded
answer with its sources.
"""

import gradio as gr

from retriever import retrieve
from generator import generate_response


def ask(question):
    """Retrieve the top 4 relevant chunks and generate a grounded answer.

    Returns (answer, sources) as strings for the two output textboxes.
    """
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    chunks = retrieve(question, top_k=4)
    result = generate_response(question, chunks)

    sources = "\n".join(result["sources"])
    return result["answer"], sources


def build_demo():
    with gr.Blocks(title="Professor Review Assistant") as demo:
        gr.Markdown("# Professor Review Assistant")
        gr.Markdown(
            "Ask about Campbellsville University professors. Answers are "
            "grounded only in the indexed review documents."
        )

        question = gr.Textbox(
            label="Your question",
            placeholder="What do students say about Professor Abiodun Robert?",
            lines=2,
        )
        ask_button = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8)
        sources = gr.Textbox(label="Sources", lines=4)

        ask_button.click(fn=ask, inputs=question, outputs=[answer, sources])
        question.submit(fn=ask, inputs=question, outputs=[answer, sources])

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
