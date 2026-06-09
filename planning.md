# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

This project focuses on student-generated professor reviews at Campbellsville University. The domain covers CS, Mathematics, Business, and English department professors. This knowledge is valuable because official university channels only provide formal faculty bios and course catalogs — they never tell students whether a professor grades harshly, gives extra credit, is approachable, or makes classes engaging. Students rely on word-of-mouth and platforms like Rate My Professors to make informed decisions about which professors to take. This system makes that scattered knowledge searchable and answerable.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Rate My Professors | Reviews for Prof. Abiodun Robert (CS) | documents/abiodun_robert.txt |
| 2 | Rate My Professors | Reviews for Prof. Thomas Jeffrey (CS) | documents/thomas_jeffrey.txt |
| 3 | Rate My Professors | Reviews for Prof. Shane Green (CS) | documents/shane_green.txt |
| 4 | Rate My Professors | Reviews for Prof. Ehi Aimiuwu (CS) | documents/ehi_aimiuwu.txt |
| 5 | Rate My Professors | Reviews for Prof. Robert Street (CS) | documents/robert_street.txt |
| 6 | Rate My Professors | Reviews for Prof. Troy Young (Math) | documents/troy_young.txt |
| 7 | Rate My Professors | Reviews for Prof. Chris Bullock (Math) | documents/chris_bullock.txt |
| 8 | Rate My Professors | Reviews for Prof. William McNear (Business) | documents/william_mcnear.txt |
| 9 | Rate My Professors | Reviews for Prof. Chryslee Hines (Business) | documents/chryslee_hines.txt |
| 10 | Rate My Professors | Reviews for Prof. Dave Harrity (English) | documents/dave_harrity.txt |

---

## Chunking Strategy

**Chunk size:** 300 characters

**Overlap:** 50 characters

**Reasoning:** The documents are review-style text — short opinions packed with specific facts (grading style, difficulty, attendance policies). A 300-character chunk is large enough to contain one complete student opinion or observation, but small enough to be targeted during retrieval. If chunks were too large (e.g. 1000+ characters), a single chunk might cover multiple unrelated opinions, making it hard to match a specific query. If chunks were too small (e.g. 50 characters), a single sentence would be split across chunks, losing context. The 50-character overlap ensures that a review spanning a chunk boundary is still retrievable as a coherent thought.

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers (runs locally, no API key needed)

**Top-k:** 4

**Production tradeoff reflection:** For a production system, I would consider OpenAI's text-embedding-ada-002 or a larger sentence-transformers model for better accuracy on domain-specific text. Tradeoffs include: cost (API-based models charge per token vs free local models), context length (larger models handle longer chunks), multilingual support (if students write in other languages), and latency (local models are slower on CPU but avoid network calls). For this student review use case, all-MiniLM-L6-v2 is a good fit because reviews are short English text and speed matters.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about Professor Abiodun Robert's teaching style? | Students say he explains concepts clearly with examples, is friendly, and gives good feedback. Highly recommended. |
| 2 | Is Professor Chris Bullock a good math professor? | Mixed reviews — some say he explains well, others say he moves too fast and assumes too much prior knowledge. |
| 3 | Does Professor Chryslee Hines offer extra credit? | Yes, multiple reviews mention she offers extra credit and is very helpful. |
| 4 | What is Professor Shane Green's rating and what do students say? | 1/5 rating. Students say he is unresponsive, grades poorly, and puts in no effort. |
| 5 | What do students say about Professor Dave Harrity's English class workload? | Heavy workload with lots of papers and reading, but he gives good feedback and is caring. |

---

## Anticipated Challenges

1. **Chunks splitting key review information across boundaries:** Some reviews are short (1-2 sentences) and may get split at the wrong point, causing retrieval to return incomplete opinions. This could cause the system to miss the most important part of a review.

2. **Off-topic retrieval across departments:** Since we have professors from CS, Math, Business, and English, a query about one professor might accidentally retrieve chunks from a different professor's reviews if the language is similar (e.g. "good feedback" appears in many reviews). Source metadata will help attribute answers correctly.

---

## Architecture

```
Document Ingestion        Chunking              Embedding + Vector Store
(load .txt files)   -->  (300 char chunks,  --> (all-MiniLM-L6-v2          
(Python / os)             50 char overlap)       sentence-transformers
                          (custom Python)        stored in ChromaDB)
                                                        |
                                                        v
                                                    Retrieval
                                                 (semantic search,
                                                   top-k = 4,
                                                   ChromaDB query)
                                                        |
                                                        v
                                                    Generation
                                                 (Groq API,
                                           llama-3.3-70b-versatile,
                                            grounded response +
                                            source attribution)
                                                        |
                                                        v
                                                  Gradio Web UI
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
I will give Claude my Chunking Strategy section (300 char chunks, 50 char overlap) and my Documents section (10 .txt files in /documents folder) and ask it to implement an ingest.py script that loads all .txt files, cleans them by removing extra whitespace, and splits them into chunks using a sliding window approach. I will verify the output by printing 5 sample chunks and checking they are readable and self-contained.

**Milestone 4 — Embedding and retrieval:**
I will give Claude my Retrieval Approach section (all-MiniLM-L6-v2, top-k=4) and my Architecture diagram and ask it to implement a retriever.py script that embeds all chunks using sentence-transformers, stores them in ChromaDB with source metadata (filename, chunk index), and provides a retrieve() function that returns the top 4 most relevant chunks for a query. I will verify by running 3 test queries and checking distance scores are below 0.5.

**Milestone 5 — Generation and interface:**
I will give Claude my grounding requirement (answer only from retrieved context, cite sources) and ask it to implement a generator.py that calls Groq's llama-3.3-70b-versatile with a strict system prompt, and an app.py with a Gradio UI showing answer and sources. I will verify by asking a question not in any document and checking the system says it doesn't know.