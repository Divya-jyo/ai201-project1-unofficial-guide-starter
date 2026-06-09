# The Unofficial Guide — Project 1

A retrieval-augmented generation (RAG) system that answers questions about
Campbellsville University professors using real student reviews. Ask a natural
question ("Is Professor Chris Bullock a good math professor?") and get an answer
grounded only in the indexed review text, with the source files cited.

**Pipeline:** `ingest.py` (load + chunk) → `retriever.py` (embed + ChromaDB
search) → `generator.py` (Groq LLM, grounded) → `app.py` (Gradio UI).

---

## Domain

This system covers **student-generated professor reviews at Campbellsville
University**, sourced from [Rate My Professors](https://www.ratemyprofessors.com).
It spans four departments — Computer Science, Mathematics, Business, and English.

This knowledge is valuable because official university channels (faculty bios,
course catalogs) only describe *what* a course covers, never *how* it is taught.
They won't tell you whether a professor grades harshly, offers extra credit, moves
too fast through material, or makes class engaging. Students rely on scattered
word-of-mouth and review platforms to decide who to take. This project makes that
scattered, opinion-based knowledge searchable and answerable in plain language.

---

## Document Sources

All ten documents are plain-text exports of Rate My Professors review pages, one
file per professor, stored in [`documents/`](documents/).

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Rate My Professors — Abiodun Robert (Computer Science) | Student reviews (.txt) | `documents/abiodun_robert.txt` |
| 2 | Rate My Professors — Thomas Jeffrey (Computer Science) | Student reviews (.txt) | `documents/thomas_jeffrey.txt` |
| 3 | Rate My Professors — Shane Green (Computer Science) | Student reviews (.txt) | `documents/shane_green.txt` |
| 4 | Rate My Professors — Ehi Aimiuwu (Computer Science) | Student reviews (.txt) | `documents/ehi_aimiuwu.txt` |
| 5 | Rate My Professors — Robert Street (Computer Science) | Student reviews (.txt) | `documents/robert_street.txt` |
| 6 | Rate My Professors — Troy Young (Mathematics) | Student reviews (.txt) | `documents/troy_young.txt` |
| 7 | Rate My Professors — Chris Bullock (Mathematics) | Student reviews (.txt) | `documents/chris_bullock.txt` |
| 8 | Rate My Professors — William McNear (Business) | Student reviews (.txt) | `documents/william_mcnear.txt` |
| 9 | Rate My Professors — Chryslee Hines (Business) | Student reviews (.txt) | `documents/chryslee_hines.txt` |
| 10 | Rate My Professors — Dave Harrity (English) | Student reviews (.txt) | `documents/dave_harrity.txt` |

---

## Chunking Strategy

**Chunk size:** 300 characters

**Overlap:** 50 characters (implemented as a sliding window with a 250-character
step)

**Preprocessing:** Before chunking, each file is read and run through
`clean_text()` ([`ingest.py`](ingest.py)), which collapses all runs of whitespace
(newlines, tabs, repeated spaces) into single spaces and strips the ends. This
flattens the multi-line review layout into continuous text so chunk boundaries
fall on character counts rather than on the document's formatting.

**Why these choices fit the documents:** The documents are review-style text —
short, self-contained student opinions packed with concrete facts (grading style,
difficulty, attendance policy, extra credit). A 300-character chunk is large
enough to hold one complete opinion or observation but small enough to stay
targeted during retrieval. Larger chunks (1000+ chars) would bundle several
unrelated opinions into one vector, blurring the match for a specific query.
Smaller chunks (~50 chars) would split a single review sentence across boundaries
and lose context. The 50-character overlap ensures a thought that straddles a
boundary is still retrievable as a coherent unit in at least one chunk.

**Final chunk count:** **79 chunks** across all 10 documents.

| Source | Chunks | Source | Chunks |
|--------|:------:|--------|:------:|
| abiodun_robert.txt | 9 | ehi_aimiuwu.txt | 3 |
| chris_bullock.txt | 15 | robert_street.txt | 4 |
| chryslee_hines.txt | 13 | shane_green.txt | 3 |
| dave_harrity.txt | 12 | thomas_jeffrey.txt | 7 |
| troy_young.txt | 7 | william_mcnear.txt | 6 |

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, stored and queried
in an in-memory ChromaDB collection using **cosine** distance. It runs locally with
no API key, produces 384-dimensional embeddings, and is fast on CPU.

**Production tradeoff reflection:** If this were deployed for real users and cost
weren't a constraint, I would weigh several tradeoffs against `all-MiniLM-L6-v2`:

- **Accuracy on domain-specific text:** A larger model (e.g. `all-mpnet-base-v2`)
  or a hosted model (OpenAI `text-embedding-3-large`) would likely separate
  near-duplicate review language ("gives good feedback" appears across many
  professors) more cleanly, reducing cross-professor confusion.
- **Context length:** MiniLM truncates at 256 word-pieces. That is fine for
  300-character chunks, but a longer-context model would let me index larger,
  more coherent chunks without splitting reviews.
- **Latency vs. network:** Local MiniLM avoids per-call network latency and keeps
  data private, but is slower per batch on CPU than a GPU-backed API endpoint.
- **Multilingual support:** If students wrote reviews in other languages, a
  multilingual model (`paraphrase-multilingual-MiniLM`) would be necessary.

For *this* use case — short English reviews where speed and zero cost matter —
`all-MiniLM-L6-v2` is a strong default, and the retrieval results below confirm it
returns relevant chunks at usable distances.

---

## Sample Chunks

Five representative chunks produced by `ingest.py` (whitespace already collapsed):

**1. `abiodun_robert.txt` (chunk 0)**
> Professor: Abiodun Robert Department: Computer Science University: Campbellsville University Rating: 5.0/5 Difficulty: 1.8 Would Take Again: 100% --- Review 1 --- Course: DS621 Date: Feb 28th, 2026 Rating: 5.0 | Difficulty: 2.0 For Credit: Yes | Attendance: Mandatory | Online: Yes Grade: Rather not

**2. `chris_bullock.txt` (chunk 3)**
> .0 For Credit: Yes | Attendance: Mandatory Would Take Again: Yes | Grade: A | Textbook: Yes He clearly explains the material and makes sure we understand. The homework is usually about 10 problems, but it is only due 2 class periods after he finishes teaching. He gets a bad rep from students who don

**3. `chryslee_hines.txt` (chunk 6)**
> anything they will mark it wrong. Tags: Accessible outside class, Participation matters, Extra Credit --- Review 4 --- Course: BIT320 Date: Dec 15th, 2020 Rating: 5.0 | Difficulty: 5.0 For Credit: Yes | Attendance: Mandatory | Online: Yes Would Take Again: Yes | Textbook: No Professor Hines is very

**4. `dave_harrity.txt` (chunk 8)**
> and helps you. Highly recommend. Tags: Gives good feedback, Hilarious, Caring --- Review 6 --- Course: ENG210 Date: Dec 9th, 2019 Rating: 5.0 | Difficulty: 3.0 For Credit: Yes | Attendance: Mandatory | Online: Yes Would Take Again: Yes | Grade: A+ | Textbook: Yes It was a good class and he was a gre

**5. `thomas_jeffrey.txt` (chunk 1)**
> ne of my favorite professors and will be missed when I graduate. I've been in several of his classes and I always looked forward to them. --- Review 2 --- Course: BA360 Date: Dec 7th, 2021 Rating: 4.0 | Difficulty: 5.0 For Credit: Yes | Attendance: Mandatory | Online: Yes Would Take Again: Yes | Tex

Note how the overlap and sliding window mean some chunks begin mid-word (`.0`,
`ne of`) — the tradeoff that keeps boundary-spanning opinions retrievable.

---

## Retrieval Test Results

Three queries run through `retrieve()` (top-4, cosine distance — lower is more
similar):

**Query 1: "Which professor offers extra credit?"**

| Distance | Source (chunk) |
|----------|----------------|
| 0.4902 | william_mcnear.txt (3) |
| 0.5203 | thomas_jeffrey.txt (1) |
| 0.5263 | thomas_jeffrey.txt (0) |
| 0.5321 | ehi_aimiuwu.txt (2) |

**Query 2: "Who is a tough grader that moves too fast?"**

| Distance | Source (chunk) |
|----------|----------------|
| 0.5375 | dave_harrity.txt (2) |
| 0.5815 | chris_bullock.txt (4) |
| 0.6170 | shane_green.txt (1) |
| 0.6289 | thomas_jeffrey.txt (6) |

**Query 3: "What professor explains concepts clearly with examples?"**

| Distance | Source (chunk) |
|----------|----------------|
| 0.4300 | troy_young.txt (3) |
| 0.4675 | abiodun_robert.txt (1) |
| 0.4694 | abiodun_robert.txt (7) |
| 0.5064 | abiodun_robert.txt (6) |

Query 3 is the cleanest: the top hits cluster on Troy Young and Abiodun Robert,
both reviewed for clear, example-driven teaching. Query 2 correctly surfaces
Chris Bullock (fast pace) but at higher distances (>0.5), reflecting that
"tough grader" language is spread thinly across many reviews — an early signal of
the cross-professor confusion discussed in the failure analysis.

---

## Grounded Generation

**System prompt grounding instruction:** The generator sends this exact system
message ([`generator.py`](generator.py)) with `temperature=0`:

> Answer using only the information in the provided documents. If the answer is
> not in the provided text, say I don't have enough information on that. Do not
> draw on outside knowledge.

**Structural choices that reinforce grounding:**

- **Context formatting:** Retrieved chunks are injected as a numbered, attributed
  block — each chunk is prefixed with `[Document N] (source: filename.txt)` — so
  the model can ground claims in a specific source and the prompt clearly
  separates "context" from "question."
- **Zero temperature:** `temperature=0` makes generation deterministic and
  discourages creative extrapolation beyond the supplied text.
- **Top-4 retrieval only:** The model never sees the full corpus, only the four
  most relevant chunks, so it physically cannot answer from professors that
  weren't retrieved (this is exactly what produces the honest refusal in Q4).

**How source attribution is surfaced:** `generate_response()` returns a dict with
`answer` and `sources`, where `sources` is the de-duplicated list of filenames
that supplied the context. The Gradio UI ([`app.py`](app.py)) renders these in a
dedicated **Sources** textbox beneath the answer.

---

## Evaluation Report

All five test questions from `planning.md`, run through the full pipeline.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Professor Abiodun Robert's teaching style? | Clear explanations with examples, friendly, good feedback; highly recommended. | "Lectures are clear; assignments/exams challenging but fair; an awesome instructor." Sources: abiodun_robert.txt, dave_harrity.txt | Relevant | Accurate |
| 2 | Is Professor Chris Bullock a good math professor? | Mixed — some say explains well, others say too fast / assumes prior knowledge. | Captures both sides: low ratings for over-complicating and fast pace, but notes 3.5/5 overall and 50% would retake. Sources: chris_bullock.txt, troy_young.txt | Relevant | Accurate |
| 3 | Does Professor Chryslee Hines offer extra credit? | Yes — multiple reviews mention extra credit; very helpful. | "Yes — offers extra credit described as 'amazing'; 'offers tons of extra credit'." Sources: chryslee_hines.txt | Relevant | Accurate |
| 4 | What is Professor Shane Green's rating and what do students say? | 1/5 rating; unresponsive, grades poorly, no effort. | "I don't have enough information on that." — shane_green.txt was **not retrieved**. Sources: abiodun_robert.txt, dave_harrity.txt, william_mcnear.txt | Off-target | Inaccurate |
| 5 | What do students say about Professor Dave Harrity's English class workload? | Heavy workload, lots of papers/reading, but good feedback and caring. | "Significant workload — 'so many papers,' lots of reading, draft/peer-review process, but many find it manageable." Sources: dave_harrity.txt | Relevant | Accurate |

**Score: 4/5 accurate.** Four questions produced grounded, correct answers with
the right sources. Q4 failed at the retrieval stage (analyzed below). Notably,
the failure manifested *safely* — the grounding prompt made the model refuse
rather than fabricate Shane Green's rating.

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "What is Professor Shane Green's rating and what do
students say?"

**What the system returned:** "I don't have enough information on that." The
retrieved sources were `abiodun_robert.txt`, `dave_harrity.txt`, and
`william_mcnear.txt` — `shane_green.txt` was never retrieved, so the model
correctly recognized it had no Shane Green content and refused.

**Root cause (retrieval stage, not generation):** This is a retrieval miss, not a
grounding or generation error. `shane_green.txt` is one of the shortest documents
(only **3 chunks**), and its single review is dominated by strongly negative,
generic complaint language ("unresponsive," "puts in 0 effort," "grades horribly")
rather than by the professor's name. The query embedding for *"rating and what do
students say"* is semantically diffuse — almost every review "says something" and
has a "rating" — so the cosine search spread across many files and the few
Shane Green chunks didn't rank in the top 4. This is precisely the
"off-topic retrieval across departments" risk anticipated in `planning.md`: when
the distinguishing signal (the professor's name) is sparse and the surrounding
language is common to many reviews, name-based queries can be crowded out.

**What I would change to fix it:** (1) Add metadata filtering — detect a professor
name in the query and pre-filter the ChromaDB query to that `source`, so a named
lookup can never drift to the wrong file. (2) Prepend the professor's name to
every chunk during ingestion so the name is present in every embedding, not just
the header chunk. (3) Increase top-k or add a name-match re-ranking pass. Option
(1) is the most direct fix for name-specific questions.

---

## Spec Reflection

**One way the spec helped me during implementation:** Writing the Chunking
Strategy and Retrieval Approach sections in `planning.md` *before* coding meant I
could hand each AI generation step a precise, self-contained instruction (300-char
chunks / 50-char overlap / top-k=4 / specific model name) instead of vague goals.
That specificity meant the generated `ingest.py` and `retriever.py` matched my
intent on the first pass and I spent my time verifying behavior rather than
re-explaining requirements. The Architecture diagram also kept the four modules
cleanly separated (ingest → retrieve → generate → UI), which made each milestone
independently testable.

**One way my implementation diverged from the spec, and why:** The plan assumed
chunks could be retrieved straightforwardly, but during evaluation I discovered the
spec's own anticipated challenge — cross-professor retrieval confusion — actually
caused a real failure (Q4, Shane Green). The plan treated source metadata as
sufficient to "help attribute answers correctly," but in practice metadata only
attributes what is *retrieved*; it does nothing when the right document never makes
the top-4. So my mental model diverged: I now see metadata-based *filtering* (not
just attribution) as the necessary fix, which wasn't in the original plan. I left
the in-memory ChromaDB index rather than persisting to disk, since the corpus is
small (79 chunks) and rebuilding on startup is fast — a deliberate simplification
over a production persistence layer.

---

## AI Usage

**Instance 1 — Ingestion and chunking (`ingest.py`)**

- *What I gave the AI:* My Chunking Strategy section from `planning.md` (300-char
  chunks, 50-char overlap, sliding window) and the Documents section (10 `.txt`
  files in `documents/`), asking it to load all files, strip extra whitespace, and
  chunk them, printing the total count.
- *What it produced:* A working `ingest.py` with `clean_text()`, a sliding-window
  `chunk_text()`, and a `main()` that printed per-file and total chunk counts.
- *What I changed or overrode:* The first version returned chunks as plain strings
  with no metadata. When I moved on to retrieval I needed source attribution, so I
  directed the AI to refactor it into a `get_chunks()` function that returns dicts
  with `text`, `source`, and `chunk_index` — keeping `main()`'s printed output
  identical. I also caught early that several document files were empty (0 bytes,
  producing 0 chunks) and fixed the data before trusting the chunk count.

**Instance 2 — Grounded generation (`generator.py`)**

- *What I gave the AI:* My grounding requirement — answer only from retrieved
  context, refuse otherwise, cite sources — plus the exact strict system prompt and
  the model name `llama-3.3-70b-versatile`, asking for a `generate_response(query,
  chunks)` returning `answer` and `sources`.
- *What it produced:* A `generator.py` that loads `GROQ_API_KEY` via
  `python-dotenv`, formats chunks into a numbered source-attributed context block,
  and calls the Groq client.
- *What I changed or overrode:* I directed it to set `temperature=0` (the initial
  draft left it at the default) for deterministic, non-creative answers, and to
  de-duplicate the `sources` list while preserving order so the UI wouldn't show
  the same filename four times. I verified grounding by running the Shane Green
  query and confirming the model refused rather than inventing a rating — which is
  exactly the behavior the strict prompt was meant to produce.

---

## Running the Project

```bash
pip install -r requirements.txt          # install dependencies
cp .env.example .env                      # then add your Groq API key
python ingest.py                          # verify chunk count (79)
python retriever.py                       # verify retrieval + distances
python app.py                             # launch the Gradio UI (http://127.0.0.1:7860)
```

Get a free Groq API key (no credit card) at <https://console.groq.com>.
