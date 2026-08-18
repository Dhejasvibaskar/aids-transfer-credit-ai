# AIDS Transfer Credit Evaluation System — AI Core (Person B)

AI / matching engine half of the AIDS Department Transfer Credit Evaluation
System. Build window: Aug 8 – Aug 18, 2026.

This repo covers the modules owned by Person B: embeddings, the
topic-pooling matching engine (with cross-course allocation), decision
threshold logic, the Gemini-based reasoning generator, and the pipeline
that chains them together end to end.

## Setup

```bash
pip install -e . --break-system-packages
```

This installs the project in editable mode so everything under `src/` is
importable as a top-level package (`decision_engine`, `matching_engine`,
`reasoning_generator`, `evaluation_pipeline`, `embeddings`) regardless of
your working directory. Requires Python 3.11+.

## Structure

```
src/
├── decision_engine.py       # coverage % -> accept / partial / redo (+ red/yellow/green)
├── matching_engine.py       # match_course() + allocate_across_courses()
├── reasoning_generator.py   # Gemini-based structured reasoning, with rule-based fallback
├── evaluation_pipeline.py   # orchestrates the above end to end
└── embeddings/
    ├── embed_topics.py            # shared core: text -> 384-dim embedding
    ├── embed_rec_syllabus.py      # one-time: embeds REC's permanent target syllabus
    └── embed_source_syllabus.py   # per-evaluation: embeds a student's submitted topics
tests/
    (mirrors the above, one test file per module)
```

## Files

| File | Purpose | Status |
| --- | --- | --- |
| `embeddings/embed_topics.py` | Text -> 384-dim embedding (shared core) | Aug 8 ✅ |
| `embeddings/embed_rec_syllabus.py` | Embed REC's permanent target syllabus | Aug 8 ✅ |
| `embeddings/embed_source_syllabus.py` | Embed a student's submitted topics per evaluation | Aug 8 ✅ |
| `matching_engine.py` | `match_course()` — cosine similarity + coverage % | Aug 9 ✅ |
| `decision_engine.py` | Coverage % -> accept / partial / redo, + red/yellow/green | Aug 10 ✅ |
| `reasoning_generator.py` | Gemini-based structured reasoning | Aug 11 ✅ |
| `matching_engine.py` (`allocate_across_courses`) | Greedy cross-course allocation — prevents double-counting | Aug 12 ✅ |
| `evaluation_pipeline.py` | Full embed→match→allocate→decide→reason pipeline | Aug 13 ✅ |

## Running tests

```bash
pytest tests/ -v
```

87 tests, all offline/deterministic (embedding model calls are mocked in
tests via hand-crafted vectors — no network or GPU needed to run the suite).

## Running the live demo (real model, needs internet)

```bash
python3 src/evaluation_pipeline.py
```

Runs 3 mock transcripts through the real `sentence-transformers` model,
covering full match, empty transcript, and single-topic-course edge cases.

## Gemini API

`reasoning_generator.py` falls back to a rule-based narrative automatically
if `GEMINI_API_KEY` isn't set. To use the real API:

```bash
export GEMINI_API_KEY="your-key-here"
```
