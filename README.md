# Employee Assistant (RAG Prototype)

A simple internal assistant that lets you ask natural language questions about employee data and get an AI generated answer, grounded in the actual records.

Built as a take home assessment. Focus was on clarity over complexity, as requested in the brief.

## How it works

1. Employee data (xlsx) is cleaned and converted into short text descriptions, one per employee
2. Those descriptions are embedded and stored in a local vector database (ChromaDB)
3. When a question comes in, it's embedded too, and the most similar employee records are retrieved
4. Those records are passed to a local LLM (Ollama, llama3), which generates an answer grounded in that context
5. A FastAPI endpoint wraps this whole flow

See `docs/architecture-diagram.png` and Part 1 of the report for the full picture.

## Tech stack and why

- **sentence-transformers (all-MiniLM-L6-v2)** for embeddings: small, fast on CPU, no API key needed
- **ChromaDB** for the vector store: simpler than FAISS for this scale, handles persistence and metadata out of the box
- **Ollama (llama3)** for answer generation: fully local, no API costs, no key management
- **FastAPI** for the API layer: automatic input validation and interactive docs

Note: torch is pinned to 2.2.2 in `requirements.txt` because this project was built on an Intel Mac, which lost support in newer PyTorch versions. `sentence-transformers` and `transformers` are pinned to matching compatible versions.

## Setup

1. Clone the repo and enter the folder
```bash
   git clone 
   cd nesma
```

2. Create and activate a virtual environment
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Make sure Ollama is installed and pull the model
```bash
   ollama pull llama3
```

5. Place the dataset at `data/raw/employee_np.xlsx` (already included in this repo)

## Running it

Run these in order from the project root, with the virtual environment active.

1. Process the raw data into text chunks
```bash
   python3 src/ingest.py
```

2. Generate embeddings and build the vector store
```bash
   python3 src/embed.py
```

3. (Optional) Test retrieval and answer generation directly from the terminal
```bash
   python3 src/query.py
```

4. Start the API
```bash
   uvicorn src.api:app --reload
```

5. Ask a question
```bash
   curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Who works in the HR department?"}'
```

   Or open `http://127.0.0.1:8000/docs` in a browser for an interactive interface.

## Project structure

```
nesma/
├── data/
│   ├── raw/              original xlsx, untouched
│   └── processed/        cleaned JSON chunks and vector store
├── src/
│   ├── ingest.py          Part 2: load, clean, convert rows to text
│   ├── embed.py            Part 3: generate embeddings, build vector store
│   ├── query.py            Part 3: retrieval and answer generation logic
│   ├── api.py                Part 4: FastAPI app
│   └── scratch/
│       └── retrieval_only.py   earlier draft, kept for reference
├── docs/
│   └── NP_AI_Engineer_Assessment.docx
├── requirements.txt
└── README.md
```

## Development notes

`scratch/retrieval_only.py` shows an earlier version of the retrieval step, before the LLM answer generation was added in `src/query.py`. Kept for reference to show the iterative process.

## Known limitations

Full discussion is in the report, but briefly: this uses pure semantic search, which is strong for meaning based questions but weak on exact name lookups and "list all" style queries. See Part 5 of the report for the full breakdown and possible improvements.
