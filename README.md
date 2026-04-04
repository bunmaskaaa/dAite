dAite 💘

> Trust-first, AI-powered dating compatibility engine

🚀 **Live API:** https://daite-production.up.railway.app/docs

dAite matches users based on **personality, values, and relationship goals** — not photos.
It uses NLP embeddings and vector similarity search to find genuinely compatible people,
reducing choice overload and encouraging meaningful connections.

---

## How It Works

1. **User profiles** are built from questionnaire responses (interests, personality, values, bio)
2. Each profile is encoded into a **384-dimensional vector** using Sentence Transformers (`all-MiniLM-L6-v2`)
3. Vectors are stored in a **FAISS index** for efficient nearest-neighbor search
4. **Cosine similarity** finds the closest matches
5. A **compatibility scoring system** layers in relationship goal alignment on top of the base similarity score
6. An **anti-ghosting engine** analyzes green flags and risk factors for each match
7. Results are served via a **FastAPI REST API** with request logging and input validation

---

## ML Pipeline
User Profile Text
↓
Sentence Transformer (all-MiniLM-L6-v2)
↓
384-dim Vector Embedding
↓
FAISS Index (cosine similarity)
↓
Top-K Nearest Neighbors
↓
Compatibility Scoring + Anti-Ghosting Analysis
↓
Ranked Match Results

---

## Evaluation Results

Run `python3 evaluate.py` to generate a full report. Latest results on 50 users:

| Metric | Value |
|---|---|
| Mean compatibility score | 85.04 |
| Goal alignment rate | 42.7% |
| Embedding diversity score | 31.5% |
| Matches below 60 score | 0 |
| Total matches analyzed | 150 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| API | FastAPI |
| Server | Uvicorn |
| Deployment | Railway |
| Data | NumPy, Pandas |

---

## Project Structure
dAite/
├── data/
│   └── users.json              # 50 synthetic users
├── models/
│   └── matcher.py              # Embedding pipeline + FAISS matching + anti-ghosting
├── api/
│   └── main.py                 # FastAPI REST API with logging + validation
├── generate_users.py           # Synthetic dataset generator
├── evaluate.py                 # Matching engine evaluation script
├── test_matcher.py             # 27-test suite
├── requirements.txt
└── README.md

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/users` | List all users |
| GET | `/users/{id}` | Get a single user |
| GET | `/match/{user_id}` | Top matches for an existing user |
| GET | `/match/{user_id}/ghosting` | Matches with anti-ghosting analysis |
| POST | `/match/new` | Match a brand new user against the database |
| POST | `/similar` | Find similar users from raw text |
| GET | `/stats` | Dataset + embedding space analytics |
| GET | `/health` | Health check |

---

## Quickstart
```bash
# Clone the repo
git clone https://github.com/bunmaskaaa/dAite.git
cd dAite

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## Run Tests
```bash
python3 test_matcher.py
```
Results: 27 passed, 0 failed

## Run Evaluation
```bash
python3 evaluate.py
```

---

## Sample Match Result
```json
POST /match/new
{
  "name": "Hardik",
  "age": 24,
  "interests": "machine learning, coding, cricket",
  "personality": "analytical, curious, introverted",
  "values": "ambition, honesty, personal growth",
  "relationship_goal": "long-term relationship",
  "bio": "ML engineer building cool things."
}

Response:
[
  { "name": "Ryan Park", "compatibility_score": 75.91 },
  { "name": "Marcus Williams", "compatibility_score": 60.63 },
  { "name": "Leila Hassan", "compatibility_score": 56.80 }
]
```

---

## Planned Features

- pgvector integration for persistent vector storage
- AI-generated message suggestions (clearly labeled)
- Frontend demo interface

---

*Built as part of an ML engineering portfolio project.*