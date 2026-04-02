# dAite 💘

> Trust-first, AI-powered dating compatibility engine

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
6. Results are served via a **FastAPI REST API**

---

## ML Pipeline
```
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
Compatibility Scoring
      ↓
Ranked Match Results
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| API | FastAPI |
| Server | Uvicorn |
| Data | NumPy, Pandas |

---

## Project Structure
```
dAite/
├── data/
│   └── users.json        # Synthetic user dataset
├── models/
│   └── matcher.py        # Embedding pipeline + FAISS matching
├── api/
│   └── main.py           # FastAPI REST API
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/users` | List all users |
| GET | `/users/{id}` | Get a single user |
| GET | `/match/{user_id}` | Get top matches for an existing user |
| POST | `/match/new` | Match a brand new user against the database |

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
- Anti-ghosting nudge system
- AI-generated message suggestions (clearly labeled)
- Frontend demo interface

---
