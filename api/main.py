from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sys
import os

# Add parent directory to path so we can import from models/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.matcher import (
    load_users,
    generate_embeddings,
    build_faiss_index,
    find_matches,
    build_profile_text
)
from sentence_transformers import SentenceTransformer
import faiss

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="dAite API",
    description="Trust-first AI dating compatibility engine",
    version="1.0.0"
)

# ── Load everything once at startup ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
USERS_PATH = os.path.join(BASE_DIR, "data", "users.json")

print("Loading users...")
users = load_users(USERS_PATH)

print("Generating embeddings...")
embeddings = generate_embeddings(users)

print("Building FAISS index...")
index = build_faiss_index(embeddings)

model = SentenceTransformer("all-MiniLM-L6-v2")
print("dAite API ready!")


# ── Request/Response models ────────────────────────────────────────────────────
class NewUser(BaseModel):
    name: str
    age: int
    gender: str
    interests: str
    personality: str
    values: str
    relationship_goal: str
    bio: str


class MatchResult(BaseModel):
    id: int
    name: str
    age: int
    relationship_goal: str
    bio: str
    compatibility_score: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "dAite",
        "tagline": "Trust-first AI dating",
        "version": "1.0.0",
        "endpoints": ["/users", "/match/{user_id}", "/match/new"]
    }


@app.get("/users")
def get_users():
    """Return all users in the system."""
    return {
        "total": len(users),
        "users": [
            {
                "id": u["id"],
                "name": u["name"],
                "age": u["age"],
                "gender": u["gender"],
                "relationship_goal": u["relationship_goal"]
            }
            for u in users
        ]
    }


@app.get("/users/{user_id}")
def get_user(user_id: int):
    """Return a single user by ID."""
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


@app.get("/match/{user_id}", response_model=list[MatchResult])
def get_matches(user_id: int, top_k: int = 3):
    """Get top matches for an existing user by ID."""
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    matches = find_matches(user_id, users, index, embeddings, top_k=top_k)
    if not matches:
        raise HTTPException(status_code=404, detail="No matches found")

    return matches
@app.get("/stats")
def get_stats():
    """Returns dataset statistics and embedding space insights."""
    from collections import Counter
    import numpy as np

    goals = Counter(u["relationship_goal"] for u in users)
    genders = Counter(u["gender"] for u in users)
    ages = [u["age"] for u in users]

    # Compute average pairwise similarity across all users
    # This tells us how "spread out" the embedding space is
    sample_size = min(20, len(users))
    sample_embeddings = embeddings[:sample_size]
    
    similarities = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            a = sample_embeddings[i]
            b = sample_embeddings[j]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            similarities.append(sim)

    return {
        "total_users": len(users),
        "gender_distribution": dict(genders),
        "relationship_goals": dict(goals),
        "age_stats": {
            "min": min(ages),
            "max": max(ages),
            "average": round(sum(ages) / len(ages), 1)
        },
        "embedding_space": {
            "dimensions": int(embeddings.shape[1]),
            "avg_pairwise_similarity": round(float(np.mean(similarities)), 4),
            "similarity_std": round(float(np.std(similarities)), 4),
            "note": "Lower avg similarity = more diverse user embeddings"
        }
    }
@app.get("/match/{user_id}/ghosting")
def get_ghosting_analysis(user_id: int, top_k: int = 3):
    """Returns matches with anti-ghosting analysis for each."""
    from models.matcher import anti_ghosting_score

    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    matches = find_matches(user_id, users, index, embeddings, top_k=top_k)

    results = []
    for match in matches:
        matched_user = next(u for u in users if u["id"] == match["id"])
        ghosting = anti_ghosting_score(user, matched_user, match["compatibility_score"])
        results.append({
            **match,
            "anti_ghosting": ghosting
        })

    return results

@app.post("/match/new", response_model=list[MatchResult])
def match_new_user(new_user: NewUser, top_k: int = 3):
    """
    Match a brand new user (not in the database) against all existing users.
    This is the core dAite flow — embed their profile on the fly and find matches.
    """
    # Build profile text and embed it
    user_dict = new_user.model_dump()
    profile_text = build_profile_text(user_dict)
    query_embedding = model.encode([profile_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Search the index
    distances, indices = index.search(query_embedding, top_k)

    matches = []
    for dist, idx in zip(distances[0], indices[0]):
        matched_user = users[idx]
        from models.matcher import compatibility_score
        score = compatibility_score(user_dict, matched_user, float(dist))
        matches.append(MatchResult(
            id=matched_user["id"],
            name=matched_user["name"],
            age=matched_user["age"],
            relationship_goal=matched_user["relationship_goal"],
            bio=matched_user["bio"],
            compatibility_score=score
        ))

    matches.sort(key=lambda x: x.compatibility_score, reverse=True)
    return matches