from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# ── 1. Load the model ──────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is a lightweight but powerful sentence embedding model
# It converts text into 384-dimensional vectors
model = SentenceTransformer("all-MiniLM-L6-v2")


# ── 2. Build a text profile from a user dict ───────────────────────────────────
# We combine all meaningful fields into one rich text string
# This is what gets embedded into a vector
def build_profile_text(user: dict) -> str:
    return (
        f"Interests: {user['interests']}. "
        f"Personality: {user['personality']}. "
        f"Values: {user['values']}. "
        f"Relationship goal: {user['relationship_goal']}. "
        f"Bio: {user['bio']}"
    )


# ── 3. Load users from JSON ────────────────────────────────────────────────────
def load_users(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


# ── 4. Generate embeddings for all users ───────────────────────────────────────
# Each user profile becomes a 384-dim float32 vector
def generate_embeddings(users: list) -> np.ndarray:
    profiles = [build_profile_text(u) for u in users]
    embeddings = model.encode(profiles, convert_to_numpy=True)
    return embeddings.astype("float32")


# ── 5. Build a FAISS index ─────────────────────────────────────────────────────
# FAISS stores all vectors and lets us search them by similarity instantly
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # Normalize vectors so dot product == cosine similarity
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


# ── 6. Compute compatibility score ────────────────────────────────────────────
# Combines cosine similarity with a relationship goal bonus
def compatibility_score(user_a: dict, user_b: dict, cosine_sim: float) -> float:
    score = cosine_sim * 100  # base score out of 100

    # Bonus: same relationship goal is a strong compatibility signal
    if user_a["relationship_goal"] == user_b["relationship_goal"]:
        score += 10

    # Cap at 100
    return round(min(score, 100), 2)


# ── 7. Find top matches for a given user ──────────────────────────────────────
def find_matches(user_id: int, users: list, index: faiss.IndexFlatIP,
                 embeddings: np.ndarray, top_k: int = 3) -> list:

    # Find the user
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        return []

    # Get this user's embedding and search the index
    user_idx = users.index(user)
    query = embeddings[user_idx].reshape(1, -1).copy()
    faiss.normalize_L2(query)

    # Search for top_k + 1 to exclude the user themselves
    distances, indices = index.search(query, top_k + 1)

    matches = []
    for dist, idx in zip(distances[0], indices[0]):
        matched_user = users[idx]

        # Skip the user themselves
        if matched_user["id"] == user_id:
            continue

        score = compatibility_score(user, matched_user, float(dist))
        matches.append({
            "id": matched_user["id"],
            "name": matched_user["name"],
            "age": matched_user["age"],
            "relationship_goal": matched_user["relationship_goal"],
            "bio": matched_user["bio"],
            "compatibility_score": score
        })

    # Sort by score descending
    matches.sort(key=lambda x: x["compatibility_score"], reverse=True)
    return matches[:top_k]


# ── 8. Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    users_path = os.path.join(base_dir, "data", "users.json")

    print("Loading users...")
    users = load_users(users_path)

    print("Generating embeddings...")
    embeddings = generate_embeddings(users)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("\n--- Testing matcher for Alex Johnson (id=1) ---")
    matches = find_matches(1, users, index, embeddings)
    for m in matches:
        print(f"  {m['name']} — Score: {m['compatibility_score']}")

    print("\n--- Testing matcher for Marcus Williams (id=3) ---")
    matches = find_matches(3, users, index, embeddings)
    for m in matches:
        print(f"  {m['name']} — Score: {m['compatibility_score']}")