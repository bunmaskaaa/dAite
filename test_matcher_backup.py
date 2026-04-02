import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(__file__))

from models.matcher import (
    load_users,
    generate_embeddings,
    build_faiss_index,
    find_matches,
    build_profile_text,
    compatibility_score
)

# ── Setup: load everything once for all tests ──────────────────────────────────
USERS_PATH = os.path.join(os.path.dirname(__file__), "data", "users.json")
users = load_users(USERS_PATH)
embeddings = generate_embeddings(users)
index = build_faiss_index(embeddings)

print("=" * 55)
print("  dAite Matching Engine — Test Suite")
print("=" * 55)

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ PASS — {name}")
        passed += 1
    else:
        print(f"  ❌ FAIL — {name}")
        if detail:
            print(f"           {detail}")
        failed += 1


# ── Test 1: Data loading ───────────────────────────────────────────────────────
print("\n📦 Data Loading")
test("Users loaded successfully", len(users) > 0)
test("Correct number of users", len(users) == 10,
     f"Expected 10, got {len(users)}")
test("Each user has required fields",
     all("id" in u and "bio" in u and "interests" in u and
         "personality" in u and "values" in u and
         "relationship_goal" in u for u in users))
test("User IDs are unique",
     len(set(u["id"] for u in users)) == len(users))


# ── Test 2: Embedding pipeline ─────────────────────────────────────────────────
print("\n🧠 Embedding Pipeline")
test("Embeddings shape is correct",
     embeddings.shape == (10, 384),
     f"Expected (10, 384), got {embeddings.shape}")
test("Embeddings are float32",
     embeddings.dtype == np.float32,
     f"Got dtype: {embeddings.dtype}")
test("No NaN values in embeddings",
     not np.isnan(embeddings).any())
test("Embeddings are non-zero",
     np.all(np.linalg.norm(embeddings, axis=1) > 0))


# ── Test 3: Profile text builder ──────────────────────────────────────────────
print("\n📝 Profile Text Builder")
sample = users[0]
profile = build_profile_text(sample)
test("Profile text contains interests",
     sample["interests"] in profile)
test("Profile text contains values",
     sample["values"] in profile)
test("Profile text contains bio",
     sample["bio"] in profile)
test("Profile text is non-empty",
     len(profile) > 50)


# ── Test 4: FAISS index ────────────────────────────────────────────────────────
print("\n🔍 FAISS Index")
test("Index contains all users",
     index.ntotal == 10,
     f"Expected 10 vectors, got {index.ntotal}")
test("Index dimension is correct",
     index.d == 384,
     f"Expected 384, got {index.d}")


# ── Test 5: Match quality ──────────────────────────────────────────────────────
print("\n💘 Match Quality")

# Alex (id=1) and Priya (id=2) are both introverted, thoughtful, long-term
alex_matches = find_matches(1, users, index, embeddings, top_k=3)
alex_match_ids = [m["id"] for m in alex_matches]
test("Alex gets 3 matches", len(alex_matches) == 3)
test("Alex is not matched with himself",
     1 not in alex_match_ids)
test("Alex's top match has high score",
     alex_matches[0]["compatibility_score"] >= 70,
     f"Top score was {alex_matches[0]['compatibility_score']}")

# Marcus (id=3) and Sofia (id=4) are both extroverted, energetic, casual
marcus_matches = find_matches(3, users, index, embeddings, top_k=3)
marcus_match_ids = [m["id"] for m in marcus_matches]
test("Marcus gets 3 matches", len(marcus_matches) == 3)
test("Sofia is Marcus's top match",
     marcus_match_ids[0] == 4,
     f"Expected id=4 (Sofia), got id={marcus_match_ids[0]}")

# James (id=5) and Aisha (id=6) are both analytical, philosophical
james_matches = find_matches(5, users, index, embeddings, top_k=3)
james_match_ids = [m["id"] for m in james_matches]
test("Aisha is James's top match",
     james_match_ids[0] == 6,
     f"Expected id=6 (Aisha), got id={james_match_ids[0]}")


# ── Test 6: Compatibility scoring ─────────────────────────────────────────────
print("\n📊 Compatibility Scoring")

user_a = {"relationship_goal": "long-term relationship"}
user_b = {"relationship_goal": "long-term relationship"}
user_c = {"relationship_goal": "casual dating"}

score_same = compatibility_score(user_a, user_b, 0.80)
score_diff = compatibility_score(user_a, user_c, 0.80)

test("Same relationship goal scores higher than different",
     score_same > score_diff,
     f"Same: {score_same}, Different: {score_diff}")
test("Score is capped at 100",
     compatibility_score(user_a, user_b, 1.0) <= 100)
test("Score is non-negative",
     compatibility_score(user_a, user_c, 0.0) >= 0)
test("Scores are sorted descending in results",
     all(alex_matches[i]["compatibility_score"] >=
         alex_matches[i+1]["compatibility_score"]
         for i in range(len(alex_matches)-1)))


# ── Test 7: Edge cases ─────────────────────────────────────────────────────────
print("\n⚠️  Edge Cases")
test("Invalid user ID returns empty list",
     find_matches(999, users, index, embeddings) == [])
test("top_k=1 returns exactly 1 match",
     len(find_matches(1, users, index, embeddings, top_k=1)) == 1)
test("top_k=5 returns exactly 5 matches",
     len(find_matches(1, users, index, embeddings, top_k=5)) == 5)


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  Results: {passed} passed, {failed} failed")
print("=" * 55)