import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))

from models.matcher import (
    load_users,
    generate_embeddings,
    build_faiss_index,
    find_matches,
    compatibility_score
)

print("=" * 60)
print("  dAite — Matching Engine Evaluation Report")
print("=" * 60)

# ── Setup ──────────────────────────────────────────────────────
USERS_PATH = os.path.join(os.path.dirname(__file__), "data", "users.json")
users = load_users(USERS_PATH)
embeddings = generate_embeddings(users)
index = build_faiss_index(embeddings)

# ── 1. Average compatibility score across all users ────────────
print("\n📊 1. Average Compatibility Scores")
all_scores = []
for user in users:
    matches = find_matches(user["id"], users, index, embeddings, top_k=3)
    scores = [m["compatibility_score"] for m in matches]
    all_scores.extend(scores)

print(f"   Mean compatibility score:   {np.mean(all_scores):.2f}")
print(f"   Median compatibility score: {np.median(all_scores):.2f}")
print(f"   Std deviation:              {np.std(all_scores):.2f}")
print(f"   Min score:                  {np.min(all_scores):.2f}")
print(f"   Max score:                  {np.max(all_scores):.2f}")

# ── 2. Relationship goal alignment rate ────────────────────────
print("\n🎯 2. Relationship Goal Alignment")
aligned = 0
total_matches = 0
for user in users:
    matches = find_matches(user["id"], users, index, embeddings, top_k=3)
    for m in matches:
        matched_user = next(u for u in users if u["id"] == m["id"])
        if user["relationship_goal"] == matched_user["relationship_goal"]:
            aligned += 1
        total_matches += 1

alignment_rate = (aligned / total_matches) * 100
print(f"   Matches with same goal:     {aligned}/{total_matches}")
print(f"   Goal alignment rate:        {alignment_rate:.1f}%")

# ── 3. Embedding space diversity ───────────────────────────────
print("\n🧠 3. Embedding Space Diversity")
sample = min(30, len(users))
sample_embeddings = embeddings[:sample]
similarities = []
for i in range(sample):
    for j in range(i + 1, sample):
        a = sample_embeddings[i]
        b = sample_embeddings[j]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        similarities.append(sim)

print(f"   Avg pairwise similarity:    {np.mean(similarities):.4f}")
print(f"   Similarity std deviation:   {np.std(similarities):.4f}")
print(f"   Min pairwise similarity:    {np.min(similarities):.4f}")
print(f"   Max pairwise similarity:    {np.max(similarities):.4f}")
diversity = (1 - np.mean(similarities)) * 100
print(f"   Embedding diversity score:  {diversity:.1f}%")

# ── 4. Score distribution analysis ────────────────────────────
print("\n📈 4. Score Distribution")
buckets = defaultdict(int)
for score in all_scores:
    if score >= 90:
        buckets["90-100 (Excellent)"] += 1
    elif score >= 80:
        buckets["80-89 (Great)"] += 1
    elif score >= 70:
        buckets["70-79 (Good)"] += 1
    elif score >= 60:
        buckets["60-69 (Fair)"] += 1
    else:
        buckets["<60 (Low)"] += 1

for label in ["90-100 (Excellent)", "80-89 (Great)", "70-79 (Good)",
              "60-69 (Fair)", "<60 (Low)"]:
    count = buckets[label]
    bar = "█" * (count * 2)
    print(f"   {label}: {bar} {count}")

# ── 5. Per relationship goal analysis ─────────────────────────
print("\n💘 5. Match Quality by Relationship Goal")
goal_scores = defaultdict(list)
for user in users:
    matches = find_matches(user["id"], users, index, embeddings, top_k=3)
    for m in matches:
        goal_scores[user["relationship_goal"]].append(m["compatibility_score"])

for goal, scores in sorted(goal_scores.items()):
    print(f"   {goal}:")
    print(f"     Avg score: {np.mean(scores):.2f} | "
          f"Users: {len([u for u in users if u['relationship_goal'] == goal])}")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Evaluation Complete")
print(f"  Total users evaluated: {len(users)}")
print(f"  Total matches analyzed: {total_matches}")
print(f"  Overall avg score: {np.mean(all_scores):.2f}")
print(f"  Goal alignment rate: {alignment_rate:.1f}%")
print("=" * 60)