"""
dAite Agentic Layer — v2
─────────────────────────
Structured onboarding flow: collects age, gender, location, preferences,
and dealbreakers before running semantic search + compatibility scoring.

Drop this file into: api/agent.py
"""

import os
import time
import json
import logging
from anthropic import Anthropic

logger = logging.getLogger("dAite.agent")

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_profiles",
        "description": (
            "Search the dAite profile database using a natural language description "
            "of what the user is looking for. Returns semantically similar profiles. "
            "Only call this once you have collected ALL of: the searcher's age, gender, "
            "location, preferences, and dealbreakers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Rich natural language description combining the user's ideal match traits, values, interests, lifestyle, and relationship goals"
                },
                "searcher_gender": {
                    "type": "string",
                    "description": "Gender of the person searching (male/female/non-binary/other)"
                },
                "searcher_age": {
                    "type": "integer",
                    "description": "Age of the person searching"
                },
                "searcher_location": {
                    "type": "string",
                    "description": "Location/city of the person searching"
                },
                "dealbreakers": {
                    "type": "string",
                    "description": "Dealbreakers stated by the user — used to filter or down-rank matches"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of profiles to return (default 3, max 5)",
                    "default": 3
                }
            },
            "required": ["query", "searcher_gender", "searcher_age", "searcher_location"]
        }
    },
    {
        "name": "compute_compatibility",
        "description": (
            "Compute a detailed compatibility score between the searcher's full profile "
            "and a specific candidate. Returns score 0-100. "
            "Call this for each profile returned by search_profiles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "profile_id": {
                    "type": "integer",
                    "description": "ID of the candidate profile to score"
                },
                "searcher_age": {
                    "type": "integer",
                    "description": "Age of the person searching"
                },
                "searcher_gender": {
                    "type": "string",
                    "description": "Gender of the person searching"
                },
                "searcher_location": {
                    "type": "string",
                    "description": "Location of the person searching"
                },
                "preferences": {
                    "type": "string",
                    "description": "Full description of what the searcher is looking for"
                },
                "dealbreakers": {
                    "type": "string",
                    "description": "Dealbreakers stated by the searcher"
                }
            },
            "required": ["profile_id", "searcher_age", "searcher_gender", "searcher_location", "preferences"]
        }
    },
    {
        "name": "get_anti_ghosting",
        "description": (
            "Run anti-ghosting analysis between the searcher and a candidate. "
            "Returns ghosting risk (Low/Medium/High), engagement score, green flags, "
            "and risk factors. Call this for the top 1-2 matches only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "profile_id": {
                    "type": "integer",
                    "description": "Profile ID of the candidate"
                },
                "searcher_goal": {
                    "type": "string",
                    "description": "Relationship goal of the searcher (long-term relationship / casual dating / friendship first)"
                },
                "compatibility_score": {
                    "type": "number",
                    "description": "Compatibility score already computed for this pair"
                }
            },
            "required": ["profile_id", "searcher_goal", "compatibility_score"]
        }
    }
]

SYSTEM_PROMPT = """You are the dAite matching agent — a warm, intelligent AI relationship compatibility assistant.

## Your goal
Help users find meaningful connections by first understanding who they are and what they want, then searching the dAite database for the best matches.

## Onboarding flow (ALWAYS follow this order)

### Step 1 — Greet and collect basics
Start by warmly greeting the user. Then ask in ONE message:
- Their name
- Age
- Gender
- Location (city is fine)

Wait for their response before continuing.

### Step 2 — Collect preferences
Ask in ONE message what they are looking for in a partner:
- Values and personality traits
- Interests and lifestyle
- Relationship goal (long-term, casual, friendship first)
- Age range preference

Wait for their response before continuing.

### Step 3 — Collect dealbreakers
Ask in ONE message:
- Any dealbreakers or must-haves
- Anything else important to them

Wait for their response.

### Step 4 — Run the tools (ONLY after collecting all the above)
Once you have name, age, gender, location, preferences, and dealbreakers:
1. Call search_profiles with a rich query combining everything
2. Call compute_compatibility for each returned profile (top 3)
3. Call get_anti_ghosting for the top 2 matches
4. Present results

## Presenting results
- Lead with the top match and explain specifically WHY they match
- Show compatibility score and ghosting risk for each
- Reference actual profile details (interests, values, goals)
- Be warm and specific — never generic
- Offer to refine the search if they want different results

## Rules
- Never invent profiles or scores — everything must come from the tools
- Never run tools until you have collected ALL required info (name, age, gender, location, preferences, dealbreakers)
- If the user skips a question, make a reasonable assumption and state it
- Keep each question message concise and conversational"""


# ── Tool execution ────────────────────────────────────────────────────────────

def execute_tool(name: str, input_data: dict, users: list, embeddings, index, model) -> dict:
    import faiss
    import numpy as np
    from models.matcher import (
        build_profile_text,
        compatibility_score,
        anti_ghosting_score
    )

    if name == "search_profiles":
        query = input_data["query"]
        top_k = min(input_data.get("top_k", 3), 5)
        searcher_gender = input_data.get("searcher_gender", "").lower()
        searcher_age = input_data.get("searcher_age", 0)
        dealbreakers = input_data.get("dealbreakers", "").lower()

        # Embed query
        query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)

        # Search full index first (get more candidates for filtering)
        distances, indices = index.search(query_emb, min(top_k * 3, len(users)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            u = users[idx]

            # Basic dealbreaker filtering (keyword match on profile text)
            if dealbreakers:
                profile_text = (u.get("interests", "") + " " + u.get("personality", "") + " " + u.get("bio", "")).lower()
                flagged = any(db.strip() in profile_text for db in dealbreakers.split(",") if len(db.strip()) > 2)
                if flagged:
                    continue

            results.append({
                "id": u["id"],
                "name": u["name"],
                "age": u["age"],
                "gender": u["gender"],
                "relationship_goal": u["relationship_goal"],
                "interests": u["interests"],
                "personality": u["personality"],
                "values": u["values"],
                "bio": u["bio"],
                "semantic_similarity": round(float(dist) * 100, 1)
            })

            if len(results) >= top_k:
                break

        logger.info(f"search_profiles: returned {len(results)} results for age={searcher_age} gender={searcher_gender}")
        return {"profiles": results, "total": len(results)}

    elif name == "compute_compatibility":
        profile_id = input_data["profile_id"]
        preferences = input_data.get("preferences", "")
        searcher_age = input_data.get("searcher_age", 0)
        searcher_gender = input_data.get("searcher_gender", "")
        searcher_location = input_data.get("searcher_location", "")
        dealbreakers = input_data.get("dealbreakers", "").lower()

        profile = next((u for u in users if u["id"] == profile_id), None)
        if not profile:
            return {"error": f"Profile {profile_id} not found"}

        # Semantic similarity between preferences and profile
        pref_emb = model.encode([preferences], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(pref_emb)

        profile_text = build_profile_text(profile)
        profile_emb = model.encode([profile_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(profile_emb)

        cosine_sim = float(np.dot(pref_emb[0], profile_emb[0]))

        # Build searcher dict for compatibility_score
        searcher = {
            "relationship_goal": _extract_goal(preferences),
            "personality": "",
            "values": ""
        }
        base_score = compatibility_score(searcher, profile, cosine_sim)

        # Age proximity bonus (within 5 years = small bonus)
        age_diff = abs(searcher_age - profile.get("age", searcher_age))
        age_bonus = max(0, 5 - age_diff) * 0.5  # max +2.5 pts

        # Dealbreaker penalty
        db_penalty = 0
        if dealbreakers:
            profile_full = (profile.get("interests", "") + " " + profile.get("personality", "") + " " + profile.get("bio", "")).lower()
            hits = sum(1 for db in dealbreakers.split(",") if db.strip() in profile_full)
            db_penalty = hits * 10

        final_score = round(min(100, max(0, base_score + age_bonus - db_penalty)), 2)

        logger.info(f"compute_compatibility: profile_id={profile_id} score={final_score}")
        return {
            "profile_id": profile_id,
            "name": profile["name"],
            "age": profile["age"],
            "location": searcher_location,
            "compatibility_score": final_score,
            "cosine_similarity": round(cosine_sim * 100, 1),
            "age_difference": age_diff,
            "goal_match": searcher["relationship_goal"] == profile["relationship_goal"]
        }

    elif name == "get_anti_ghosting":
        profile_id = input_data["profile_id"]
        searcher_goal = input_data["searcher_goal"]
        comp_score = input_data["compatibility_score"]

        profile = next((u for u in users if u["id"] == profile_id), None)
        if not profile:
            return {"error": f"Profile {profile_id} not found"}

        searcher = {"relationship_goal": searcher_goal, "personality": "", "values": ""}
        result = anti_ghosting_score(searcher, profile, comp_score)

        logger.info(f"get_anti_ghosting: profile_id={profile_id} risk={result['ghosting_risk']}")
        return {"profile_id": profile_id, "name": profile["name"], **result}

    return {"error": f"Unknown tool: {name}"}


def _extract_goal(text: str) -> str:
    text_lower = text.lower()
    if "casual" in text_lower:
        return "casual dating"
    if "friendship" in text_lower or "friends first" in text_lower:
        return "friendship first"
    return "long-term relationship"


# ── Main agent loop ───────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    conversation_history: list,
    users: list,
    embeddings,
    index,
    model
) -> dict:
    """
    Runs one turn of the Claude tool-calling agent loop.

    Returns:
        {
            "response": str,
            "tool_calls": list,
            "tool_results": list,
            "latency_ms": int,
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float
        }
    """
    client = Anthropic()
    start_time = time.time()

    messages = list(conversation_history)
    messages.append({"role": "user", "content": user_message})

    tool_calls_log = []
    tool_results_log = []
    total_input_tokens = 0
    total_output_tokens = 0
    final_text = ""

    max_iterations = 8
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        text_blocks = [b for b in response.content if b.type == "text"]
        if text_blocks:
            final_text = text_blocks[0].text

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_use_blocks:
            tool_name = tool_use.name
            tool_input = tool_use.input

            tool_calls_log.append({"name": tool_name, "input": tool_input})
            logger.info(f"Tool call: {tool_name}({json.dumps(tool_input)[:120]})")

            result = execute_tool(tool_name, tool_input, users, embeddings, index, model)
            tool_results_log.append({"name": tool_name, "result": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result)
            })

        messages.append({"role": "user", "content": tool_results})

    latency_ms = int((time.time() - start_time) * 1000)

    # claude-sonnet-4 pricing: $3/M input, $15/M output
    cost_usd = (total_input_tokens / 1_000_000 * 3.0) + (total_output_tokens / 1_000_000 * 15.0)

    logger.info(
        f"Agent turn | iterations={iteration} | "
        f"tokens={total_input_tokens + total_output_tokens} | "
        f"latency={latency_ms}ms | cost=${cost_usd:.5f}"
    )

    return {
        "response": final_text,
        "tool_calls": tool_calls_log,
        "tool_results": tool_results_log,
        "latency_ms": latency_ms,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost_usd": round(cost_usd, 6)
    }