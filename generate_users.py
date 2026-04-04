import json
import random

# ── Data pools to mix and match ────────────────────────────────────────────────
first_names_male = [
    "Alex", "Marcus", "James", "Ryan", "David", "Ethan", "Noah", "Liam",
    "Omar", "Carlos", "Raj", "Kevin", "Daniel", "Chris", "Andre", "Miguel"
]
first_names_female = [
    "Priya", "Sofia", "Aisha", "Nina", "Leila", "Maya", "Sara", "Emma",
    "Zoe", "Fatima", "Isabel", "Hannah", "Chloe", "Nadia", "Yuki", "Amara"
]
last_names = [
    "Johnson", "Sharma", "Williams", "Reyes", "Chen", "Patel", "Park",
    "Gupta", "Kim", "Hassan", "Martinez", "Nguyen", "Ali", "Brown",
    "Garcia", "Wilson", "Anderson", "Thomas", "Jackson", "White"
]

interests_pool = [
    "hiking", "photography", "cooking", "travel", "reading", "yoga",
    "art galleries", "basketball", "music production", "gaming", "fitness",
    "chess", "podcasts", "philosophy", "cycling", "writing", "documentaries",
    "anime", "board games", "surfing", "coffee", "dancing", "meditation",
    "rock climbing", "painting", "running", "movies", "volunteering",
    "tech", "cricket", "football", "tennis", "swimming", "gardening"
]

personality_pool = [
    "introverted", "extroverted", "analytical", "creative", "adventurous",
    "empathetic", "humorous", "calm", "energetic", "thoughtful",
    "spontaneous", "ambitious", "loyal", "curious", "optimistic",
    "independent", "warm", "confident", "patient", "passionate"
]

values_pool = [
    "honesty", "adventure", "personal growth", "family", "loyalty",
    "ambition", "fun", "friendship", "intellectual growth", "independence",
    "depth", "creativity", "authenticity", "freedom", "experiences",
    "positivity", "mindfulness", "compassion", "integrity", "balance"
]

relationship_goals = [
    "long-term relationship", "long-term relationship", "long-term relationship",
    "casual dating", "casual dating", "friendship first"
]

bio_templates = [
    "Lover of {i1} and {i2}. I'm {p1} by nature but always up for new experiences. Looking for someone who values {v1} and {v2}.",
    "You'll find me {i1} on weekends and {i2} on weeknights. {p1} and {p2} describe me best. Seeking genuine connection.",
    "I believe life is about {v1} and {v2}. Passionate about {i1} and {i2}. Let's see where things go.",
    "A {p1} soul who loves {i1}. Also really into {i2}. {v1} is non-negotiable for me.",
    "When I'm not {i1}, you'll find me exploring {i2}. I'm {p1} and {p2}, and I care deeply about {v1}.",
    "Big on {v1} and {v2}. My hobbies include {i1} and {i2}. I'm {p1} but love good conversation.",
    "{p1} and {p2} with a passion for {i1}. Also love {i2} in my downtime. Values: {v1} above all.",
    "I spend my time between {i1} and {i2}. Friends describe me as {p1} and {p2}. Looking for {v1}.",
]


def generate_user(user_id: int) -> dict:
    gender = random.choice(["male", "female"])
    if gender == "male":
        name = f"{random.choice(first_names_male)} {random.choice(last_names)}"
    else:
        name = f"{random.choice(first_names_female)} {random.choice(last_names)}"

    age = random.randint(22, 35)

    # Pick random interests, personality traits, values
    interests = random.sample(interests_pool, 4)
    personality = random.sample(personality_pool, 4)
    values = random.sample(values_pool, 4)
    goal = random.choice(relationship_goals)

    # Fill in a bio template
    template = random.choice(bio_templates)
    bio = template.format(
        i1=interests[0], i2=interests[1],
        p1=personality[0], p2=personality[1],
        v1=values[0], v2=values[1]
    )

    return {
        "id": user_id,
        "name": name,
        "age": age,
        "gender": gender,
        "interests": ", ".join(interests),
        "personality": ", ".join(personality),
        "values": ", ".join(values),
        "relationship_goal": goal,
        "bio": bio
    }


def generate_dataset(n: int = 50, output_path: str = "data/users.json"):
    users = [generate_user(i + 1) for i in range(n)]
    with open(output_path, "w") as f:
        json.dump(users, f, indent=2)
    print(f"Generated {n} users → {output_path}")

    # Print a summary
    goals = [u["relationship_goal"] for u in users]
    genders = [u["gender"] for u in users]
    print(f"  Long-term: {goals.count('long-term relationship')}")
    print(f"  Casual: {goals.count('casual dating')}")
    print(f"  Friendship first: {goals.count('friendship first')}")
    print(f"  Male: {genders.count('male')}, Female: {genders.count('female')}")


if __name__ == "__main__":
    generate_dataset(50)