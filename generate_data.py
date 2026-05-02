"""
Generates a realistic dummy dataset for Fake News Detection.
Produces data/dataset.csv with 'text' and 'label' columns (0=Real, 1=Fake).
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# --- Real news sentence templates ---
REAL_TEMPLATES = [
    "The {org} announced {action} following {event} in {place}.",
    "According to official reports, {official} stated that {policy} would be implemented by {year}.",
    "Scientists at {university} have published findings on {topic} in the {journal} journal.",
    "The {country} government approved a {amount} billion dollar budget for {sector}.",
    "Market analysts say {company} shares rose {pct}% after strong {quarter} earnings.",
    "Health authorities confirmed {number} new cases of {disease} in {region}.",
    "The {committee} passed legislation to regulate {industry} following {event}.",
    "An investigation by {org} found no evidence of {claim} in the recent {event}.",
    "World leaders gathered in {place} to discuss {topic} at the annual {conference}.",
    "The central bank raised interest rates by {pct}% to combat rising inflation.",
]

REAL_SLOTS = {
    "org": ["WHO", "United Nations", "Reuters", "AP", "the Federal Reserve", "NASA", "the Pentagon"],
    "action": ["new guidelines", "emergency measures", "a formal inquiry", "financial aid"],
    "event": ["rising tensions", "the economic downturn", "last month's summit", "the regional crisis"],
    "place": ["Washington D.C.", "Brussels", "Geneva", "Tokyo", "London", "New Delhi"],
    "official": ["the Secretary of State", "the Prime Minister", "the Director General", "the spokesperson"],
    "policy": ["the new trade agreement", "stricter emissions standards", "expanded healthcare access"],
    "year": ["2024", "2025", "next fiscal year"],
    "university": ["MIT", "Oxford", "Stanford", "Johns Hopkins", "Cambridge"],
    "topic": ["climate change", "genomic research", "AI safety", "vaccine efficacy", "urban planning"],
    "journal": ["Nature", "The Lancet", "Science", "NEJM"],
    "country": ["The US", "Germany", "India", "Japan", "Canada", "Australia"],
    "amount": ["1.2", "5", "20", "0.8", "3.5"],
    "sector": ["healthcare", "infrastructure", "renewable energy", "defense", "education"],
    "company": ["Apple", "Toyota", "Samsung", "BP", "Goldman Sachs"],
    "pct": ["2.5", "0.75", "4", "1.2", "3"],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "number": ["340", "1,200", "78", "4,500"],
    "disease": ["influenza A", "dengue fever", "COVID-19 subvariant", "measles"],
    "region": ["Southeast Asia", "Western Europe", "sub-Saharan Africa", "South America"],
    "committee": ["Senate", "Parliament", "the regulatory body", "the oversight committee"],
    "industry": ["social media platforms", "pharmaceutical companies", "financial institutions"],
    "conference": ["G20 Summit", "Climate Conference", "Security Forum", "Economic Forum"],
    "claim": ["fraud", "misconduct", "data manipulation", "bias"],
    "journal": ["Nature", "Science", "The Lancet"],
}

# --- Fake news sentence templates ---
FAKE_TEMPLATES = [
    "SHOCKING: {celebrity} EXPOSED for secretly {action} — the media won't tell you!",
    "BREAKING: Scientists BANNED from revealing the truth about {topic}. Share before it's deleted!",
    "You won't BELIEVE what {authority} is hiding about {product}. The cover-up is REAL.",
    "This {product} CURES {disease} in {days} days — Big Pharma doesn't want you to know!",
    "{celebrity} CONFIRMS {conspiracy} in leaked video that mainstream media is ignoring.",
    "Government CAUGHT {action} to control the population — whistleblower reveals ALL.",
    "The REAL reason {event} happened will make your blood boil. They lied to us again.",
    "URGENT: Forward this to everyone — {authority} is planning {conspiracy} by {year}.",
    "ALIEN structures discovered on {place}. NASA is covering it up. See the proof here!",
    "One weird trick {authority} doesn't want you to know about {topic}. It's being SUPPRESSED.",
]

FAKE_SLOTS = {
    "celebrity": ["Bill Gates", "George Soros", "a top politician", "a famous actor", "the elite"],
    "action": ["microchipping citizens", "poisoning the water supply", "funding global warming hoax", "mind control experiments"],
    "topic": ["5G towers", "chemtrails", "the flat earth", "vaccine microchips", "the deep state", "lizard people"],
    "authority": ["the Deep State", "Big Pharma", "the government", "the Illuminati", "the CDC"],
    "product": ["this simple herb", "apple cider vinegar", "this ancient remedy", "black seed oil"],
    "disease": ["cancer", "diabetes", "COVID", "autism", "Alzheimer's"],
    "days": ["3", "7", "10", "just 2"],
    "conspiracy": ["the New World Order plan", "the great reset agenda", "population control"],
    "event": ["the pandemic", "the 2020 election", "9/11", "the economic crash"],
    "year": ["2025", "this year", "before the election"],
    "place": ["Mars", "the Moon", "Antarctica", "the Arctic"],
}


def fill_template(template, slots):
    import re
    keys = re.findall(r"\{(\w+)\}", template)
    result = template
    for key in keys:
        if key in slots:
            result = result.replace(f"{{{key}}}", np.random.choice(slots[key]), 1)
    return result


def generate_dataset(num_samples=2000):
    records = []

    half = num_samples // 2

    for _ in range(half):
        template = np.random.choice(REAL_TEMPLATES)
        text = fill_template(template, REAL_SLOTS)
        # Add 1–3 more sentences
        extra = np.random.randint(1, 4)
        for _ in range(extra):
            t2 = np.random.choice(REAL_TEMPLATES)
            text += " " + fill_template(t2, REAL_SLOTS)
        records.append({"text": text.strip(), "label": 0})

    for _ in range(half):
        template = np.random.choice(FAKE_TEMPLATES)
        text = fill_template(template, FAKE_SLOTS)
        extra = np.random.randint(1, 4)
        for _ in range(extra):
            t2 = np.random.choice(FAKE_TEMPLATES)
            text += " " + fill_template(t2, FAKE_SLOTS)
        records.append({"text": text.strip(), "label": 1})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset.csv", index=False)
    print(f"[OK] Dataset saved: {len(df)} samples -> data/dataset.csv")
    print(df["label"].value_counts().rename({0: "Real", 1: "Fake"}))
    return df


if __name__ == "__main__":
    generate_dataset(2000)
