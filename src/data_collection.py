
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0"
}

urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Generative_artificial_intelligence"
]


corpus = []

for url in urls:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if len(text) > 20:
            corpus.append(text)

print("Total text samples collected:", len(corpus))

with open("ai_corpus.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        f.write(line + "\n")



import re

def clean_text(text):
    # Remove citation references like [1], [2], [a], [b]
    text = re.sub(r'\[[^\]]+\]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()



cleaned_corpus = []

for line in corpus:
    cleaned = clean_text(line)
    if len(cleaned.split()) > 8:
        cleaned_corpus.append(cleaned)

print("Final cleaned samples:", len(cleaned_corpus))

with open("ai_corpus_cleaned.txt", "w", encoding="utf-8") as f:
    for line in cleaned_corpus:
        f.write(line + "\n")