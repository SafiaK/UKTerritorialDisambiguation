import re
import spacy
from LegislationHandler import LegislationParser
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Any

TERRITORIES = [
    "England", "Wales", "Scotland", "Northern Ireland", "England and Wales", "United Kingdom"
]

def extract_territory_blocks(text, nlp):
    """
    Split section text into blocks by territory using regex and spaCy NER.
    Returns: {territory: [list of text blocks]}
    """
    doc = nlp(text)
    blocks = defaultdict(list)
    # Regex for explicit territory markers
    pattern = re.compile(r"(in|for|applies to|extends to|except in|does not apply to) ([A-Za-z ]+)", re.IGNORECASE)
    last_territory = None
    last_pos = 0
    for match in pattern.finditer(text):
        territory = match.group(2).strip()
        for terr in TERRITORIES:
            if terr.lower() in territory.lower():
                # Save previous block
                if last_territory:
                    blocks[last_territory].append(text[last_pos:match.start()].strip())
                last_territory = terr
                last_pos = match.end()
    # Save last block
    if last_territory:
        blocks[last_territory].append(text[last_pos:].strip())
    # Fallback: use NER to find territory mentions in sentences
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ == "GPE" and ent.text in TERRITORIES:
                blocks[ent.text].append(sent.text)
    return blocks

def filter_sections_with_territorial_mentions(sections: List[Dict[str, Any]], nlp=None) -> List[Dict[str, Any]]:
    """
    Filters sections that mention any UK territory in the context of legal phrases like 'applies to', 'extends to', 'except in', etc.
    
    - This function is designed for use in the main pipeline to select only those sections where a territory is mentioned in a legal context, not just anywhere in the text.
    - It uses a regex pattern to match phrases such as 'applies to England', 'extends to Wales', 'except in Scotland', etc. The pattern is:
        (in|for|applies to|extends to|except in|does not apply to) ([A-Za-z ]+)
      This ensures we only pick up territory mentions that are part of a legal scope or application clause.
    - Additionally, it uses spaCy's NER to look for GPE (Geo-Political Entity) entities, which in spaCy are used to tag place names, countries, cities, etc. This acts as a fallback to catch edge cases where the regex might miss a territory mention.
    - For each section, if any territory is found in these contexts, the function returns the full section text and a list of detected territories for further processing (e.g., LLM analysis).
    - Output: List of dicts with 'id', 'text', and 'territories'.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    filtered = []
    pattern = re.compile(r"(in|for|applies to|extends to|except in|does not apply to) ([A-Za-z ]+)", re.IGNORECASE)
    for section in sections:
        found_territories = set()
        # Regex-based detection for legal context
        for match in pattern.finditer(section['text']):
            territory = match.group(2).strip()
            for terr in TERRITORIES:
                if terr.lower() in territory.lower():
                    found_territories.add(terr)
        # NER-based detection (GPE = Geo-Political Entity, e.g., country, region)
        doc = nlp(section['text'])
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text in TERRITORIES:
                found_territories.add(ent.text)
        if found_territories:
            filtered.append({
                'id': section['id'],
                'text': section['text'],
                'territories': list(found_territories)
            })
    return filtered

def compare_territory_rules(blocks):
    """
    Compare the text/rules for each territory using semantic similarity.
    Returns: {territory: rule_text}, and a flag if all are the same.
    """
    rules = {k: " ".join(v) for k, v in blocks.items()}
    if len(rules) <= 1:
        return rules, True
    # Use TF-IDF to compare
    texts = list(rules.values())
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)
    similarities = cosine_similarity(vectors)
    # If all pairs are highly similar, treat as same
    all_similar = all(similarities[i, j] > 0.95 for i in range(len(texts)) for j in range(i+1, len(texts)))
    return rules, all_similar

def analyze_act(url, output_json="retained_EU_law_territorial_disambiguation_cases.json"):
    nlp = spacy.load("en_core_web_sm")
    parser = LegislationParser(url)
    sections = parser.get_sections()
    disambiguation_cases = {}
    for section in sections:
        section_id = section['id']
        text = section['text']
        blocks = extract_territory_blocks(text, nlp)
        if not blocks:
            print(f"Section {section_id}: No explicit territorial distinction found.")
            continue
        rules, all_same = compare_territory_rules(blocks)
        if not all_same:
            disambiguation_cases[section_id] = rules
            print(f"Section {section_id}: Law differs by territory:")
            for terr, rule in rules.items():
                print(f"  {terr}: {rule[:200]}{'...' if len(rule) > 200 else ''}")
            print()
    if disambiguation_cases:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(disambiguation_cases, f, indent=2, ensure_ascii=False)
        print(f"Disambiguation cases written to {output_json}")
    else:
        print("No territorial disambiguation cases found.")

if __name__ == "__main__":
    #url = input("Enter legislation.gov.uk Act URL: ").strip()
    url = "https://www.legislation.gov.uk/ukpga/2023/28"
    analyze_act(url) 