import json
import re

INPUT_FILE = "scraper/schemes_data.json"
OUTPUT_FILE = "rag_pipeline/cleaned_documents.json"

# Fields we want to show, in a sensible reading order.
# (We skip "Scheme Type" and "Uploaded File" - mostly empty/useless in our data)
FIELD_ORDER = [
    "Funding Pattern",
    "Beneficiaries",
    "Types of Benefits",
    "Description",
    "How To avail",
    "Sponsered By",
]

def clean_text(text):
    """Fix messy line breaks and extra spaces from the scraped website text."""
    if not text:
        return ""
    text = text.replace("\r\n", " ")        # convert raw line breaks to spaces
    text = re.sub(r"\s+", " ", text)         # collapse multiple spaces/newlines into one
    return text.strip()

def is_useless(value):
    """Detect placeholder values that carry no real information."""
    if not value:
        return True
    return value.strip().lower() in ["-", "na", "n/a", ""]

def format_scheme_as_text(scheme):
    """Turn one scheme's dictionary into a clean, readable paragraph."""
    lines = [f"Scheme Name: {scheme.get('name', 'Unknown')}"]

    for field in FIELD_ORDER:
        value = scheme.get(field, "")
        cleaned = clean_text(value)
        if not is_useless(cleaned):
            lines.append(f"{field}: {cleaned}")

    return "\n".join(lines)

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    documents = []
    for scheme in schemes:
        text = format_scheme_as_text(scheme)
        document = {
            "page_content": text,
            "metadata": {
                "scheme_name": scheme.get("name", "Unknown"),
                "source_url": scheme.get("url", "")
            }
        }
        documents.append(document)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"✅ Prepared {len(documents)} clean documents → {OUTPUT_FILE}")
    print("\n--- Sample document ---\n")
    print(documents[0]["page_content"])

if __name__ == "__main__":
    main()