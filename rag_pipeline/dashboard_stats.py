import json
from collections import Counter

def get_dashboard_stats():
    with open("scraper/schemes_data.json", "r", encoding="utf-8") as f:
        schemes = json.load(f)

    total_schemes = len(schemes)

    benefit_types = Counter()
    beneficiary_types = Counter()

    for s in schemes:
        benefit = s.get("Types of Benefits", "Unspecified").strip() or "Unspecified"
        beneficiary = s.get("Beneficiaries", "Unspecified").strip() or "Unspecified"
        benefit_types[benefit] += 1
        beneficiary_types[beneficiary] += 1

    return {
        "total_schemes": total_schemes,
        "benefit_types": dict(benefit_types),
        "beneficiary_types": dict(beneficiary_types),
    }