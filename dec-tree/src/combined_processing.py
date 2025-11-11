#!/usr/bin/env python3
"""Combined data preprocessing and assessment transformation script."""

import json
from bson import ObjectId
from pathlib import Path

# Paths
INPUT_FILE = "/Users/chrisdev/Healthflex/mcp-clinicagent/dec-tree/user-data/stance-dashboard.reports.json"
FIRST_ASSESSMENTS_DIR = "/Users/chrisdev/Healthflex/mcp-clinicagent/dec-tree/first_assessments"

# Create directory
Path(FIRST_ASSESSMENTS_DIR).mkdir(exist_ok=True)

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

def remove_provisional_diagnosis(obj):
    """Recursively remove any key named 'provisionalDiagnosis'"""
    if isinstance(obj, dict):
        return {k: remove_provisional_diagnosis(v) for k, v in obj.items() if k != "provisionalDiagnosis"}
    elif isinstance(obj, list):
        return [remove_provisional_diagnosis(item) for item in obj]
    else:
        return obj



def main():
    # Load and filter data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    first_assessments = [
        report for report in data
        if report.get("isFirstAssessment") is True
        and report.get("records")
        and isinstance(report["records"], dict)
        and len(report["records"]) > 0
    ]

    print(f"Found {len(first_assessments)} first assessment(s). Processing...")

    for report in first_assessments:
        # Extract required fields
        extracted = {
            "patient": str(report["patient"]["$oid"]) if isinstance(report["patient"], dict) else str(report["patient"]),
            "appointment": str(report["appointment"]["$oid"]) if isinstance(report["appointment"], dict) else str(report["appointment"]),
            "records": remove_provisional_diagnosis(report.get("records", {}))
        }

        seq_no = report.get("seqNo", f"unknown_{extracted['patient']}")
        filename = f"{seq_no}.json"

        # Save preprocessed file
        preprocessed_path = Path(FIRST_ASSESSMENTS_DIR) / filename
        with open(preprocessed_path, 'w', encoding='utf-8') as f:
            json.dump(extracted, f, indent=2, cls=JSONEncoder, ensure_ascii=False)

        print(f"✓ {filename}")

    print(f"\nDone! {len(first_assessments)} files saved in '{FIRST_ASSESSMENTS_DIR}/'")

if __name__ == "__main__":
    main()