import json
from bson import ObjectId
from pathlib import Path

# Input and output paths
INPUT_FILE = r"C:\healthflex\mcp server\dec-tree\user-data\stance-dashboard.reports.json"
OUTPUT_DIR = "first_assessments"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Custom JSON encoder to handle ObjectId
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

# Load the JSON file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter only first assessments
first_assessments = [report for report in data if report.get("isFirstAssessment") is True]

print(f"Found {len(first_assessments)} first assessment(s). Processing...")

# Process each first assessment
for report in first_assessments:
    # Extract only required fields
    extracted = {
        "patient": str(report["patient"]["$oid"]) if isinstance(report["patient"], dict) else str(report["patient"]),
        "appointment": str(report["appointment"]["$oid"]) if isinstance(report["appointment"], dict) else str(report["appointment"]),
        "records": remove_provisional_diagnosis(report.get("records", {}))
    }

    # Use seqNo as filename
    seq_no = report.get("seqNo", f"unknown_{extracted['patient']}")
    filename = f"{seq_no}.json"
    filepath = Path(OUTPUT_DIR) / filename

    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, indent=2, cls=JSONEncoder, ensure_ascii=False)

    print(f"Saved: {filepath}")

print(f"\nDone! {len(first_assessments)} first assessment(s) saved in '{OUTPUT_DIR}/'")