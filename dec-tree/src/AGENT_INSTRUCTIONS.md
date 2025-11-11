# Agent Instructions for Diagnosis Generation

## Input Data Structure

Each patient file contains:
- `records.clinicalDetails.chiefComplaints` - Primary symptoms and history
- `records.clinicalDetails.clientHistory` - Medical history
- `records.objectiveAssessment.tests[]` - Clinical test results with testName and result
- `records.subjectiveAssessment.assessment` - Therapist's assessment and plan

## Output Requirements

Generate a JSON response with:

1. **llm_diagnosis** (string): Top 5 suspected diagnoses with probabilities (85%, 70%, 60%, 45%, 30%)
   - Format: "1. Suspected [diagnosis]? (XX%)\n2. Suspected [diagnosis]? (XX%)"
   - Always use "Suspected" prefix and "?" suffix
   - Use descending probability order

2. **diagnoses_with_probability** (array): Structured version of llm_diagnosis
   ```json
   [
     {"diagnosis": "1. Suspected [diagnosis]?", "probability": 85},
     {"diagnosis": "2. Suspected [diagnosis]?", "probability": 70}
   ]
   ```

3. **required_fields** (string): List of JSON paths used for diagnosis
   - Format: "- records.clinicalDetails.chiefComplaints\n- records.objectiveAssessment.tests[].testName"
   - Include specific test names if relevant: "(Clark's Test, Thessaly's Test - positive findings)"

4. **original_diagnosis** (string): Extract from assessment or infer from complaints (uppercase)

## Key Rules

- Focus on positive test findings in objectiveAssessment.tests[]
- Consider chief complaints and client history together
- Use medical abbreviations (PFPS, ACLR, IT band, etc.)
- Always include anatomical side (R/L/bilateral)
- Probabilities must be: 85%, 70%, 60%, 45%, 30%
- All diagnoses should end with "?"

## Example Mapping

**Input:**
- chiefComplaints: "R knee pain after ACLR"
- tests: [{"testName": "Lachman's Test", "result": "positive", "side": "R"}]

**Output:**
- "1. Suspected R knee ACL graft failure? (85%)"
- required_fields includes: "records.objectiveAssessment.tests[].testName (Lachman's Test - positive findings)"
