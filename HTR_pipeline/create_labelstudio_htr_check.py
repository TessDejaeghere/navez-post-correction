"""
Create LabelStudio JSON for HTR Quality Check

This script:
1. Reads the Nodegoat CSV with HTR clean status
2. Matches files in All_KBR_Raw.zip
3. Creates a LabelStudio JSON for annotation
"""

import pandas as pd
import zipfile
import json
import os
from pathlib import Path

# Paths
CSV_PATH = "HTR_pipeline/Nodegoat_export_2303.csv"
ZIP_PATH = "HTR_pipeline/All_KBR_Raw.zip"
OUTPUT_JSON = "HTR_pipeline/labelstudio_htr_quality_check.json"

print("=" * 70)
print("LabelStudio HTR Quality Check - Data Preparation")
print("=" * 70)
print()

# Read the CSV
print(f"Reading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✓ Found {len(df)} entries in CSV")
print()

# Clean column names (remove potential spaces)
df.columns = df.columns.str.strip()

# Check HTR clean column
htr_clean_col = "HTR clean?"
if htr_clean_col not in df.columns:
    print("Available columns:", list(df.columns))
    raise ValueError(f"Column '{htr_clean_col}' not found!")

# Create a lookup dictionary: Name -> HTR clean status
# Convert Name format: "Ms. II 70/1/91" -> "Ms. II 70_1_91"
htr_status = {}
for idx, row in df.iterrows():
    name = str(row['Name']).strip()
    status = str(row[htr_clean_col]).strip()
    
    # Convert to filename format (/ to _)
    filename_base = name.replace('/', '_')
    
    # Determine status
    is_clean = status.lower() == 'yes'
    
    htr_status[filename_base] = {
        'original_name': name,
        'is_clean': is_clean,
        'status_text': 'Clean' if is_clean else 'Needs Review'
    }

print(f"✓ Processed {len(htr_status)} letter statuses")
print(f"  - Clean: {sum(1 for v in htr_status.values() if v['is_clean'])}")
print(f"  - Needs Review: {sum(1 for v in htr_status.values() if not v['is_clean'])}")
print()

# Read files from ZIP
print(f"Reading ZIP: {ZIP_PATH}")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    # Filter for text files only
    txt_files = [f for f in file_list if f.endswith('.txt') and not f.startswith('__MACOSX')]
    
    print(f"✓ Found {len(txt_files)} text files in ZIP")
    print()
    
    # Create LabelStudio tasks
    tasks = []
    matched = 0
    unmatched = []
    
    print("Matching files with HTR status...")
    for txt_file in txt_files:
        # Extract filename without extension
        filename = os.path.basename(txt_file)
        file_base = filename.replace('.txt', '')
        
        # Try to match with HTR status
        if file_base in htr_status:
            matched += 1
            status_info = htr_status[file_base]
            
            # Read the file content
            with zip_ref.open(txt_file) as f:
                content = f.read().decode('utf-8', errors='ignore')
            
            # Create LabelStudio task
            task = {
                "data": {
                    "text": content,
                    "letter_id": status_info['original_name'],
                    "filename": filename,
                    "current_status": status_info['status_text']
                },
                "annotations": [
                    {
                        "result": [],
                        "was_cancelled": False,
                        "ground_truth": False,
                        "lead_time": 0
                    }
                ],
                "predictions": [
                    {
                        "model_version": "nodegoat_export",
                        "result": [
                            {
                                "value": {
                                    "choices": ["Clean" if status_info['is_clean'] else "Needs Review"]
                                },
                                "from_name": "quality",
                                "to_name": "text",
                                "type": "choices"
                            }
                        ]
                    }
                ]
            }
            tasks.append(task)
        else:
            unmatched.append(filename)
    
    print(f"✓ Matched: {matched} files")
    if unmatched:
        print(f"⚠️  Unmatched: {len(unmatched)} files")
        if len(unmatched) <= 10:
            for f in unmatched:
                print(f"   - {f}")
    print()

# Save to JSON
print(f"Saving LabelStudio JSON: {OUTPUT_JSON}")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(tasks)} tasks")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total tasks created: {len(tasks)}")
print(f"")
print(f"Tasks by status:")
clean_count = sum(1 for t in tasks if t['predictions'][0]['result'][0]['value']['choices'][0] == 'Clean')
needs_review_count = len(tasks) - clean_count
print(f"  - Clean (pre-marked): {clean_count}")
print(f"  - Needs Review (pre-marked): {needs_review_count}")
print()
print(f"📁 Output file: {OUTPUT_JSON}")
print()
print("Next steps:")
print("1. Import this JSON file into LabelStudio")
print("2. Create a project with a 'Text Classification' template")
print("3. Add choices: 'Clean' and 'Needs Review'")
print("4. Your students can review and correct the pre-annotations")
print("=" * 70)
