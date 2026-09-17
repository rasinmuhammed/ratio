#!/usr/bin/env python3
"""Export Human-in-the-Loop DPO feedback to HuggingFace JSONL format.

This script reads all corrected answers from the audit.db and formats them 
as a preference dataset for Direct Preference Optimization (DPO).

Format: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/audit.db")
OUTPUT_PATH = Path("data/dpo_dataset.jsonl")

def main():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    print(f"Reading DPO feedback from {DB_PATH}...")
    
    records_exported = 0
    with sqlite3.connect(DB_PATH) as conn:
        try:
            cursor = conn.execute("SELECT query, original_answer, corrected_answer FROM dpo_feedback")
            
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                for row in cursor:
                    query, rejected, chosen = row
                    
                    # HuggingFace TRL expects this standard format
                    record = {
                        "prompt": query,
                        "chosen": chosen,
                        "rejected": rejected
                    }
                    
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    records_exported += 1
                    
            print(f"Successfully exported {records_exported} DPO records to {OUTPUT_PATH}.")
        except sqlite3.OperationalError as e:
            print(f"Database error (have you submitted any corrections yet?): {e}")

if __name__ == "__main__":
    main()
