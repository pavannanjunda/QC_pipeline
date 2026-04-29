"""
audit_schema.py
---------------
Analyzes the MongoDB 'sessions' collection for schema inconsistencies.
"""

from pymongo import MongoClient
import os
from dotenv import load_dotenv
from collections import Counter
import json

load_dotenv()

def audit():
    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client["test"]
        sessions_col = db["sessions"]
        
        total_docs = sessions_col.count_documents({})
        print(f"--- MongoDB Schema Audit ---")
        print(f"Total Documents: {total_docs}\n")

        all_keys_counter = Counter()
        schema_groups = {}
        missing_critical = {
            "nas_file_path": [],
            "qc_score": [],
            "createdAt": [],
            "metadata": []
        }

        cursor = sessions_col.find({})
        for doc in cursor:
            # 1. Track all top-level keys
            doc_keys = tuple(sorted(doc.keys()))
            all_keys_counter.update(doc_keys)
            
            # 2. Group by schema signature
            schema_groups[doc_keys] = schema_groups.get(doc_keys, 0) + 1
            
            # 3. Check for missing critical fields
            for field in missing_critical.keys():
                if field not in doc:
                    missing_critical[field].append(doc["_id"])
            
            # 4. Check nested metadata consistency
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                meta = doc["metadata"]
                if "qc_detail" not in meta:
                    # Some old docs might have metadata but no qc_detail
                    pass

        print("Key Frequency (How many docs have this field):")
        for key, count in all_keys_counter.most_common():
            percentage = (count / total_docs) * 100
            print(f"  - {key:20} : {count:5} ({percentage:5.1f}%)")

        print("\nSchema Variations (Sets of fields found):")
        for schema, count in sorted(schema_groups.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {count:3} docs have these fields: {list(schema)}")

        print("\nCritical Missing Fields (Action required):")
        for field, ids in missing_critical.items():
            if ids:
                print(f"  - Missing '{field}': {len(ids)} documents")
                if len(ids) <= 5:
                    print(f"    IDs: {ids}")
                else:
                    print(f"    IDs (First 5): {ids[:5]}...")

    except Exception as e:
        print(f"Audit failed: {e}")

if __name__ == "__main__":
    audit()
