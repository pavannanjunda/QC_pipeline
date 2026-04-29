import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "test")
NAS_PATH = "/mnt/nas/xp-capture-staging/Validating"

def sync():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    sessions_col = db["sessions"]
    
    # Get template IDs from the latest session
    template = sessions_col.find_one()
    user_id = template.get("user_id", "5c46f762-bd0c-4e14-9898-f6881e9bbfb5")
    task_id = template.get("task_id", "25e59758-66ab-4525-9ab6-9c92e09293a9")

    print(f"Syncing NAS files from {NAS_PATH}...")
    files = [f for f in os.listdir(NAS_PATH) if f.endswith(".zip")]
    
    count = 0
    for filename in files:
        # Check if already exists
        # Filename example: session_1777366517_iPhone.zip
        # We can use the filename as part of the ID
        session_id = f"Session_{filename.replace('.zip', '')}"
        
        if sessions_col.find_one({"_id": session_id}):
            print(f"Skipping {session_id} (already exists)")
            continue
            
        new_session = {
            "_id": session_id,
            "user_id": user_id,
            "task_id": task_id,
            "upload_status": "Uploaded",
            "annotation_status": "Queued",
            "nas_file_path": filename, # Relative to VIDEO_BASE_DIR
            "qc_score": 0,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "metadata": {}
        }
        
        sessions_col.insert_one(new_session)
        print(f"Created session: {session_id}")
        count += 1
        
    print(f"Done! Created {count} new sessions.")

if __name__ == "__main__":
    sync()
