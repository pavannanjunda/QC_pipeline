import os
import uuid
import secrets
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Load config from env
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "test")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "sessions")
NAS_PATH = os.getenv("VIDEO_BASE_DIR")

def generate_short_uuid():
    """Generate a 10-character hex string like the ones in your DB."""
    return secrets.token_hex(5)

def sync():
    if not MONGO_URI or not NAS_PATH:
        print("Error: MONGO_URI or VIDEO_BASE_DIR not set in .env")
        return

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    sessions_col = db[COLLECTION_NAME]
    
    # Defaults if no template found
    user_id = "5c46f762-bd0c-4e14-9898-f6881e9bbfb5"
    task_id = "25e59758-66ab-4525-9ab6-9c92e09293a9"

    print(f"Syncing NAS files from {NAS_PATH} into collection '{COLLECTION_NAME}'...")
    
    if not os.path.exists(NAS_PATH):
        print(f"Error: Path {NAS_PATH} does not exist.")
        return

    # Find all relevant files
    files = [f for f in os.listdir(NAS_PATH) if f.endswith(".zip") or f.endswith(".mp4") or f.endswith(".mov")]
    
    count = 0
    for filename in files:
        slug = filename.replace(".zip", "").replace(".mp4", "").replace(".mov", "")
        session_id = f"Session_{slug}"
        
        # Check if already exists by _id
        if sessions_col.find_one({"_id": session_id}):
            continue
            
        new_session = {
            "_id": session_id,
            "user_id": user_id,
            "task_id": task_id,
            "video_uuid": generate_short_uuid(), # Generate unique UUID to satisfy DB index
            "upload_status": "Uploaded",
            "annotation_status": "Queued",
            "nas_file_path": filename,
            "qc_score": 0,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "metadata": {}
        }
        
        try:
            sessions_col.insert_one(new_session)
            print(f"Created session: {session_id}")
            count += 1
        except Exception as e:
            print(f"Failed to insert {session_id}: {e}")
        
    print(f"Done! Created {count} new sessions in '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    sync()
