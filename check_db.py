import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# Add local bin to path just in case
sys.path.append(os.path.expanduser("~/.local/lib/python3.12/site-packages"))

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "test")

def check_db():
    print(f"Connecting to {DB_NAME}...")
    client = MongoClient(MONGO_URI)
    try:
        client.admin.command('ping')
        print("MongoDB Connection: SUCCESS")
        
        db = client[DB_NAME]
        sessions_col = db['sessions']
        
        pending_count = sessions_col.count_documents({
            "upload_status": "Uploaded",
            "qc_score": {"$in": [None, float("nan")]}
        })
        
        print(f"Pending sessions for QC: {pending_count}")
        
        if pending_count > 0:
            sample = sessions_col.find_one({"upload_status": "Uploaded", "qc_score": {"$in": [None, float("nan")]}})
            print("\nSample Pending Session:")
            print(f"  _id: {sample['_id']}")
            print(f"  video_uuid: {sample.get('video_uuid')}")
            print(f"  nas_file_path: {sample.get('nas_file_path')}")
            print(f"  storage_dir: {sample.get('storage_dir')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
