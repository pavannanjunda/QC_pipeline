import os
import sys
import json

# Add local packages to path
sys.path.append(os.path.expanduser("~/.local/lib/python3.12/site-packages"))

from pipeline.orchestrator import run_pipeline
from utils.logger import logger

def run_test():
    # Mock session
    session = {
        "_id": "test_session_001",
        "video_uuid": "test-v-001",
        "nas_file_path": "dummy_test.mp4",
        "metadata": {
            "task_description": "A green square moving across a black background",
            "task_keywords": ["green", "square", "moving"]
        }
    }
    
    logger.info("Starting test run with dummy video...")
    try:
        result = run_pipeline(session)
        print("\n" + "="*60)
        print("PIPELINE TEST RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        print("="*60)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
