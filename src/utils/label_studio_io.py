import requests
import threading
from flask import Flask
import requests
import logging

PORT=6009
WEBHOOK_PORT = 8080
BRIDGE_URL = f"http://localhost:{PORT}"

# 1. Create the global event flag that freezes/unfreezes your pipeline
labeling_completed_event = threading.Event()

# 2. Setup the tiny Flask app
app = Flask(__name__)
# Silence the standard Flask logs so it doesn't spam your terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/webhook/labeling_done', methods=['POST'])
def labeling_done_webhook():
    """This gets triggered when the Bridge tells us the queue is empty."""
    print("\n🔔 [Pipeline] Wake-up call received! All videos in this round are labeled.")
    labeling_completed_event.set() # This unfreezes the wait() command!
    return {"status": "success"}, 200

def start_webhook_listener(port=WEBHOOK_PORT):
    """Runs the listener quietly in the background."""
    thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False))
    thread.daemon = True
    thread.start()

def upload_new_batch(group_name):
    """
    Tells the bridge to:
    1. Scan /home/youruser/model_outputs/<folder_name>
    2. Upload videos to Azure
    3. Sync Label Studio
    """
    print(f"🚀 Triggering upload for: {group_name}")
    response = requests.post(f"{BRIDGE_URL}/upload", params={"group_name": group_name})
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Sync complete. Annotators can now see the videos.")
        return data
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None

def download_labels(group_name, iteration: int = 0, round: int = 0, purge: bool = False):
    """
    Tells the bridge to:
    1. Download all completed annotations
    2. Wipe Azure and Label Studio to prepare for the next batch
    """
    print("📥 Collecting annotations and purging Azure Blob storage...")
    try:
        response = requests.get(
            f"{BRIDGE_URL}/collect", 
            params={"group_name": group_name, "iteration": iteration, "round": round, "purge": str(purge).lower()},
            timeout=30 # Prevent the pipeline from hanging forever
        )
        response.raise_for_status() # Raises an exception for 4xx/5xx errors
        
        data = response.json()
        print(f"✅ Collected {data['annotations_processed']} annotations.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error connecting to Bridge: {e}")
        # Return a safe fallback dictionary so the rest of your code doesn't crash
        return {"annotations_processed": 0, "preference_data": [], "purged_count": 0, "remaining_in_queue": -1}
    