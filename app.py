# app.py

import os, tempfile, shutil, subprocess, json
from typing import List, Optional, Tuple
from flask import Flask, request, jsonify, Response
import requests
from supabase import create_client, Client
from supabase.lib.storage_http_exceptions import StorageHttpException
from flask_cors import CORS
from pathlib import Path
from urllib.parse import urlparse

# ------------ Config via env ------------
# IMPORTANT: Use the Supabase Service Role Key for elevated storage permissions
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET_NAME", "processed_data")

# --- PATH TEMPLATES ---
# Input Embedding Path in Supabase: processed/{patient_id}/patient_{patient_id}_processed.npy
EMB_BUCKET_PATH_TEMPLATE = "processed/{patient_id}/patient_{patient_id}_processed.npy"

# Output Video Path in Supabase: processed/{patient_id}/{recording_unique_id}_tracked.mp4
VIDEO_BUCKET_PATH_TEMPLATE = "processed/{patient_id}/{recording_unique_id}_tracked.mp4"

TRACK_SCRIPT_PATH = os.environ.get("TRACK_SCRIPT_PATH", "scripts/track_patient_deepsort.py")
# ---------------------------------------

app = Flask(__name__)
CORS(app)
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper Functions ---
def _download_from_storage_to(path_in_bucket: str, dest_file: str, bucket: str):
    """Download a file from a specified Supabase bucket."""
    try:
        blob = sb.storage.from_(bucket).download(path_in_bucket)
        with open(dest_file, "wb") as f:
            f.write(blob)
    except StorageHttpException as e:
        if "404" in str(e):
             raise FileNotFoundError(f"Supabase file not found: {bucket}/{path_in_bucket}")
        raise e

def _download_from_url_to(url: str, dest_file: str):
    """Download a file from a direct URL (e.g., recordings URL)."""
    r = requests.get(url, timeout=300) 
    r.raise_for_status()
    with open(dest_file, "wb") as f:
        f.write(r.content)

def _upload_to_storage(local_path: str, bucket_path: str, bucket_name: str) -> str:
    """Uploads a file to a bucket and returns the public URL."""
    with open(local_path, 'rb') as f:
        sb.storage.from_(bucket_name).upload(bucket_path, f.read(), {'content-type': 'video/mp4'})
    
    public_url = sb.storage.from_(bucket_name).get_public_url(bucket_path)
    return public_url

def _run_tracker_script_in(temp_root: str, video_in: str, emb_in: str, video_out: str) -> str:
    """Run the deepsort script with dynamic paths."""
    script_path_full = os.path.join(temp_root, TRACK_SCRIPT_PATH)
    
    # We execute the script using its full path and pass absolute file paths as arguments
    result = subprocess.run(
        ["python", script_path_full, video_in, emb_in, video_out],
        cwd=temp_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # IMPORTANT: Increase timeout or use asynchronous processing for long videos
        timeout=3500 # Slightly less than the max 3600s
    )
    if result.returncode != 0:
        print(f"--- Tracker Script STDOUT ---\n{result.stdout}")
        print(f"--- Tracker Script STDERR ---\n{result.stderr}")
        raise RuntimeError(f"Tracker script failed with return code {result.returncode}.")
    return result.stdout
# --- End Helper Functions ---

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.post("/track")
def track():
    """
    POST API call to perform DeepSort tracking on a recording.
    Input: { "patient_id": "...", "recording_url": "..." }
    """
    if not request.is_json:
        return jsonify(error="send JSON payload"), 400
    data = request.get_json()
    patient_id = data.get("patient_id")
    recording_url = data.get("recording_url") 
    recording_id = data.get("recording_id") 

    if not patient_id or not recording_url or not recording_id:
        return jsonify(error="patient_id, recording_url, and recording_id are required"), 400

    # Extract a unique ID from the recording URL (e.g., the filename part)
    try:
        path = urlparse(recording_url).path
        recording_unique_id_with_ext = path.split('/')[-1]
        
        if recording_unique_id_with_ext.lower().endswith('.mp4'):
            recording_unique_id = recording_unique_id_with_ext.rsplit('.', 1)[0]
        else:
            recording_unique_id = recording_unique_id_with_ext
            
    except Exception as e:
        return jsonify(error=f"Invalid recording_url format for ID extraction: {str(e)}"), 400

    temp_root = None
    try:
        temp_root = tempfile.mkdtemp(prefix=f"track_{patient_id}_{recording_unique_id}_")
        
        # Determine unique paths
        emb_bucket_path = EMB_BUCKET_PATH_TEMPLATE.format(patient_id=patient_id)
        video_bucket_path = VIDEO_BUCKET_PATH_TEMPLATE.format(
            patient_id=patient_id, 
            recording_unique_id=recording_unique_id
        )

        # Local file paths
        local_video_in_path = os.path.join(temp_root, "input_video.mp4")
        local_emb_path = os.path.join(temp_root, "input_ref_emb.npy")
        local_video_out_path_abs = os.path.join(temp_root, "outputs", f"{recording_unique_id}_tracked.mp4")

        Path(os.path.dirname(local_video_out_path_abs)).mkdir(parents=True, exist_ok=True)
        
        # 2. Download patient embedding (from 'processed_data')
        _download_from_storage_to(emb_bucket_path, local_emb_path, STORAGE_BUCKET)
        
        # 3. Download recording video
        _download_from_url_to(recording_url, local_video_in_path)

        # 4. Run the deepsort script
        _run_tracker_script_in(
            temp_root, 
            local_video_in_path, 
            local_emb_path, 
            local_video_out_path_abs
        )

        # 5. Upload the resulting video (to 'processed_data' with unique path)
        if not os.path.exists(local_video_out_path_abs):
            raise RuntimeError("Output video not generated by tracker script.")

        processed_video_url = _upload_to_storage(
            local_video_out_path_abs, video_bucket_path, STORAGE_BUCKET
        )
        
        # 6. Database Update (Crucial)
        # In a real scenario, you'd update your 'recordings' table here:
        # sb.table('recordings').update({'processed_video_url': processed_video_url, 'status': 'COMPLETED'}).eq('some_unique_id', recording_url_id).execute()
        # 6. Database Update (CRUCIAL ASYNCHRONOUS STEP)
        sb.table('recordings').update({
            'processed_video_url': processed_video_url, 
            'status': 'completed', 
            'updated_at': 'now()' # Optional: useful timestamp
        }).eq('id', recording_id).execute()

        return jsonify(processed_video_url=processed_video_url, status="COMPLETED"), 200

    except FileNotFoundError as e:
        return jsonify(error=f"File not found: {str(e)}"), 404
    except Exception as e:
        print(f"Tracking error: {str(e)}")
        return jsonify(error=f"Tracking failed: {str(e)}"), 500
    finally:
        if temp_root and os.path.exists(temp_root):
            shutil.rmtree(temp_root, ignore_errors=True)