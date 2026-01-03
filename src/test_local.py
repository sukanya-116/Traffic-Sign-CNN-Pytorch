import os
import sys
import time
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# -----------------------------
# Configuration
# -----------------------------
SERVER_URL = os.environ.get("PREDICT_SERVER_URL", "http://127.0.0.1:8080/predict")
TIMEOUT = 10
MAX_WORKERS = 2   

# -----------------------------
# Resolve paths
# -----------------------------
repo_root = Path(__file__).resolve().parents[1]
images_dir = repo_root / "test-images"

if not images_dir.is_dir():
    print(f"❌ Test images directory not found: {images_dir}")
    sys.exit(1)

images = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    print(f"❌ No images found in {images_dir}")
    sys.exit(1)

# -----------------------------
# Helper functions
# -----------------------------
def upload_image(session: requests.Session, img_path: Path):
    mime_type, _ = mimetypes.guess_type(img_path)
    mime_type = mime_type or "application/octet-stream"

    try:
        with img_path.open("rb") as f:
            files = {"file": (img_path.name, f, mime_type)}
            resp = session.post(SERVER_URL, files=files, timeout=TIMEOUT)

        result = {
            "image": img_path.name,
            "status": resp.status_code,
            "response": None,
            "error": None,
        }

        try:
            result["response"] = resp.json()
        except Exception:
            result["response"] = resp.text

        return result

    except requests.exceptions.RequestException as e:
        return {
            "image": img_path.name,
            "status": None,
            "response": None,
            "error": str(e),
        }

# -----------------------------
# Main test runner
# -----------------------------
def test_all_images():
    print(f"🚀 Sending {len(images)} images to {SERVER_URL}")
    start_time = time.time()

    results = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(upload_image, session, img): img
                for img in images
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result["error"]:
                    print(f"❌ {result['image']} → ERROR: {result['error']}")
                else:
                    print(
                        f"✅ {result['image']} → "
                        f"HTTP {result['status']} | {result['response']}"
                    )

    elapsed = time.time() - start_time

    # -----------------------------
    # Summary
    # -----------------------------
    success = sum(1 for r in results if r["status"] == 200)
    failed = len(results) - success

    print("\n================ SUMMARY ================")
    print(f"Total images : {len(results)}")
    print(f"Successful  : {success}")
    print(f"Failed      : {failed}")
    print(f"Elapsed     : {elapsed:.2f}s")
    print("========================================")

    return results


if __name__ == "__main__":
    test_all_images()
