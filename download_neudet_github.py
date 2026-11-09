"""
Download NEU-DET dataset from GitHub repository.
"""
import requests
import os
import base64
import sys
from pathlib import Path

GITHUB_API = "https://api.github.com"
REPO_OWNER = "siddhartamukherjee"
REPO_NAME = "NEU-DET-Steel-Surface-Defect-Detection"

def get_dir_contents(path=""):
    """Get directory contents from GitHub API."""
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"  Error {resp.status_code}: {resp.json().get('message', 'unknown')}")
        return None

def download_file(remote_path, local_path):
    """Download a single file from GitHub."""
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{remote_path}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data["encoding"] == "base64":
                content = base64.b64decode(data["content"])
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(content)
                return len(content)
        return 0
    except Exception as e:
        print(f"  Error downloading {remote_path}: {e}")
        return 0

def download_directory(remote_dir, local_dir, extensions=None, max_files=None):
    """Download all files from a directory recursively."""
    print(f"Listing {remote_dir}...")
    contents = get_dir_contents(remote_dir)
    if not contents:
        return 0

    count = 0
    total_size = 0

    for item in contents:
        if item["type"] == "dir":
            # Recurse into subdirectory
            sub_remote = item["path"]
            sub_local = os.path.join(local_dir, item["name"])
            c, s = download_directory(sub_remote, sub_local, extensions, max_files)
            count += c
            total_size += s
        elif item["type"] == "file":
            # Check extension filter
            if extensions:
                ext = os.path.splitext(item["name"])[1].lower()
                if ext not in extensions:
                    continue

            if max_files and count >= max_files:
                break

            local_file = os.path.join(local_dir, item["name"])
            if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
                # Skip already downloaded files
                count += 1
                continue

            size = download_file(item["path"], local_file)
            if size > 0:
                count += 1
                total_size += size
                if count % 50 == 0:
                    print(f"  Downloaded {count} files ({total_size/1024/1024:.1f} MB)...")

    return count, total_size

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/NEU-DET"

    print("=" * 60)
    print(f"Downloading NEU-DET dataset from GitHub")
    print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()

    # Check IMAGES directory
    print("Step 1: Checking IMAGES directory...")
    images = get_dir_contents("IMAGES")
    if images:
        print(f"  Found {len(images)} items in IMAGES/")
        # Show sample
        for img in images[:5]:
            print(f"    {img['name']} ({img.get('size', '?')} bytes)")
    print()

    # Check ANNOTATIONS directory
    print("Step 2: Checking ANNOTATIONS directory...")
    annotations = get_dir_contents("ANNOTATIONS")
    if annotations:
        print(f"  Found {len(annotations)} items in ANNOTATIONS/")
        for ann in annotations[:5]:
            print(f"    {ann['name']} ({ann.get('size', '?')} bytes)")
    print()

    # Estimate total size
    if images and annotations:
        total_img_size = sum(img.get("size", 0) for img in images if img["type"] == "file")
        total_ann_size = sum(ann.get("size", 0) for ann in annotations if ann["type"] == "file")
        print(f"Estimated total size:")
        print(f"  Images: {len(images)} files, {total_img_size/1024/1024:.1f} MB")
        print(f"  Annotations: {len(annotations)} files, {total_ann_size/1024:.1f} KB")
        print(f"  Total: {(total_img_size + total_ann_size)/1024/1024:.1f} MB")
        print()

    # Download
    print("Step 3: Downloading images...")
    img_count, img_size = download_directory("IMAGES", os.path.join(output_dir, "IMAGES"),
                                              extensions=[".jpg", ".jpeg", ".png", ".bmp"])
    print(f"  Downloaded {img_count} images ({img_size/1024/1024:.1f} MB)")
    print()

    print("Step 4: Downloading annotations...")
    ann_count, ann_size = download_directory("ANNOTATIONS", os.path.join(output_dir, "ANNOTATIONS"),
                                              extensions=[".xml"])
    print(f"  Downloaded {ann_count} annotations ({ann_size/1024:.1f} KB)")
    print()

    print("=" * 60)
    print("Download complete!")
    print(f"  Total: {img_count} images, {ann_count} annotations")
    print(f"  Total size: {(img_size + ann_size)/1024/1024:.1f} MB")
    print(f"  Saved to: {output_dir}")
    print("=" * 60)
