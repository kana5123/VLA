"""Download full BridgeData V2 tfrecord shards from GCS.

Downloads all 1024 train shards (~117GB total) to /ceph_data/kana5123/bridge_data_v2/.
Resumes from last completed shard if interrupted.

Usage:
    python download_full_bridge.py [--max_shards 1024]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import gcsfs


GCS_BUCKET = "gresearch/robotics/bridge_data_v2/0.0.1"
LOCAL_DIR = Path("/ceph_data/kana5123/bridge_data_v2/0.0.1")
TOTAL_SHARDS = 1024


def download_shards(max_shards: int = TOTAL_SHARDS) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    fs = gcsfs.GCSFileSystem(token="anon")

    # Download metadata files first
    for meta_file in ["dataset_info.json", "features.json"]:
        local_path = LOCAL_DIR / meta_file
        if not local_path.exists():
            print(f"Downloading {meta_file}...")
            fs.get(f"{GCS_BUCKET}/{meta_file}", str(local_path))

    # Download train shards
    num_shards = min(max_shards, TOTAL_SHARDS)
    completed = 0
    skipped = 0

    for i in range(num_shards):
        shard_name = f"bridge_data_v2-train.tfrecord-{i:05d}-of-{TOTAL_SHARDS:05d}"
        local_path = LOCAL_DIR / shard_name
        remote_path = f"{GCS_BUCKET}/{shard_name}"

        if local_path.exists():
            # Verify size matches
            remote_size = fs.info(remote_path)["size"]
            local_size = local_path.stat().st_size
            if local_size == remote_size:
                skipped += 1
                continue
            else:
                print(f"  Shard {i}: size mismatch (local={local_size}, remote={remote_size}), re-downloading")

        print(f"Downloading shard {i:05d}/{num_shards:05d} ({shard_name})...")
        try:
            # Download to temp file first, then rename (atomic)
            tmp_path = local_path.with_suffix(".tmp")
            fs.get(remote_path, str(tmp_path))
            tmp_path.rename(local_path)
            completed += 1
        except Exception as e:
            print(f"  ERROR on shard {i}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            continue

        if (completed + skipped) % 10 == 0:
            pct = (completed + skipped) / num_shards * 100
            print(f"  Progress: {completed + skipped}/{num_shards} ({pct:.1f}%)")

    print(f"\nDone! Downloaded: {completed}, Skipped (already exist): {skipped}")
    print(f"Data at: {LOCAL_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download full BridgeData V2 from GCS")
    parser.add_argument(
        "--max_shards",
        type=int,
        default=TOTAL_SHARDS,
        help=f"Max number of shards to download (default: {TOTAL_SHARDS})",
    )
    args = parser.parse_args()
    download_shards(max_shards=args.max_shards)


if __name__ == "__main__":
    main()
