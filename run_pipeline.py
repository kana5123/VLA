"""Run the full OpenVLA Attention Analysis Pipeline.

Usage:
    python run_pipeline.py --all                    # Full pipeline
    python run_pipeline.py --download               # Download data only
    python run_pipeline.py --extract                 # Extract attention only (requires GPU)
    python run_pipeline.py --visualize               # Visualize results only
    python run_pipeline.py --extract --visualize     # Extract + visualize
    python run_pipeline.py --all --num_episodes 5    # Full pipeline, 5 episodes
    python run_pipeline.py --extract --episodes 0,1  # Extract specific episodes
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA Attention Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all                     Full pipeline (download → extract → visualize)
  python run_pipeline.py --download                Download BridgeData V2 only (CPU OK)
  python run_pipeline.py --extract --device cuda   Extract attention (GPU required)
  python run_pipeline.py --visualize               Generate visualizations from saved JSONs
  python run_pipeline.py --extract --episodes 0,1  Process specific episodes only
        """,
    )

    # Pipeline stages
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--download", action="store_true", help="Download BridgeData V2")
    parser.add_argument("--extract", action="store_true", help="Extract attention weights (GPU)")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")

    # Parameters
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to download")
    parser.add_argument("--episodes", type=str, default=None, help="Comma-separated episode IDs for extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device for extraction (default: cuda)")

    args = parser.parse_args()

    # If no stage specified, show help
    if not (args.all or args.download or args.extract or args.visualize):
        parser.print_help()
        sys.exit(0)

    run_download = args.all or args.download
    run_extract = args.all or args.extract
    run_visualize = args.all or args.visualize

    total_start = time.time()

    # ── Stage 1: Download ─────────────────────────────────────────────────
    if run_download:
        print("=" * 60)
        print("STAGE 1: Downloading BridgeData V2")
        print("=" * 60)
        stage_start = time.time()

        from download_bridge_data import download_bridge_data
        import config

        num_episodes = args.num_episodes or config.NUM_EPISODES
        download_bridge_data(num_episodes=num_episodes)

        elapsed = time.time() - stage_start
        print(f"Download completed in {elapsed:.1f}s\n")

    # ── Stage 2: Extract Attention ────────────────────────────────────────
    if run_extract:
        print("=" * 60)
        print("STAGE 2: Extracting Attention Weights")
        print("=" * 60)
        stage_start = time.time()

        from extract_attention import run_extraction

        episode_ids = None
        if args.episodes:
            episode_ids = [int(x) for x in args.episodes.split(",")]

        run_extraction(episode_ids=episode_ids, device=args.device)

        elapsed = time.time() - stage_start
        print(f"Extraction completed in {elapsed:.1f}s\n")

    # ── Stage 3: Visualize ────────────────────────────────────────────────
    if run_visualize:
        print("=" * 60)
        print("STAGE 3: Generating Visualizations")
        print("=" * 60)
        stage_start = time.time()

        from visualize_results import run_visualization

        episode_ids = None
        if args.episodes:
            episode_ids = [int(x) for x in args.episodes.split(",")]

        run_visualization(episode_ids=episode_ids)

        elapsed = time.time() - stage_start
        print(f"Visualization completed in {elapsed:.1f}s\n")

    # ── Done ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"Pipeline completed in {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
