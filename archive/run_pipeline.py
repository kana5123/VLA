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
    parser.add_argument("--enhance", action="store_true", help="Run attention enhancement experiment")
    parser.add_argument("--compare", action="store_true", help="Compare enhancement results")
    parser.add_argument("--enhance_v3", action="store_true", help="Run V3 research-based enhancement (VAR/ACT/SPIN)")
    parser.add_argument("--compare_v3", action="store_true", help="Compare V3 enhancement results")

    # Parameters
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to download")
    parser.add_argument("--episodes", type=str, default=None, help="Comma-separated episode IDs for extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device for extraction (default: cuda)")
    parser.add_argument("--enhance_method", type=str, default="all", help="Enhancement methods (comma-separated or 'all')")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline in enhancement experiment")

    args = parser.parse_args()

    # If no stage specified, show help
    if not (args.all or args.download or args.extract or args.visualize
            or args.enhance or args.compare or args.enhance_v3 or args.compare_v3):
        parser.print_help()
        sys.exit(0)

    run_download = args.all or args.download
    run_extract = args.all or args.extract
    run_visualize = args.all or args.visualize
    run_enhance = args.enhance
    run_compare = args.compare
    run_enhance_v3 = args.enhance_v3
    run_compare_v3 = args.compare_v3

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

    # ── Stage 4: Enhancement Experiment ─────────────────────────────────
    if run_enhance:
        print("=" * 60)
        print("STAGE 4: Attention Enhancement Experiment")
        print("=" * 60)
        stage_start = time.time()

        from run_enhancement import run_experiment, METHODS

        episode_ids = None
        if args.episodes:
            episode_ids = [int(x) for x in args.episodes.split(",")]

        methods = (
            METHODS if args.enhance_method == "all"
            else [m.strip() for m in args.enhance_method.split(",")]
        )
        run_experiment(
            episode_ids=episode_ids, device=args.device,
            methods=methods, skip_baseline=args.skip_baseline,
        )

        elapsed = time.time() - stage_start
        print(f"Enhancement completed in {elapsed:.1f}s\n")

    # ── Stage 5: Compare Results ─────────────────────────────────────────
    if run_compare:
        print("=" * 60)
        print("STAGE 5: Compare Enhancement Results")
        print("=" * 60)
        stage_start = time.time()

        from compare_results import run_comparison
        run_comparison()

        elapsed = time.time() - stage_start
        print(f"Comparison completed in {elapsed:.1f}s\n")

    # ── Stage 6: V3 Enhancement (Research-based) ────────────────────────
    if run_enhance_v3:
        print("=" * 60)
        print("STAGE 6: V3 Enhancement (VAR/ACT/SPIN)")
        print("=" * 60)
        stage_start = time.time()

        from run_v3_experiment import run_v3_experiment

        episode_ids = None
        if args.episodes:
            episode_ids = [int(x) for x in args.episodes.split(",")]

        run_v3_experiment(
            episode_ids=episode_ids, device=args.device,
            reuse_baseline=True,
        )

        elapsed = time.time() - stage_start
        print(f"V3 Enhancement completed in {elapsed:.1f}s\n")

    # ── Stage 7: Compare V3 Results ───────────────────────────────────
    if run_compare_v3:
        print("=" * 60)
        print("STAGE 7: Compare V3 Results")
        print("=" * 60)
        stage_start = time.time()

        from compare_v3_results import run_v3_comparison
        run_v3_comparison()

        elapsed = time.time() - stage_start
        print(f"V3 Comparison completed in {elapsed:.1f}s\n")

    # ── Done ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"Pipeline completed in {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
