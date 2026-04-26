"""
Main application for Tekken 8 video analysis.
Processes videos and generates statistics on UI elements.
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Load .env from project directory (not only cwd — fixes Mongo/keys when run elsewhere)
try:
    from dotenv import load_dotenv
    _ENV_FILE = Path(__file__).resolve().parent / ".env"
    load_dotenv(_ENV_FILE)
    load_dotenv()  # cwd .env overrides if present
except ImportError:
    pass

from ui_detector import UIDetector
from video_processor import VideoProcessor
from stats_generator import StatsGenerator
from youtube_input import (
    download_youtube,
    is_youtube_url,
    parse_js_runtimes_specs,
    parse_remote_components_specs,
)


def _resolve_yt_dlp_js_runtimes(cli_specs: list | None) -> dict | None:
    specs = cli_specs
    if not specs:
        env_val = os.environ.get("YTDLP_JS_RUNTIMES", "").strip()
        if env_val:
            specs = [x for x in re.split(r"[\s,]+", env_val) if x]
    if not specs:
        return None
    return parse_js_runtimes_specs(specs)


def _resolve_yt_dlp_remote_components(cli_specs: list | None) -> list | None:
    specs = cli_specs
    if not specs:
        env_val = os.environ.get("YTDLP_REMOTE_COMPONENTS", "").strip()
        if env_val:
            specs = [x.strip() for x in env_val.split(",") if x.strip()]
    if not specs:
        return None
    return parse_remote_components_specs(specs)


def _fmt_time(seconds: float) -> str:
    """Format seconds as m:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def make_progress_callback(frame_skip: int = 1):
    """Return a progress callback that renders a live progress bar."""
    start_wall = time.time()

    def callback(frame: int, total: int, video_time: float, total_events: int):
        pct = frame / total if total > 0 else 0
        elapsed = time.time() - start_wall

        video_duration = video_time / pct if pct > 0 else 0
        eta = (elapsed / pct - elapsed) if pct > 0 else 0

        bar_width = 28
        filled = int(bar_width * pct)
        enc = getattr(sys.stdout, "encoding", "ascii") or "ascii"
        try:
            "\u2588\u2591".encode(enc)
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
            skip_chr = "\xd7"
        except (UnicodeEncodeError, LookupError):
            bar = "#" * filled + "-" * (bar_width - filled)
            skip_chr = "x"

        line = (
            f"\r[{bar}] {pct * 100:5.1f}%"
            f"  {_fmt_time(video_time)} / {_fmt_time(video_duration)}"
            f"  skip {skip_chr}{frame_skip}"
            f"  Elapsed {_fmt_time(elapsed)}"
            f"  ETA {_fmt_time(eta)}"
            f"  {total_events} event{'s' if total_events != 1 else ''}"
        )
        print(line, end="", flush=True)

    return callback


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Analyze Tekken 8 videos and track UI elements'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Local video file path or a YouTube URL (watch, shorts, youtu.be)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output)'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=15,
        help='Process every Nth frame (default: 15 — ~4 checks/sec at 60 fps, '
             '~7 checks across the ~110-frame popup window; use 6 for higher precision)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='OCR confidence threshold (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualization charts'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU only (default: use GPU when available)'
    )
    parser.add_argument(
        '--cooldown',
        type=float,
        default=2.0,
        metavar='SECONDS',
        help='Ignore same UI element for N seconds after it appears (default: 2.0)'
    )
    parser.add_argument(
        '--full-frame',
        action='store_true',
        help='Scan full frame instead of center hit zone (more detections, more false positives)'
    )
    parser.add_argument(
        '--roboflow',
        type=str,
        metavar='MODEL_ID',
        default=None,
        help='Use Roboflow model instead of OCR (e.g. "my-workspace/tekken8-ui/1"). Requires ROBOFLOW_API_KEY.'
    )
    parser.add_argument(
        '--yolo',
        type=str,
        metavar='WEIGHTS',
        default=None,
        help='Use a local YOLOv8 model instead of OCR (e.g. "runs/detect/tekken8/weights/best.pt").'
    )
    parser.add_argument(
        '--no-mongo',
        action='store_true',
        help='Do not save to MongoDB even if MONGO_URI is set in .env'
    )
    parser.add_argument(
        '--require-match',
        action='store_true',
        help='When using a YouTube URL, exit with error if no document is found in matches for that videoUrl'
    )
    parser.add_argument(
        '--youtube-download-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Save YouTube downloads here (default: temp folder, deleted after unless --keep-youtube-download)'
    )
    parser.add_argument(
        '--keep-youtube-download',
        action='store_true',
        help='Keep downloaded YouTube file (only applies when using default temp download folder)'
    )
    parser.add_argument(
        '--cookies',
        type=str,
        default=None,
        metavar='FILE',
        help='Netscape cookies.txt for yt-dlp (YouTube bot checks). Overrides YTDLP_COOKIE_FILE.'
    )
    parser.add_argument(
        '--cookies-from-browser',
        type=str,
        default=None,
        metavar='BROWSER',
        help='Browser name for yt-dlp (e.g. chrome, firefox, edge, brave). Optional profile: chrome:Default. '
             'Overrides YTDLP_COOKIES_FROM_BROWSER.'
    )
    parser.add_argument(
        '--yt-dlp-js-runtimes',
        nargs='*',
        default=None,
        metavar='SPEC',
        help='YouTube n-challenge: JS runtimes for yt-dlp (e.g. node, deno, or node:C:\\path\\to\\node.exe). '
             'Default is Deno only. Overrides YTDLP_JS_RUNTIMES.'
    )
    parser.add_argument(
        '--yt-dlp-remote-components',
        nargs='*',
        default=None,
        metavar='COMPONENT',
        help='Allow yt-dlp to fetch EJS assets (e.g. ejs:github). Overrides YTDLP_REMOTE_COMPONENTS.'
    )

    args = parser.parse_args()

    youtube_url = None
    yt_info = {}
    cleanup_download_dir = None
    raw_input = args.video_path.strip()

    if is_youtube_url(raw_input):
        youtube_url = raw_input
        dl_dir = Path(args.youtube_download_dir) if args.youtube_download_dir else None
        cookiefile = (args.cookies or os.environ.get("YTDLP_COOKIE_FILE") or "").strip() or None
        cookies_browser = (
            args.cookies_from_browser or os.environ.get("YTDLP_COOKIES_FROM_BROWSER") or ""
        ).strip() or None
        yt_js_runtimes = _resolve_yt_dlp_js_runtimes(args.yt_dlp_js_runtimes)
        yt_remote = _resolve_yt_dlp_remote_components(args.yt_dlp_remote_components)
        print("YouTube URL detected — downloading video (this may take a while)...")
        try:
            video_path, yt_info = download_youtube(
                youtube_url,
                download_dir=dl_dir,
                cookiefile=cookiefile,
                cookies_from_browser=cookies_browser,
                js_runtimes=yt_js_runtimes,
                remote_components=yt_remote,
            )
        except ImportError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: YouTube download failed: {e}")
            err = str(e).lower()
            if "not a bot" in err or "sign in" in err:
                print(
                    "Hint: pass cookies — e.g. --cookies-from-browser chrome "
                    "or set YTDLP_COOKIES_FROM_BROWSER / YTDLP_COOKIE_FILE in .env"
                )
            elif "cookie database" in err or "could not copy" in err:
                print(
                    "Hint (Windows / Chrome & Edge): the browser locks its cookie file while open. "
                    "Fully quit Chrome/Edge (check Task Manager for background processes), then retry; "
                    "or use --cookies-from-browser firefox if you're logged into YouTube there; "
                    "or export Netscape cookies.txt and use --cookies. "
                    "See https://github.com/yt-dlp/yt-dlp/issues/7271"
                )
            elif "dpapi" in err:
                print(
                    "Hint (Chrome on Windows): yt-dlp could not decrypt Chrome's cookies (DPAPI). "
                    "Try --cookies-from-browser firefox, or export Netscape cookies.txt from Chrome "
                    "(e.g. \"Get cookies.txt LOCALLY\" extension) and use --cookies; run the same Windows "
                    "user as the browser (not mixed admin/elevated). pip install -U yt-dlp. "
                    "See https://github.com/yt-dlp/yt-dlp/issues/10927"
                )
            elif (
                "challenge" in err
                or "n challenge" in err
                or "only images are available" in err
                or ("format is not available" in err and "requested format" in err)
            ):
                print(
                    "Hint (YouTube JS challenge): install a JS runtime yt-dlp can use — Deno 2+ (recommended) "
                    "or Node.js 20+, then: pip install -U \"yt-dlp[default]\". "
                    "If you have Node but not Deno: --yt-dlp-js-runtimes node (or YTDLP_JS_RUNTIMES=node in .env). "
                    "If scripts are missing: pip install yt-dlp-ejs or --yt-dlp-remote-components ejs:github. "
                    "See https://github.com/yt-dlp/yt-dlp/wiki/EJS"
                )
            sys.exit(1)
        if dl_dir is None:
            cleanup_download_dir = video_path.parent
        print(f"Downloaded to: {video_path}\n")
    else:
        video_path = Path(raw_input)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Tekken 8 Video Analyzer")
    print("="*60)
    print(f"Video: {video_path}" + (f"\nSource: {youtube_url}" if youtube_url else ""))
    print(f"Output directory: {output_dir}")
    checks_per_sec = 60 // args.frame_skip
    print(f"Frame skip: every {args.frame_skip} frame(s) (~{checks_per_sec} checks/sec at 60 fps)")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Device: {'CPU (forced)' if args.cpu else 'GPU if available'}")
    print(f"Event cooldown: {args.cooldown}s")
    print(f"Scan: {'full frame' if args.full_frame else 'center hit zone only'}")
    if args.yolo:
        detector_label = f"YOLO ({args.yolo})"
    elif args.roboflow:
        detector_label = f"Roboflow ({args.roboflow})"
    else:
        detector_label = "OCR (EasyOCR)"
    print(f"Detector: {detector_label}")
    print("="*60 + "\n")

    try:
        run_start = time.time()

        # Initialize detector (YOLO, Roboflow, or OCR)
        print("Initializing detector...")
        if args.yolo:
            from yolo_detector import YOLODetector
            detector = YOLODetector(
                weights_path=args.yolo,
                confidence=args.confidence,
            )
        elif args.roboflow:
            from roboflow_detector import RoboflowDetector
            detector = RoboflowDetector(
                model_id=args.roboflow,
                confidence=args.confidence
            )
        else:
            detector = UIDetector(
                confidence_threshold=args.confidence,
                use_gpu=not args.cpu
            )

        # Initialize processor
        processor = VideoProcessor(
            detector,
            frame_skip=args.frame_skip,
            cooldown_seconds=args.cooldown,
            full_frame=args.full_frame
        )

        # Process video
        print("\nProcessing video...")
        analysis_data = processor.process_video(
            str(video_path),
            progress_callback=make_progress_callback(args.frame_skip)
        )

        print(f"\n\nVideo processing complete!")

        elapsed = time.time() - run_start
        mm, ss = divmod(elapsed, 60)
        analysis_data["video_info"]["task_duration_seconds"] = round(elapsed, 1)
        analysis_data["video_info"]["task_duration"] = f"{int(mm)}m {ss:.1f}s"
        if youtube_url:
            analysis_data["video_info"]["source"] = "youtube"
            analysis_data["video_info"]["youtube_url"] = youtube_url
            if yt_info.get("id"):
                analysis_data["video_info"]["youtube_id"] = yt_info["id"]
            if yt_info.get("title"):
                analysis_data["video_info"]["youtube_title"] = yt_info["title"]

        # Generate statistics
        print("\nGenerating statistics...")
        stats_gen = StatsGenerator(analysis_data)

        stats_gen.print_summary()

        # Export data
        print("Exporting data...")
        stats_gen.export_json(output_dir / "stats.json")
        stats_gen.export_csv(output_dir / "detections.csv")

        if not args.no_mongo:
            mongo_uri = os.environ.get("MONGO_URI", "").strip()
            if mongo_uri:
                print("Saving to MongoDB...")
                from mongo_store import save_analysis
                if args.yolo:
                    det, det_detail = "yolo", args.yolo
                elif args.roboflow:
                    det, det_detail = "roboflow", args.roboflow
                else:
                    det, det_detail = "ocr", None
                try:
                    from fighters_edge_format import export_fighters_edge_csv
                    mongo_result = save_analysis(
                        stats_gen.to_dict(),
                        video_path=str(video_path),
                        detector=det,
                        detector_detail=det_detail,
                        video_url=youtube_url,
                        require_match=args.require_match,
                    )
                    if mongo_result:
                        print(
                            f"MongoDB: inserted id {mongo_result['inserted_id']}"
                            f" | MatchId={mongo_result['match_id']!r} "
                            f"({mongo_result['match_type']})"
                            f" | lookup={'ok' if mongo_result['match_found'] else 'no match'}"
                        )
                        export_fighters_edge_csv(
                            output_dir / "fighters_edge_analyses.csv",
                            row_id=mongo_result["inserted_id"],
                            match_type=mongo_result["match_type"],
                            match_id=mongo_result["match_id"],
                            detections=mongo_result["detections"],
                        )
                        print(f"Fighters-Edge CSV: {output_dir / 'fighters_edge_analyses.csv'}")
                except Exception as mongo_err:
                    print(f"MongoDB save failed: {mongo_err}")
            else:
                print(
                    "Skipping MongoDB: MONGO_URI is not set. "
                    "Add it to .env next to main.py (see .env.example) or export it in your shell."
                )

        # Create visualizations
        if not args.no_visualizations:
            print("\nGenerating visualizations...")
            stats_gen.create_visualizations(str(output_dir))
            print(f"Visualizations saved to {output_dir}")

        print(f"\nAnalysis complete! Check {output_dir} for results.")
        print(f"Total time: {analysis_data['video_info']['task_duration']}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if cleanup_download_dir and not args.keep_youtube_download:
            shutil.rmtree(cleanup_download_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
