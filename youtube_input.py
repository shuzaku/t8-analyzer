"""
Download YouTube videos for offline analysis (yt-dlp).
"""

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


_YOUTUBE_HOST = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?|shorts/|embed/)|youtu\.be/|m\.youtube\.com/)",
    re.IGNORECASE,
)


def is_youtube_url(text: str) -> bool:
    """True if the string looks like a YouTube watch / shorts / youtu.be URL."""
    t = text.strip()
    if not t.startswith(("http://", "https://")):
        return False
    return bool(_YOUTUBE_HOST.search(t))


def parse_js_runtimes_specs(specs: List[str]) -> Dict[str, dict]:
    """
    Build a YoutubeDL ``js_runtimes`` dict.

    Each spec is ``node``, ``deno``, ``bun``, ``quickjs``, or ``runtime:path/to/exe``.
    """
    out: Dict[str, dict] = {}
    for raw in specs:
        s = raw.strip()
        if not s:
            continue
        if ":" in s:
            name, path = s.split(":", 1)
            out[name.strip().lower()] = {"path": path.strip()}
        else:
            out[s.lower()] = {}
    return out


def parse_remote_components_specs(specs: List[str]) -> List[str]:
    """Normalize ``--remote-components``-style tokens (e.g. ejs:github)."""
    return [s.strip() for s in specs if s.strip()]


def _cookiesfrombrowser_tuple(spec: str) -> Tuple:
    """
    Build yt-dlp cookiesfrombrowser tuple from a short string.

    Examples: "chrome", "firefox", "edge", "brave", "chrome:Profile 1"
    """
    s = spec.strip()
    if ":" in s:
        browser, profile = s.split(":", 1)
        return (browser.strip(), profile.strip())
    return (s,)


def download_youtube(
    url: str,
    *,
    download_dir: Optional[Path] = None,
    format_selector: str = "bv*+ba/b",
    cookiefile: Optional[Union[str, Path]] = None,
    cookies_from_browser: Optional[str] = None,
    js_runtimes: Optional[Dict[str, dict]] = None,
    remote_components: Optional[List[str]] = None,
) -> Tuple[Path, dict]:
    """
    Download a YouTube video to disk and return the path to the media file.

    Args:
        url: Full YouTube URL
        download_dir: Folder to save into (default: system temp directory)
        format_selector: yt-dlp -f string (default: best video+audio merged to mp4 when possible)
        cookiefile: Path to Netscape-format cookies.txt (see yt-dlp wiki on exporting YouTube cookies)
        cookies_from_browser: e.g. "chrome", "firefox", "edge", "brave", or "chrome:Profile Name"
        js_runtimes: YoutubeDL js_runtimes dict (e.g. {"node": {}}). None = yt-dlp default (Deno).
        remote_components: e.g. ["ejs:github"] if challenge scripts must be fetched remotely.

    Returns:
        (path_to_video_file, info_dict) with keys like id, title, duration from yt-dlp
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "YouTube support requires yt-dlp. Install with: pip install yt-dlp"
        )

    if download_dir is None:
        download_dir = Path(tempfile.mkdtemp(prefix="t8_analyzer_yt_"))
    else:
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

    final_path: Optional[Path] = None

    def hook(d: dict) -> None:
        nonlocal final_path
        if d.get("status") == "finished":
            name = d.get("filename")
            if name:
                final_path = Path(name)

    ydl_opts = {
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "format": format_selector,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [hook],
    }
    if cookiefile:
        p = Path(cookiefile).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Cookie file not found: {p}")
        ydl_opts["cookiefile"] = str(p)
    elif cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = _cookiesfrombrowser_tuple(cookies_from_browser)

    if js_runtimes is not None:
        ydl_opts["js_runtimes"] = js_runtimes
    if remote_components:
        ydl_opts["remote_components"] = remote_components

    info: dict = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if final_path is None and info:
            try:
                final_path = Path(ydl.prepare_filename(info))
            except Exception:
                pass

    if final_path is None or not final_path.exists():
        candidates = sorted(
            download_dir.glob("*"),
            key=lambda p: p.stat().st_mtime if p.is_file() else 0,
            reverse=True,
        )
        for p in candidates:
            if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}:
                final_path = p
                break

    if final_path is None or not final_path.exists():
        raise RuntimeError("yt-dlp finished but the output file could not be found.")

    return final_path.resolve(), info or {}
