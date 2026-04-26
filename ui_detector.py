"""
UI Element Detector for Tekken 8 videos.
Detects UI elements like "COUNTER HIT", "JUST FRAME", "RAGE ART", etc. and returns
their horizontal screen position so the caller can assign player ownership.

Each entry in UI_ELEMENTS is a (canonical_name, [alias, ...]) pair.  All aliases
for the same event are treated as equivalent — the first alias that matches in the
OCR output wins, and the result is always reported under the canonical name.
"""

# Fix Pillow 10+ compatibility with EasyOCR (ANTIALIAS was removed)
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import cv2
import easyocr
import numpy as np
from typing import Dict, List, Optional, Tuple
import re


class UIDetector:
    """Detects UI elements in Tekken 8 video frames."""

    # Each entry is (canonical_name, [alias, ...]).
    # Rules:
    #   • Longer/more-specific canonicals FIRST — e.g. "HEAT BURST" before "HEAT"
    #     so consumed_words suppresses the shorter form after the longer one fires.
    #   • Aliases within an entry are tried in order; list longer/more-specific ones first.
    #   • The canonical name is what appears in stats.json / detections.csv — it never
    #     changes even when a different alias was matched in the video.
    UI_ELEMENTS: List[Tuple[str, List[str]]] = [
        # --- Hit quality ---
        ("COUNTER HIT",     ["COUNTER HIT", "CH"]),
        ("COUNTER",         ["COUNTER"]),       # standalone counter — must be after COUNTER HIT
        ("PUNISH",          ["PUNISH"]),
        # --- Rage system ---
        ("RAGE ART",        ["RAGE ART"]),
        ("RAGE DRIVE",      ["RAGE DRIVE"]),
        # --- Heat system ---
        ("HEAT BURST",      ["HEAT BURST"]),
        ("HEAT SMASH",      ["HEAT SMASH"]),
        ("HEAT ENGAGER",    ["HEAT ENGAGER"]),
        ("HEAT",            ["HEAT"]),
        # --- Knockdown types ---
        ("WALL SPLAT",      ["WALL SPLAT"]),
        ("WALL BREAK",      ["WALL BREAK"]),
        ("FLOOR BREAK",     ["FLOOR BREAK"]),
        # --- Throw / escape ---
        ("THROW",           ["THROW"]),
        # --- Punish / defensive ---
        ("REVERSAL",        ["REVERSAL"]),
        # --- Tornado ---
        ("TORNADO",         ["TORNADO"]),
        # --- Armour ---
        ("POWER CRUSH",     ["POWER CRUSH"]),
        # --- Just-frame / tight window ---
        ("JUST FRAME",      ["JUST FRAME", "JUST"]),
    ]

    def __init__(self, confidence_threshold: float = 0.5, use_gpu: bool = True):
        """
        Initialize the UI detector.

        Args:
            confidence_threshold: Minimum confidence for text detection (0-1)
            use_gpu: If True, use GPU for OCR when available (requires PyTorch with CUDA)
        """
        gpu = False
        if use_gpu:
            try:
                import torch
                gpu = torch.cuda.is_available()
                if gpu:
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("GPU requested but CUDA not available, using CPU.")
                    if not torch.version.cuda:
                        print("  → PyTorch is CPU-only. To use GPU, reinstall with CUDA:")
                        print("     pip uninstall torch torchvision -y")
                        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
                    else:
                        print("  → Check NVIDIA drivers (run: nvidia-smi)")
            except ImportError:
                print("PyTorch not found; using CPU. Install torch with CUDA for GPU support.")
        else:
            print("Using CPU (--cpu flag).")

        print("Initializing EasyOCR reader... This may take a moment on first run.")
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.confidence_threshold = confidence_threshold

        # Pre-compile one regex per alias.
        # \b word-boundaries prevent "HEAT" matching inside "HEAT BURST" etc.
        self.patterns: List[Tuple[str, List[Tuple[str, re.Pattern]]]] = [
            (
                canonical,
                [
                    (alias, re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE))
                    for alias in aliases
                ],
            )
            for canonical, aliases in self.UI_ELEMENTS
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale + adaptive threshold to sharpen UI text for OCR."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Downscale frames wider than this before OCR.  Tekken 8 popup text is very
    # large at 1080p so 960 px wide is sufficient.  Halving dimensions reduces
    # OCR time ~75% without meaningful accuracy loss.
    _OCR_MAX_WIDTH = 960

    # Expected background hue ranges (OpenCV H: 0–180) used to colour-validate
    # detections.  Tekken 8 UI banners have fixed colours:
    #   COUNTER HIT  → orange/red banner (~H 5-20)
    #   HEAT         → orange/amber      (~H 15-30)
    _ELEMENT_HUE_RANGES: Dict[str, Tuple[int, int]] = {
        "COUNTER HIT": (5, 25),   # reddish-orange COUNTER HIT banner
        "HEAT":        (15, 35),  # amber HEAT activation banner
    }

    # Elements whose in-game banners have dark/semi-transparent backgrounds.
    # A supplemental raw-OCR pass is triggered for these when not already found.
    _DARK_BG_ELEMENTS: set = {"JUST FRAME", "THROW"}

    def _sample_hue(self, frame: np.ndarray, bbox) -> Optional[float]:
        """
        Return the median hue (OpenCV 0–180) of the coloured pixels inside bbox,
        or None when the region contains too few coloured pixels to be reliable.
        """
        x1 = max(0, int(min(p[0] for p in bbox)))
        y1 = max(0, int(min(p[1] for p in bbox)))
        x2 = min(frame.shape[1], int(max(p[0] for p in bbox)))
        y2 = min(frame.shape[0], int(max(p[1] for p in bbox)))
        if x2 <= x1 or y2 <= y1:
            return None
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)
        if int(mask.sum()) < 10:
            return None
        return float(np.median(hsv[:, :, 0][mask]))

    def _ocr_results(self, frame: np.ndarray) -> List[Tuple]:
        """
        Run OCR on a frame (preprocessed first, falling back to raw if nothing found)
        and return all results above the confidence threshold as
        (bbox, text_upper, confidence).
        bbox is in the original (pre-downscale) coordinate space.
        EasyOCR bbox format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].
        """
        h, w = frame.shape[:2]

        if w > self._OCR_MAX_WIDTH:
            scale = self._OCR_MAX_WIDTH / w
            small = cv2.resize(
                frame,
                (self._OCR_MAX_WIDTH, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0
            small = frame

        def _to_kept(results):
            out = []
            for bbox, text, conf in results:
                if conf >= self.confidence_threshold and text.strip():
                    if scale < 1.0:
                        bbox = [[x / scale, y / scale] for x, y in bbox]
                    out.append((bbox, text.strip().upper(), conf))
            return out

        processed = self.preprocess_frame(small)
        kept = _to_kept(self.reader.readtext(processed))

        if len(kept) == 0:
            kept = _to_kept(self.reader.readtext(small))

        return kept

    def _ocr_results_raw(self, frame: np.ndarray) -> List[Tuple]:
        """
        Same as _ocr_results but skips preprocessing — returns raw OCR results only.
        Used as a supplemental pass for banners with dark backgrounds.
        """
        h, w = frame.shape[:2]
        if w > self._OCR_MAX_WIDTH:
            scale = self._OCR_MAX_WIDTH / w
            small = cv2.resize(
                frame,
                (self._OCR_MAX_WIDTH, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0
            small = frame

        results = self.reader.readtext(small)
        out = []
        for bbox, text, conf in results:
            if conf >= self.confidence_threshold and text.strip():
                if scale < 1.0:
                    bbox = [[x / scale, y / scale] for x, y in bbox]
                out.append((bbox, text.strip().upper(), conf))
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_ui_elements(self, frame: np.ndarray) -> List[str]:
        """
        Detect UI elements in a frame.  Returns a list of element names only
        (no position info).  Kept for backwards compatibility.
        """
        return [elem for elem, _ in self.detect_ui_elements_positioned(frame)]

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip OCR noise characters (apostrophes, brackets, etc.), keep A-Z 0-9 and spaces."""
        cleaned = re.sub(r"[^A-Z0-9 ]", " ", text.upper())
        return re.sub(r" {2,}", " ", cleaned).strip()

    def _detect_in_tokens(
        self,
        frame: np.ndarray,
        token_positions: List[Tuple[str, float, list]],
        frame_width: int,
    ) -> List[Tuple[str, float]]:
        """
        Run pattern matching against a pre-built list of (normalised_text, x, bbox)
        OCR tokens and return (canonical_name, x_center_px) pairs.

        Each call has its own consumed_words scope, so calling this separately for
        left-side and right-side tokens lets both players score the same element
        type simultaneously.
        """
        if not token_positions:
            return []

        combined_text = " ".join(t for t, _, _ in token_positions)
        consumed_words: set = set()
        detected: List[Tuple[str, float]] = []

        for canonical, alias_pairs in self.patterns:
            matched_alias: Optional[str] = None
            matched_pattern: Optional[re.Pattern] = None
            for alias, alias_pat in alias_pairs:
                alias_words = alias.split()
                if all(w in consumed_words for w in alias_words):
                    continue
                if alias_pat.search(combined_text):
                    matched_alias = alias
                    matched_pattern = alias_pat
                    break

            if matched_alias is None:
                continue

            alias_words = matched_alias.split()

            full_phrase_matches = [
                (x_px, bbox)
                for token_text, x_px, bbox in token_positions
                if matched_pattern.search(token_text)
            ]

            if full_phrase_matches:
                avg_x = sum(x for x, _ in full_phrase_matches) / len(full_phrase_matches)
                rep_bbox = full_phrase_matches[0][1]
            else:
                first_word_pat = re.compile(
                    r'\b' + re.escape(alias_words[0]) + r'\b', re.IGNORECASE
                )
                first_word_matches = [
                    (x_px, bbox)
                    for token_text, x_px, bbox in token_positions
                    if first_word_pat.search(token_text)
                ]
                if first_word_matches:
                    avg_x = sum(x for x, _ in first_word_matches) / len(first_word_matches)
                    rep_bbox = first_word_matches[0][1]
                else:
                    avg_x = frame_width / 2.0
                    rep_bbox = None

            if rep_bbox is not None and canonical in self._ELEMENT_HUE_RANGES:
                hue = self._sample_hue(frame, rep_bbox)
                if hue is not None:
                    lo, hi = self._ELEMENT_HUE_RANGES[canonical]
                    if not (lo <= hue <= hi):
                        continue

            consumed_words.update(alias_words)
            detected.append((canonical, avg_x))

        return detected

    def detect_ui_elements_positioned(self, frame: np.ndarray) -> List[Tuple[str, float]]:
        """
        Detect UI elements and return their approximate horizontal center as an
        absolute pixel x-coordinate within this frame.

        OCR tokens are split at the frame midpoint and each half is matched
        independently.  This allows the same element type to be reported for
        both players in the same frame simultaneously.

        Returns:
            List of (element_name, x_center_px) tuples.
        """
        frame_width = frame.shape[1]
        frame_mid   = frame_width / 2.0
        ocr_results = self._ocr_results(frame)

        token_positions: List[Tuple[str, float, list]] = []
        for (bbox, text, _conf) in ocr_results:
            x_center = (bbox[0][0] + bbox[2][0]) / 2.0
            clean = self._normalize(text)
            if clean:
                token_positions.append((clean, x_center, bbox))

        left_tokens  = [(t, x, b) for t, x, b in token_positions if x <  frame_mid]
        right_tokens = [(t, x, b) for t, x, b in token_positions if x >= frame_mid]

        detected: List[Tuple[str, float]] = (
            self._detect_in_tokens(frame, left_tokens,  frame_width) +
            self._detect_in_tokens(frame, right_tokens, frame_width)
        )

        # Supplemental pass for dark-background elements (e.g. JUST FRAME, THROW).
        already_detected = {elem for elem, _ in detected}
        missing_dark = self._DARK_BG_ELEMENTS - already_detected
        if missing_dark:
            raw_ocr = self._ocr_results_raw(frame)
            raw_tokens: List[Tuple[str, float, list]] = []
            for (bbox, text, _conf) in raw_ocr:
                x_center = (bbox[0][0] + bbox[2][0]) / 2.0
                clean = self._normalize(text)
                if clean:
                    raw_tokens.append((clean, x_center, bbox))

            raw_left  = [(t, x, b) for t, x, b in raw_tokens if x <  frame_mid]
            raw_right = [(t, x, b) for t, x, b in raw_tokens if x >= frame_mid]

            for canonical, alias_pairs in self.patterns:
                if canonical not in missing_dark:
                    continue
                for side_tokens in (raw_left, raw_right):
                    if not side_tokens:
                        continue
                    side_combined = " ".join(t for t, _, _ in side_tokens)
                    for alias, alias_pat in alias_pairs:
                        if not alias_pat.search(side_combined):
                            continue
                        raw_matches = [
                            (x_px, bbox)
                            for token_text, x_px, bbox in side_tokens
                            if alias_pat.search(token_text)
                        ]
                        if not raw_matches:
                            continue
                        avg_x = sum(x for x, _ in raw_matches) / len(raw_matches)
                        detected.append((canonical, avg_x))
                        break

        return detected

    def detect_in_region(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> List[str]:
        """Detect UI elements in a sub-region.  Returns element names only."""
        return [elem for elem, _ in self.detect_in_region_positioned(frame, region)]

    def detect_in_region_positioned(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> List[Tuple[str, float]]:
        """
        Detect UI elements in a sub-region and return positions in full-frame
        coordinates so the caller can compare directly against mid_x.

        Args:
            frame: Full video frame (BGR)
            region: (x, y, width, height) of the area to scan

        Returns:
            List of (element_name, abs_x_center_px) where abs_x_center_px is
            relative to the full frame (not the cropped ROI).
        """
        rx, ry, rw, rh = region
        roi = frame[ry:ry + rh, rx:rx + rw]
        roi_detections = self.detect_ui_elements_positioned(roi)
        return [(element, rx + x_px) for element, x_px in roi_detections]
