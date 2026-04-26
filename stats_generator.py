"""
Statistics generator for Tekken 8 video analysis.
Creates per-player statistics and visualizations from detected UI elements.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json
from pathlib import Path


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as M:SS.mmm (e.g. 0:00.333, 1:05.500)."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"


class StatsGenerator:
    """Generates per-player statistics and reports from video analysis."""

    def __init__(self, analysis_data: Dict):
        """
        Initialize stats generator.

        Args:
            analysis_data: Dictionary from VideoProcessor containing:
                - stats: {"P1": {element: count}, "P2": {element: count}}
                - frame_detections: [{frame, time, player, element}, ...]
                - video_info: {total_frames, processed_frames, fps, duration, resolution}
        """
        self.stats: Dict[str, Dict[str, int]] = analysis_data["stats"]
        self.frame_detections: List[Dict] = analysis_data["frame_detections"]
        self.video_info: Dict = analysis_data["video_info"]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(self) -> Dict:
        """Return summary statistics broken down by player."""
        duration = self.video_info["duration"]

        combined_counts: Dict[str, int] = {}
        for player_stats in self.stats.values():
            for element, count in player_stats.items():
                combined_counts[element] = combined_counts.get(element, 0) + count

        total_events = sum(combined_counts.values())

        summary = {
            "total_events": total_events,
            "video_duration_seconds": round(duration, 3),
            "video_duration_timestamp": seconds_to_timestamp(duration),
            "video_duration_minutes": round(duration / 60, 3),
            "events_per_minute": round(total_events / duration * 60, 2) if duration > 0 else 0,
            "unique_event_types": len(combined_counts),
            "player_stats": {p: dict(s) for p, s in self.stats.items()},
        }
        if "task_duration_seconds" in self.video_info:
            summary["task_duration_seconds"] = self.video_info["task_duration_seconds"]
            summary["task_duration"] = self.video_info["task_duration"]
        return summary

    # ------------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------------

    def generate_dataframe(self) -> pd.DataFrame:
        """Return a flat DataFrame with one row per detected event."""
        if not self.frame_detections:
            return pd.DataFrame()

        rows = []
        for event in self.frame_detections:
            rows.append({
                "frame": event["frame"],
                "time_seconds": round(event["time"], 3),
                "timestamp": seconds_to_timestamp(event["time"]),
                "player": event["player"],
                "element": event["element"],
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def create_visualizations(self, output_dir: str = "output"):
        """Create and save visualization charts to output_dir."""
        Path(output_dir).mkdir(exist_ok=True)

        all_elements = sorted(
            set(e for p_stats in self.stats.values() for e in p_stats)
        )

        if not all_elements:
            print("No events detected; skipping visualizations.")
            return

        p1_counts = [self.stats.get("P1", {}).get(e, 0) for e in all_elements]
        p2_counts = [self.stats.get("P2", {}).get(e, 0) for e in all_elements]

        # 1. Grouped event count bar chart
        x = np.arange(len(all_elements))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(all_elements) * 1.5), 6))
        ax.bar(x - bar_width / 2, p1_counts, bar_width, label="P1", color="steelblue", edgecolor="black")
        ax.bar(x + bar_width / 2, p2_counts, bar_width, label="P2", color="tomato", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(all_elements, rotation=45, ha="right")
        ax.set_xlabel("UI Element", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Tekken 8 — UI Element Counts (P1 vs P2)", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/event_counts.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2. Timeline
        df = self.generate_dataframe()
        if not df.empty:
            fig, ax = plt.subplots(figsize=(14, 6))
            colors = {"P1": "steelblue", "P2": "tomato"}
            markers = {"P1": "o", "P2": "s"}

            for element in df["element"].unique():
                for player in ("P1", "P2"):
                    subset = df[(df["element"] == element) & (df["player"] == player)]
                    if not subset.empty:
                        ax.scatter(
                            subset["time_seconds"],
                            [element] * len(subset),
                            color=colors[player],
                            marker=markers[player],
                            alpha=0.7,
                            s=60,
                            label=f"{player} {element}",
                        )

            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_ylabel("UI Element", fontsize=12)
            ax.set_title("UI Element Timeline (P1 vs P2)", fontsize=14, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/timeline.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        # 3. Events per minute
        duration_min = self.video_info["duration"] / 60
        if duration_min > 0:
            p1_rates = [c / duration_min for c in p1_counts]
            p2_rates = [c / duration_min for c in p2_counts]

            fig, ax = plt.subplots(figsize=(max(10, len(all_elements) * 1.5), 6))
            ax.bar(x - bar_width / 2, p1_rates, bar_width, label="P1", color="steelblue", edgecolor="black")
            ax.bar(x + bar_width / 2, p2_rates, bar_width, label="P2", color="tomato", edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(all_elements, rotation=45, ha="right")
            ax.set_xlabel("UI Element", fontsize=12)
            ax.set_ylabel("Events per Minute", fontsize=12)
            ax.set_title("UI Element Frequency — Events per Minute (P1 vs P2)", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/events_per_minute.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return the same structure written to stats.json (for MongoDB, APIs, etc.)."""
        summary = self.generate_summary()
        video_info = {
            **self.video_info,
            "duration_timestamp": seconds_to_timestamp(self.video_info["duration"]),
        }
        frame_detections = [
            {
                "frame": e["frame"],
                "time_seconds": round(e["time"], 3),
                "timestamp": seconds_to_timestamp(e["time"]),
                "player": e["player"],
                "element": e["element"],
            }
            for e in self.frame_detections
        ]
        return {
            "summary": summary,
            "video_info": video_info,
            "player_stats": {p: dict(s) for p, s in self.stats.items()},
            "frame_detections": frame_detections,
        }

    def export_json(self, output_path: str = "output/stats.json"):
        """Export statistics (with per-player breakdown) to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Statistics exported to {output_path}")

    def export_csv(self, output_path: str = "output/detections.csv"):
        """Export frame detections (with player column) to a CSV file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df = self.generate_dataframe()
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f"Detections exported to {output_path}")
        else:
            print("No detections to export.")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print a per-player summary to the console."""
        summary = self.generate_summary()

        print("\n" + "=" * 60)
        print("TEKKEN 8 VIDEO ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\nVideo Duration : {summary['video_duration_timestamp']} ({summary['video_duration_seconds']}s)")
        print(f"Total Events   : {summary['total_events']}")
        print(f"Events/Minute  : {summary['events_per_minute']:.2f}")
        print(f"Unique Types   : {summary['unique_event_types']}")
        if "task_duration" in summary:
            print(f"Task Duration  : {summary['task_duration']}")

        print("\n" + "-" * 60)
        print(f"{'EVENT':<22} {'P1':>6} {'P2':>6}")
        print("-" * 60)

        all_elements = sorted(
            set(e for p_stats in self.stats.values() for e in p_stats)
        )

        if all_elements:
            for element in all_elements:
                p1 = self.stats.get("P1", {}).get(element, 0)
                p2 = self.stats.get("P2", {}).get(element, 0)
                print(f"  {element:<20} {p1:>6} {p2:>6}")
        else:
            print("  No events detected.")

        print("=" * 60 + "\n")
