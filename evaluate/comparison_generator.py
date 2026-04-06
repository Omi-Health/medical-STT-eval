#!/usr/bin/env python3
"""
Comparison generator for cross-model evaluation.
Creates comparative analysis across all evaluated models.
Includes Medical WER (M-WER) and Drug M-WER from medical_wer.py output.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Models evaluated with different metrics (cpWER instead of WER)
MULTISPEAKER_MODELS = {"multitalker-parakeet-streaming-0.6b-v1"}


class ModelComparison:
    """Generate comparisons across multiple models."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.comparisons_dir = self.results_dir / "comparisons"
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)

    def load_model_metrics(self) -> Dict[str, Dict]:
        """Load WER + Medical WER metrics for all models."""
        if not self.metrics_dir.exists():
            raise FileNotFoundError(f"Metrics directory not found: {self.metrics_dir}")

        models = {}
        wer_files = list(self.metrics_dir.glob("*_wer.json"))

        if not wer_files:
            raise FileNotFoundError(f"No model metrics found in: {self.metrics_dir}")

        for wer_file in wer_files:
            # Skip medical_wer files (loaded separately)
            if "_medical_wer.json" in wer_file.name:
                continue

            model_name = wer_file.stem.replace('_wer', '')

            # Skip multispeaker models from main leaderboard
            if model_name in MULTISPEAKER_MODELS:
                continue

            try:
                with open(wer_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load matching medical WER if available
                mwer_file = self.metrics_dir / f"{model_name}_medical_wer.json"
                if mwer_file.exists():
                    with open(mwer_file, 'r', encoding='utf-8') as f:
                        mwer_data = json.load(f)
                    data['medical'] = mwer_data

                models[model_name] = data
                print(f"  Loaded: {model_name}" + (" (+M-WER)" if mwer_file.exists() else ""))
            except Exception as e:
                print(f"  Error loading {wer_file}: {e}")
                continue

        if not models:
            raise ValueError("No valid model metrics loaded")

        return models

    def generate_summary_comparison(self, models: Dict[str, Dict]) -> Dict:
        """Generate high-level comparison summary."""
        problematic_files = [
            'day1_consultation07_conversation_transcript.txt',
            'day3_consultation03_conversation_transcript.txt'
        ]

        comparison = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'models_compared': len(models),
            'summary': []
        }

        for model_name, metrics in models.items():
            summary = {'model_name': model_name}

            if 'file_details' in metrics:
                valid_files = [f for f in metrics['file_details']
                             if not any(f['file'].endswith(pf) for pf in problematic_files)]

                if valid_files:
                    import statistics
                    wers = [f['wer'] for f in valid_files if f.get('wer') is not None]
                    if wers:
                        summary.update({
                            'files_processed': len(valid_files),
                            'average_wer': statistics.mean(wers),
                            'best_wer': min(wers),
                            'worst_wer': max(wers),
                            'wer_std': statistics.stdev(wers) if len(wers) > 1 else 0,
                            'word_accuracy': 1.0 - statistics.mean(wers),
                            'total_reference_words': sum(f.get('reference_words', 0) for f in valid_files),
                        })
            else:
                summary['files_processed'] = metrics.get('file_count', 0)
                if 'wer_metrics' in metrics:
                    wer = metrics['wer_metrics']
                    summary.update({
                        'average_wer': wer.get('average_wer', 0),
                        'best_wer': wer.get('best_wer', 0),
                        'worst_wer': wer.get('worst_wer', 0),
                        'wer_std': wer.get('wer_std', 0),
                    })
                if 'word_statistics' in metrics:
                    summary.update({
                        'word_accuracy': metrics['word_statistics'].get('word_accuracy', 0),
                        'total_reference_words': metrics['word_statistics'].get('total_reference_words', 0),
                    })

            # Speed
            if 'speed_metrics' in metrics:
                speed = metrics['speed_metrics']
                summary['average_duration'] = speed.get('average_duration', 0)

            # Medical WER
            if 'medical' in metrics:
                med = metrics['medical']
                summary['m_wer'] = med.get('global_m_wer', med.get('avg_m_wer', 0))
                summary['drug_m_wer'] = med.get('global_high_risk_m_wer', 0)
                # Fallback: get drug from per_category
                if summary['drug_m_wer'] == 0 and 'global_per_category' in med:
                    drugs = med['global_per_category'].get('drugs', {})
                    summary['drug_m_wer'] = drugs.get('m_wer', 0)

            comparison['summary'].append(summary)

        # Sort by M-WER (primary), fall back to WER
        comparison['summary'].sort(
            key=lambda x: x.get('m_wer', x.get('average_wer', float('inf')))
        )

        return comparison

    def generate_detailed_comparison(self, models: Dict[str, Dict]) -> Dict:
        """Generate detailed file-by-file comparison."""
        all_files = set()
        for metrics in models.values():
            if 'file_details' in metrics:
                for file_detail in metrics['file_details']:
                    all_files.add(file_detail['file'])

        problematic_files = [
            'day1_consultation07_conversation_transcript.txt',
            'day3_consultation03_conversation_transcript.txt'
        ]
        all_files = sorted(f for f in all_files if not any(f.endswith(pf) for pf in problematic_files))

        detailed_comparison = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'models': list(models.keys()),
            'files': []
        }

        for file_name in all_files:
            file_comparison = {'file': file_name, 'models': {}}
            for model_name, metrics in models.items():
                file_metrics = None
                if 'file_details' in metrics:
                    file_metrics = next((f for f in metrics['file_details'] if f['file'].endswith(file_name)), None)
                if file_metrics:
                    file_comparison['models'][model_name] = {
                        'wer': file_metrics.get('wer'),
                        'reference_words': file_metrics.get('reference_words'),
                        'hypothesis_words': file_metrics.get('hypothesis_words'),
                    }
                else:
                    file_comparison['models'][model_name] = {'wer': None}
            detailed_comparison['files'].append(file_comparison)

        return detailed_comparison

    def generate_all_comparisons(self) -> Dict[str, str]:
        """Generate all comparison files."""
        print("Loading model metrics...")
        try:
            models = self.load_model_metrics()
        except Exception as e:
            print(f"Error: {e}")
            return {}

        print(f"Loaded {len(models)} models\n")
        output_files = {}

        # Generate summary
        print("Generating leaderboard...")
        summary_comparison = self.generate_summary_comparison(models)

        # Create leaderboard.json
        leaderboard = {
            'updated': time.strftime("%Y-%m-%d"),
            'dataset': {
                'name': 'PriMock57',
                'files': 55,
                'total_words': 81236,
                'excluded_files': ['day1_consultation07', 'day3_consultation03']
            },
            'models': []
        }

        for i, model in enumerate(summary_comparison['summary'], 1):
            entry = {
                'rank': i,
                'name': model['model_name'],
                'wer': round(model.get('average_wer', 0), 4),
                'accuracy': round(model.get('word_accuracy', 0), 4),
                'avg_speed_sec': round(model.get('average_duration', 0), 1),
                'best_wer': round(model.get('best_wer', 0), 4),
                'worst_wer': round(model.get('worst_wer', 0), 4),
                'wer_std': round(model.get('wer_std', 0), 4),
                'files_evaluated': model.get('files_processed', 55),
                'm_wer': round(model.get('m_wer', 0), 4),
                'drug_m_wer': round(model.get('drug_m_wer', 0), 4),
            }
            if entry['files_evaluated'] < 55:
                entry['note'] = f"*{entry['files_evaluated']}/55 files evaluated"
            leaderboard['models'].append(entry)

        leaderboard_file = self.comparisons_dir / "leaderboard.json"
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard, f, indent=2, ensure_ascii=False)
        output_files['leaderboard'] = str(leaderboard_file)

        # Generate per-file results
        print("Generating per-file results...")
        detailed_comparison = self.generate_detailed_comparison(models)
        per_file_file = self.comparisons_dir / "per_file_results.json"
        with open(per_file_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_comparison, f, indent=2, ensure_ascii=False)
        output_files['per_file'] = str(per_file_file)

        # Print summary
        self.print_comparison_summary(summary_comparison)

        return output_files

    def print_comparison_summary(self, summary_comparison: Dict):
        """Print comparison summary to console."""
        if not summary_comparison['summary']:
            print("No models to compare")
            return

        print(f"\n{'='*90}")
        print(f"MODEL LEADERBOARD — Ranked by M-WER ({len(summary_comparison['summary'])} models)")
        print(f"{'='*90}")
        print(f"{'#':<4} {'Model':<45} {'WER':>6} {'M-WER':>7} {'Drug':>7}")
        print(f"{'-'*90}")

        for i, model in enumerate(summary_comparison['summary'], 1):
            wer = model.get('average_wer', 0)
            mwer = model.get('m_wer', 0)
            drug = model.get('drug_m_wer', 0)
            print(f"{i:<4} {model['model_name']:<45} {wer*100:5.2f}% {mwer*100:6.2f}% {drug*100:6.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Model Comparisons")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    args = parser.parse_args()

    comparator = ModelComparison(results_dir=args.results_dir)
    try:
        output_files = comparator.generate_all_comparisons()
    except Exception as e:
        print(f"Error: {e}")
        return

    if output_files:
        print(f"\nFiles generated:")
        for comp_type, file_path in output_files.items():
            print(f"  {comp_type}: {file_path}")


if __name__ == "__main__":
    main()
