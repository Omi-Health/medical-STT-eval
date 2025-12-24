#!/usr/bin/env python3
"""
Comparison generator for cross-model evaluation.
Creates comparative analysis across all evaluated models.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ModelComparison:
    """Generate comparisons across multiple models."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.comparisons_dir = self.results_dir / "comparisons"
        
        # Create comparisons directory
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_metrics(self) -> Dict[str, Dict]:
        """Load metrics for all available models."""
        if not self.metrics_dir.exists():
            raise FileNotFoundError(f"Metrics directory not found: {self.metrics_dir}")
        
        models = {}
        
        # Find all WER metric files
        wer_files = list(self.metrics_dir.glob("*_wer.json"))
        
        if not wer_files:
            raise FileNotFoundError(f"No model metrics found in: {self.metrics_dir}")
        
        for wer_file in wer_files:
            model_name = wer_file.stem.replace('_wer', '').replace('_', '/')
            
            try:
                with open(wer_file, 'r', encoding='utf-8') as f:
                    models[model_name] = json.load(f)
                print(f"✅ Loaded metrics for: {model_name}")
            except Exception as e:
                print(f"❌ Error loading {wer_file}: {e}")
                continue
        
        if not models:
            raise ValueError("No valid model metrics loaded")
        
        return models
    
    def generate_summary_comparison(self, models: Dict[str, Dict]) -> Dict:
        """Generate high-level comparison summary."""
        # Define problematic files to exclude
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
            # Recalculate metrics excluding problematic files
            if 'file_details' in metrics:
                # Filter out problematic files
                valid_files = [f for f in metrics['file_details'] 
                             if not any(f['file'].endswith(pf) for pf in problematic_files)]
                
                if valid_files:
                    # Recalculate WER metrics
                    wers = [f['wer'] for f in valid_files if f.get('wer') is not None]
                    if wers:
                        import statistics
                        average_wer = statistics.mean(wers)
                        wer_std = statistics.stdev(wers) if len(wers) > 1 else 0
                        best_wer = min(wers)
                        worst_wer = max(wers)
                        
                        # Recalculate word statistics
                        total_ref_words = sum(f.get('reference_words', 0) for f in valid_files)
                        total_hyp_words = sum(f.get('hypothesis_words', 0) for f in valid_files)
                        
                        # Calculate word accuracy from WER
                        word_accuracy = 1.0 - average_wer if average_wer < 1.0 else 0.0
                        
                        summary = {
                            'model_name': model_name,
                            'files_processed': len(valid_files),
                            'average_wer': average_wer,
                            'best_wer': best_wer,
                            'worst_wer': worst_wer,
                            'wer_std': wer_std,
                            'word_accuracy': word_accuracy,
                            'total_reference_words': total_ref_words
                        }
                        
                        # Add speed metrics if available
                        if 'speed_metrics' in metrics:
                            speed = metrics['speed_metrics']
                            summary.update({
                                'total_duration': speed.get('total_duration', 0),
                                'average_duration': speed.get('average_duration', 0),
                                'words_per_second': speed.get('words_per_second', 0)
                            })
                        
                        comparison['summary'].append(summary)
            else:
                # Fallback to original metrics if file_details not available
                summary = {
                    'model_name': model_name,
                    'files_processed': metrics.get('file_count', 0)
                }
                
                # WER metrics
                if 'wer_metrics' in metrics:
                    wer = metrics['wer_metrics']
                    summary.update({
                        'average_wer': wer.get('average_wer', 0),
                        'best_wer': wer.get('best_wer', 0),
                        'worst_wer': wer.get('worst_wer', 0),
                        'wer_std': wer.get('wer_std', 0)
                    })
                
                # Word statistics
                if 'word_statistics' in metrics:
                    words = metrics['word_statistics']
                    summary.update({
                        'word_accuracy': words.get('word_accuracy', 0),
                        'total_reference_words': words.get('total_reference_words', 0)
                    })
                
                # Speed metrics
                if 'speed_metrics' in metrics:
                    speed = metrics['speed_metrics']
                    summary.update({
                        'total_duration': speed.get('total_duration', 0),
                        'average_duration': speed.get('average_duration', 0),
                        'words_per_second': speed.get('words_per_second', 0)
                    })
                
                comparison['summary'].append(summary)
        
        # Sort by average WER (best first)
        comparison['summary'].sort(key=lambda x: x.get('average_wer', float('inf')))
        
        return comparison
    
    def generate_detailed_comparison(self, models: Dict[str, Dict]) -> Dict:
        """Generate detailed file-by-file comparison."""
        # Get all unique files across models
        all_files = set()
        for metrics in models.values():
            if 'file_details' in metrics:
                for file_detail in metrics['file_details']:
                    all_files.add(file_detail['file'])
        
        all_files = sorted(all_files)
        
        # Filter out files that don't have valid WER for most models
        # Specifically exclude the two problematic files
        problematic_files = [
            'day1_consultation07_conversation_transcript.txt',
            'day3_consultation03_conversation_transcript.txt'
        ]
        
        all_files = [f for f in all_files if not any(f.endswith(pf) for pf in problematic_files)]
        
        detailed_comparison = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'models': list(models.keys()),
            'files': []
        }
        
        for file_name in all_files:
            file_comparison = {
                'file': file_name,
                'models': {}
            }
            
            for model_name, metrics in models.items():
                # Find this file's metrics
                file_metrics = None
                if 'file_details' in metrics:
                    file_metrics = next((f for f in metrics['file_details'] if f['file'].endswith(file_name)), None)
                
                if file_metrics:
                    file_comparison['models'][model_name] = {
                        'wer': file_metrics.get('wer', None),
                        'reference_words': file_metrics.get('reference_words', None),
                        'hypothesis_words': file_metrics.get('hypothesis_words', None),
                        'transcription_duration': file_metrics.get('transcription_duration', None)
                    }
                else:
                    file_comparison['models'][model_name] = {
                        'wer': None,
                        'reference_words': None,
                        'hypothesis_words': None,
                        'transcription_duration': None
                    }
            
            detailed_comparison['files'].append(file_comparison)
        
        return detailed_comparison
    
    def create_csv_comparison(self, summary_comparison: Dict) -> str:
        """Create CSV comparison table."""
        csv_file = self.comparisons_dir / "model_comparison.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if not summary_comparison['summary']:
                return str(csv_file)
            
            # Get all available columns
            all_columns = set()
            for model in summary_comparison['summary']:
                all_columns.update(model.keys())
            
            # Order columns logically
            ordered_columns = ['model_name', 'files_processed', 'average_wer', 'word_accuracy', 
                             'best_wer', 'worst_wer', 'wer_std', 'total_reference_words',
                             'total_duration', 'average_duration', 'words_per_second']
            
            # Add any remaining columns
            columns = [col for col in ordered_columns if col in all_columns]
            columns.extend([col for col in all_columns if col not in columns])
            
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for model in summary_comparison['summary']:
                # Fill missing values with None
                row = {col: model.get(col, None) for col in columns}
                writer.writerow(row)
        
        return str(csv_file)
    
    def create_ranking_table(self, summary_comparison: Dict) -> Dict:
        """Create ranking table for different metrics."""
        if not summary_comparison['summary']:
            return {}
        
        models = summary_comparison['summary']
        
        rankings = {
            'by_wer': sorted(models, key=lambda x: x.get('average_wer', float('inf'))),
            'by_accuracy': sorted(models, key=lambda x: x.get('word_accuracy', 0), reverse=True),
            'by_speed': sorted(models, key=lambda x: x.get('average_duration', float('inf')))
        }
        
        # Add rank numbers
        for metric, ranked_models in rankings.items():
            for i, model in enumerate(ranked_models, 1):
                model[f'{metric}_rank'] = i
        
        return {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'rankings': rankings,
            'best_overall': {
                'lowest_wer': rankings['by_wer'][0] if rankings['by_wer'] else None,
                'highest_accuracy': rankings['by_accuracy'][0] if rankings['by_accuracy'] else None,
                'fastest_speed': rankings['by_speed'][0] if rankings['by_speed'] else None
            }
        }
    
    def generate_all_comparisons(self) -> Dict[str, str]:
        """Generate all comparison files."""
        print("📊 Loading model metrics...")

        try:
            models = self.load_model_metrics()
        except Exception as e:
            print(f"❌ Error loading model metrics: {e}")
            return {}

        print(f"✅ Loaded metrics for {len(models)} models")

        output_files = {}

        # Generate summary comparison (internal use)
        print("📈 Generating leaderboard...")
        summary_comparison = self.generate_summary_comparison(models)

        # Create leaderboard.json in consolidated format
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
            files_processed = model.get('files_processed', 55)
            model_entry = {
                'rank': i,
                'name': model['model_name'],
                'wer': round(model.get('average_wer', 0), 4),
                'accuracy': round(model.get('word_accuracy', 0), 4),
                'avg_speed_sec': round(model.get('average_duration', 0), 1),
                'best_wer': round(model.get('best_wer', 0), 4),
                'worst_wer': round(model.get('worst_wer', 0), 4),
                'wer_std': round(model.get('wer_std', 0), 4),
                'files_evaluated': files_processed
            }
            # Add note for incomplete evaluations
            if files_processed < 55:
                model_entry['note'] = f"*{files_processed}/55 files evaluated"
            leaderboard['models'].append(model_entry)

        leaderboard_file = self.comparisons_dir / "leaderboard.json"
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard, f, indent=2, ensure_ascii=False)
        output_files['leaderboard'] = str(leaderboard_file)

        # Generate per-file results
        print("📋 Generating per-file results...")
        detailed_comparison = self.generate_detailed_comparison(models)
        per_file_file = self.comparisons_dir / "per_file_results.json"
        with open(per_file_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_comparison, f, indent=2, ensure_ascii=False)
        output_files['per_file'] = str(per_file_file)

        return output_files
    
    def print_comparison_summary(self, summary_comparison: Dict):
        """Print comparison summary to console."""
        if not summary_comparison['summary']:
            print("❌ No models to compare")
            return
        
        print(f"\n🏆 MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Models compared: {len(summary_comparison['summary'])}")
        print(f"Comparison timestamp: {summary_comparison['timestamp']}")
        
        print(f"\n📊 RANKING BY AVERAGE WER (Lower is better):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<30} {'Avg WER':<10} {'Accuracy':<10} {'Files':<8}")
        print("-" * 80)
        
        for i, model in enumerate(summary_comparison['summary'], 1):
            wer = model.get('average_wer', 0)
            acc = model.get('word_accuracy', 0)
            files = model.get('files_processed', 0)
            
            print(f"{i:<4} {model['model_name']:<30} {wer:.4f}    {acc:.4f}    {files:<8}")
        
        # Best performers
        if summary_comparison['summary']:
            best_model = summary_comparison['summary'][0]
            print(f"\n🥇 BEST PERFORMING MODEL:")
            print(f"   Model: {best_model['model_name']}")
            print(f"   Average WER: {best_model.get('average_wer', 0):.4f} ({best_model.get('average_wer', 0)*100:.2f}%)")
            print(f"   Word Accuracy: {best_model.get('word_accuracy', 0):.4f} ({best_model.get('word_accuracy', 0)*100:.2f}%)")
            if 'average_duration' in best_model:
                print(f"   Average Speed: {best_model.get('average_duration', 0):.2f}s per file")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Model Comparisons")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output_dir", help="Output directory (default: results/comparisons)")
    
    args = parser.parse_args()
    
    # Initialize comparison generator
    comparator = ModelComparison(results_dir=args.results_dir)
    
    if args.output_dir:
        comparator.comparisons_dir = Path(args.output_dir)
        comparator.comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all comparisons
    try:
        output_files = comparator.generate_all_comparisons()
    except Exception as e:
        print(f"❌ Error generating comparisons: {e}")
        return
    
    if not output_files:
        print("❌ No comparisons generated")
        return
    
    print(f"\n✅ COMPARISON FILES GENERATED:")
    for comp_type, file_path in output_files.items():
        print(f"   {comp_type.title()}: {file_path}")

    # Load and print leaderboard summary
    if 'leaderboard' in output_files:
        try:
            with open(output_files['leaderboard'], 'r', encoding='utf-8') as f:
                leaderboard = json.load(f)
            print(f"\n🏆 LEADERBOARD ({len(leaderboard['models'])} models)")
            print("-" * 60)
            for model in leaderboard['models'][:5]:
                print(f"  #{model['rank']} {model['name']}: {model['wer']*100:.2f}% WER")
        except Exception as e:
            print(f"❌ Error printing summary: {e}")


if __name__ == "__main__":
    main()