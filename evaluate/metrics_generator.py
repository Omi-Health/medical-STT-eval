#!/usr/bin/env python3
"""
Metrics generator for individual models.
Combines transcription speed and WER metrics with Whisper-style normalization.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Import Whisper normalizer for improved WER calculation
try:
    from whisper.normalizers import EnglishTextNormalizer
    WHISPER_NORMALIZER = EnglishTextNormalizer()
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️  Whisper normalization not available. Installing openai-whisper...")
    import os
    os.system("pip install openai-whisper")
    try:
        from whisper.normalizers import EnglishTextNormalizer
        WHISPER_NORMALIZER = EnglishTextNormalizer()
        WHISPER_AVAILABLE = True
    except ImportError:
        print("❌ Failed to install Whisper normalizer. Using basic normalization.")
        WHISPER_AVAILABLE = False

from wer_calculator import WERCalculator, calculate_detailed_wer


def calculate_whisper_normalized_wer(reference: str, hypothesis: str) -> Dict:
    """
    Calculate WER using Whisper's EnglishTextNormalizer.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        
    Returns:
        Dictionary with detailed WER metrics
    """
    if WHISPER_AVAILABLE:
        # Apply Whisper normalization to both texts
        normalized_ref = WHISPER_NORMALIZER(reference)
        normalized_hyp = WHISPER_NORMALIZER(hypothesis)
        return calculate_detailed_wer(normalized_ref, normalized_hyp)
    else:
        # Fallback to basic normalization
        return calculate_detailed_wer(reference, hypothesis)


class ModelMetricsGenerator:
    """Generate comprehensive metrics for a single model."""
    
    def __init__(self, model_name: str, results_dir: str = "results", reference_dir: str = None):
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        
        # Set up paths
        self.transcript_dir = self.results_dir / "transcripts" / model_name.replace("/", "_")
        self.metrics_dir = self.results_dir / "metrics"
        
        # Set up reference directory
        if reference_dir:
            self.reference_dir = reference_dir
        else:
            # Default to cleaned transcripts
            self.reference_dir = self.results_dir.parent / "data" / "cleaned_transcripts"
        
        # Initialize WER calculator
        self.wer_calculator = WERCalculator(self.reference_dir)
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def load_speed_metrics(self) -> Optional[Dict]:
        """Load speed metrics from transcription process."""
        speed_file = self.metrics_dir / f"{self.model_name.replace('/', '_')}_speed.json"
        
        if not speed_file.exists():
            print(f"⚠️  Speed metrics not found: {speed_file}")
            return None
        
        try:
            with open(speed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading speed metrics: {e}")
            return None
    
    def calculate_wer_metrics(self) -> List[Dict]:
        """Calculate WER metrics for all transcripts using Whisper normalization."""
        try:
            # Get transcript files directly (without console output from basic WER calculation)
            transcript_path = Path(self.transcript_dir)
            if not transcript_path.exists():
                raise FileNotFoundError(f"Transcript directory not found: {self.transcript_dir}")
            
            # Get all transcript files
            transcript_files = list(transcript_path.glob("*_transcript.txt"))
            if not transcript_files:
                # Try alternative pattern
                transcript_files = list(transcript_path.glob("*_transcription.txt"))
            
            if not transcript_files:
                raise FileNotFoundError(f"No transcript files found in: {self.transcript_dir}")
            
            print(f"📊 Evaluating {len(transcript_files)} transcripts")
            
            whisper_results = []
            for i, transcript_file in enumerate(transcript_files, 1):
                print(f"[{i}/{len(transcript_files)}] Evaluating: {transcript_file.name}")
                
                try:
                    # Find reference file
                    transcript_path_obj = Path(transcript_file)
                    base_name = transcript_path_obj.stem.replace('_transcript', '').replace('_transcription', '')
                    reference_file = self.wer_calculator.find_reference_file(base_name)
                    
                    if not reference_file:
                        print(f"❌ Error: No reference file found for transcript: {transcript_file}")
                        continue
                    
                    # Load reference and hypothesis texts
                    with open(reference_file, 'r', encoding='utf-8') as f:
                        reference = f.read().strip()
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        hypothesis = f.read().strip()
                    
                    # Calculate Whisper-normalized WER
                    whisper_wer = calculate_whisper_normalized_wer(reference, hypothesis)
                    
                    # Create result with correct WER
                    whisper_result = {
                        'reference_file': reference_file,
                        'transcript_file': str(transcript_file),
                        'wer': whisper_wer['wer'],
                        'substitutions': whisper_wer['substitutions'],
                        'deletions': whisper_wer['deletions'],
                        'insertions': whisper_wer['insertions'],
                        'correct': whisper_wer['correct'],
                        'reference_words': whisper_wer['reference_words'],
                        'hypothesis_words': whisper_wer['hypothesis_words']
                    }
                    
                    whisper_results.append(whisper_result)
                    
                    # Print the CORRECT Whisper-normalized WER
                    print(f"✅ WER: {whisper_wer['wer']:.4f}")
                    
                except Exception as file_error:
                    print(f"❌ Error processing {transcript_file.name}: {file_error}")
                    continue
            
            print(f"✅ Calculated WER for {len(whisper_results)} files using Whisper normalization")
            return whisper_results
            
        except Exception as e:
            print(f"❌ Error calculating WER metrics: {e}")
            return []
    
    def generate_comprehensive_metrics(self) -> Dict:
        """Generate comprehensive metrics combining speed and accuracy."""
        print(f"📊 Generating metrics for model: {self.model_name}")
        
        # Load speed metrics
        speed_metrics = self.load_speed_metrics()
        
        # Calculate WER metrics
        wer_results = self.calculate_wer_metrics()
        
        if not wer_results:
            print("❌ No WER results available")
            return {}
        
        # Calculate WER summary statistics
        wer_values = [r['wer'] for r in wer_results]
        wer_summary = {
            'average_wer': sum(wer_values) / len(wer_values),
            'median_wer': sorted(wer_values)[len(wer_values) // 2],
            'best_wer': min(wer_values),
            'worst_wer': max(wer_values),
            'wer_std': self._calculate_std(wer_values)
        }
        
        # Calculate word-level statistics
        total_ref_words = sum(r['reference_words'] for r in wer_results)
        total_hyp_words = sum(r['hypothesis_words'] for r in wer_results)
        total_correct = sum(r['correct'] for r in wer_results)
        total_substitutions = sum(r['substitutions'] for r in wer_results)
        total_deletions = sum(r['deletions'] for r in wer_results)
        total_insertions = sum(r['insertions'] for r in wer_results)
        
        word_stats = {
            'total_reference_words': total_ref_words,
            'total_hypothesis_words': total_hyp_words,
            'total_correct_words': total_correct,
            'total_substitutions': total_substitutions,
            'total_deletions': total_deletions,
            'total_insertions': total_insertions,
            'word_accuracy': total_correct / total_ref_words if total_ref_words > 0 else 0
        }
        
        # Combine with speed metrics
        comprehensive_metrics = {
            'model_name': self.model_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'normalization': 'Whisper EnglishTextNormalizer' if WHISPER_AVAILABLE else 'Basic normalization',
            'wer_metrics': wer_summary,
            'word_statistics': word_stats,
            'file_count': len(wer_results)
        }
        
        # Add speed metrics if available
        if speed_metrics:
            comprehensive_metrics['speed_metrics'] = speed_metrics.get('summary', {})
        
        # Add per-file details
        comprehensive_metrics['file_details'] = []
        for wer_result in wer_results:
            file_detail = {
                'file': Path(wer_result['transcript_file']).name,
                'wer': wer_result['wer'],
                'reference_words': wer_result['reference_words'],
                'hypothesis_words': wer_result['hypothesis_words'],
                'substitutions': wer_result['substitutions'],
                'deletions': wer_result['deletions'],
                'insertions': wer_result['insertions']
            }
            
            # Add speed info if available
            if speed_metrics and 'file_details' in speed_metrics:
                speed_detail = next((f for f in speed_metrics['file_details'] 
                                   if f['audio_file'].replace('_conversation.wav', '') in file_detail['file']), None)
                if speed_detail:
                    file_detail['transcription_duration'] = speed_detail['duration']
            
            comprehensive_metrics['file_details'].append(file_detail)
        
        return comprehensive_metrics
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def save_metrics(self, metrics: Dict) -> str:
        """Save metrics to file."""
        if not metrics:
            raise ValueError("No metrics to save")
        
        output_file = self.metrics_dir / f"{self.model_name.replace('/', '_')}_wer.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        return str(output_file)
    
    def print_summary(self, metrics: Dict):
        """Print metrics summary to console."""
        if not metrics:
            print("❌ No metrics available")
            return
        
        print(f"\n📊 METRICS SUMMARY: {self.model_name}")
        print("=" * 60)
        
        # WER metrics
        if 'wer_metrics' in metrics:
            wer = metrics['wer_metrics']
            print(f"🎯 Word Error Rate:")
            print(f"   Average WER: {wer['average_wer']:.4f} ({wer['average_wer']*100:.2f}%)")
            print(f"   Best WER:    {wer['best_wer']:.4f} ({wer['best_wer']*100:.2f}%)")
            print(f"   Worst WER:   {wer['worst_wer']:.4f} ({wer['worst_wer']*100:.2f}%)")
            print(f"   Std Dev:     {wer.get('wer_std', 0):.4f}")
        
        # Word statistics
        if 'word_statistics' in metrics:
            words = metrics['word_statistics']
            print(f"\n📝 Word Statistics:")
            print(f"   Total reference words: {words['total_reference_words']:,}")
            print(f"   Word accuracy:         {words['word_accuracy']:.4f} ({words['word_accuracy']*100:.2f}%)")
            print(f"   Substitutions:         {words['total_substitutions']:,}")
            print(f"   Deletions:             {words['total_deletions']:,}")
            print(f"   Insertions:            {words['total_insertions']:,}")
        
        # Speed metrics
        if 'speed_metrics' in metrics:
            speed = metrics['speed_metrics']
            print(f"\n⚡ Speed Metrics:")
            print(f"   Total processing time: {speed.get('total_duration', 0):.2f}s")
            print(f"   Average per file:      {speed.get('average_duration', 0):.2f}s")
            print(f"   Fastest file:          {speed.get('fastest_file', 0):.2f}s")
            print(f"   Slowest file:          {speed.get('slowest_file', 0):.2f}s")
        
        print(f"\n📁 Files processed: {metrics.get('file_count', 0)}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Model Metrics")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--reference_dir", help="Reference texts directory")
    parser.add_argument("--output", help="Output file (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Initialize generator
    try:
        generator = ModelMetricsGenerator(
            model_name=args.model_name,
            results_dir=args.results_dir,
            reference_dir=args.reference_dir
        )
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return
    
    # Generate metrics
    try:
        metrics = generator.generate_comprehensive_metrics()
    except Exception as e:
        print(f"❌ Error generating metrics: {e}")
        return
    
    if not metrics:
        print("❌ No metrics generated")
        return
    
    # Save metrics
    try:
        output_file = args.output if args.output else generator.save_metrics(metrics)
        if not args.output:
            output_file = generator.save_metrics(metrics)
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            output_file = args.output
        
        print(f"💾 Metrics saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
    
    # Print summary
    generator.print_summary(metrics)


if __name__ == "__main__":
    main()