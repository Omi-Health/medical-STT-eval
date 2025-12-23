#!/usr/bin/env python3
"""
Word Error Rate (WER) calculation utilities.
Shared across all models for consistent evaluation.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def normalize_text(text: str) -> List[str]:
    """
    Normalize text for WER calculation.
    
    Args:
        text: Input text
        
    Returns:
        List of normalized words
    """
    # Remove newlines and normalize whitespace
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words and remove empty strings
    return [word for word in text.split() if word]


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) using edit distance.
    
    WER = (S + D + I) / N
    Where:
    - S = Substitutions
    - D = Deletions
    - I = Insertions
    - N = Number of words in reference
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (model output)
        
    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    # Calculate edit distance using dynamic programming
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # WER = edit distance / number of reference words
    return dp[m][n] / len(ref_words)


def calculate_detailed_wer(reference: str, hypothesis: str) -> Dict:
    """
    Calculate detailed WER metrics including substitutions, deletions, and insertions.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        
    Returns:
        Dictionary with detailed metrics
    """
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)
    
    if not ref_words:
        return {
            'wer': 1.0 if hyp_words else 0.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': len(hyp_words),
            'correct': 0,
            'reference_words': 0,
            'hypothesis_words': len(hyp_words)
        }
    
    # Calculate edit distance with operation tracking
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack to count operations
    i, j = m, n
    substitutions = deletions = insertions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    
    correct = len(ref_words) - (substitutions + deletions)
    
    return {
        'wer': dp[m][n] / len(ref_words),
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'correct': correct,
        'reference_words': len(ref_words),
        'hypothesis_words': len(hyp_words)
    }


class WERCalculator:
    """WER calculator for batch processing."""
    
    def __init__(self, reference_dir: str):
        self.reference_dir = Path(reference_dir)
        if not self.reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    
    def load_reference_text(self, reference_file: str) -> str:
        """Load reference text from file."""
        try:
            with open(reference_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                # Normalize whitespace
                text = re.sub(r'\s+', ' ', text)
                return text
        except Exception as e:
            raise Exception(f"Error loading reference {reference_file}: {e}")
    
    def find_reference_file(self, audio_file: str) -> Optional[str]:
        """Find corresponding reference file for an audio file."""
        audio_path = Path(audio_file)
        base_name = audio_path.stem.replace('_conversation', '')
        
        # Try different possible reference file patterns
        patterns = [
            f"{base_name}_pure_text.txt",
            f"{base_name}_transcript.txt",
            f"{base_name}.txt"
        ]
        
        for pattern in patterns:
            ref_file = self.reference_dir / pattern
            if ref_file.exists():
                return str(ref_file)
        
        return None
    
    def evaluate_transcript(self, transcript_file: str, audio_file: str = None) -> Dict:
        """
        Evaluate a single transcript file.
        
        Args:
            transcript_file: Path to transcript file
            audio_file: Optional path to corresponding audio file (for reference matching)
            
        Returns:
            Dictionary with evaluation results
        """
        # Load hypothesis text
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                hypothesis = f.read().strip()
        except Exception as e:
            raise Exception(f"Error loading transcript {transcript_file}: {e}")
        
        # Find reference file
        if audio_file:
            reference_file = self.find_reference_file(audio_file)
        else:
            # Try to infer from transcript filename
            transcript_path = Path(transcript_file)
            base_name = transcript_path.stem.replace('_transcript', '').replace('_transcription', '')
            reference_file = self.find_reference_file(base_name)
        
        if not reference_file:
            raise FileNotFoundError(f"No reference file found for transcript: {transcript_file}")
        
        # Load reference text
        reference = self.load_reference_text(reference_file)
        
        # Calculate WER
        wer_details = calculate_detailed_wer(reference, hypothesis)
        
        return {
            'transcript_file': transcript_file,
            'reference_file': reference_file,
            'audio_file': audio_file,
            **wer_details,
            'reference_text': reference,
            'hypothesis_text': hypothesis
        }
    
    def evaluate_model_transcripts(self, transcript_dir: str) -> List[Dict]:
        """
        Evaluate all transcripts for a model.
        
        Args:
            transcript_dir: Directory containing model transcripts
            
        Returns:
            List of evaluation results
        """
        transcript_path = Path(transcript_dir)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
        
        # Get all transcript files
        transcript_files = list(transcript_path.glob("*_transcript.txt"))
        if not transcript_files:
            # Try alternative pattern
            transcript_files = list(transcript_path.glob("*_transcription.txt"))
        
        if not transcript_files:
            raise FileNotFoundError(f"No transcript files found in: {transcript_dir}")
        
        results = []
        
        print(f"📊 Evaluating {len(transcript_files)} transcripts")
        
        for i, transcript_file in enumerate(transcript_files, 1):
            print(f"[{i}/{len(transcript_files)}] Evaluating: {transcript_file.name}")
            
            try:
                result = self.evaluate_transcript(str(transcript_file))
                results.append(result)
                print(f"✅ WER: {result['wer']:.4f}")
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        return results


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WER Calculation")
    parser.add_argument("--reference_dir", required=True, help="Directory containing reference texts")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing model transcripts")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Initialize calculator
    try:
        calculator = WERCalculator(args.reference_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Evaluate transcripts
    try:
        results = calculator.evaluate_model_transcripts(args.transcript_dir)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    if not results:
        print("❌ No transcripts were evaluated successfully")
        return
    
    # Calculate summary statistics
    total_wer = sum(r['wer'] for r in results)
    avg_wer = total_wer / len(results)
    
    summary = {
        'total_files': len(results),
        'average_wer': avg_wer,
        'best_wer': min(r['wer'] for r in results),
        'worst_wer': max(r['wer'] for r in results),
        'total_reference_words': sum(r['reference_words'] for r in results),
        'total_hypothesis_words': sum(r['hypothesis_words'] for r in results)
    }
    
    print(f"\n📊 EVALUATION SUMMARY")
    print(f"=" * 50)
    print(f"Total files: {summary['total_files']}")
    print(f"Average WER: {summary['average_wer']:.4f}")
    print(f"Best WER: {summary['best_wer']:.4f}")
    print(f"Worst WER: {summary['worst_wer']:.4f}")
    
    # Save results
    if args.output:
        output_data = {
            'summary': summary,
            'detailed_results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()