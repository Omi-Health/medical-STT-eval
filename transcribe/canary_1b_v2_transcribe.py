#!/usr/bin/env python3
"""
NVIDIA Canary-1B-v2 Speech Recognition Model.
Uses native long-form dynamic chunking for files >40s.
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe.base_transcriber import BaseTranscriber, TranscriptionResult

# Try to import NeMo
try:
    from nemo.collections.asr.models import ASRModel
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("NeMo not available. Install with: pip install 'nemo_toolkit[asr]'")


class Canary1Bv2Transcriber(BaseTranscriber):
    """NVIDIA Canary-1B-v2 transcriber with native long-form support."""

    def __init__(self, model_name: str = "canary-1b-v2", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit required")

        print(f"Loading nvidia/canary-1b-v2...")
        self.model = ASRModel.from_pretrained("nvidia/canary-1b-v2")
        print("Model loaded successfully")

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe using native long-form dynamic chunking."""
        start_time = time.time()

        try:
            # batch_size=1 enables dynamic chunking for long audio
            output = self.model.transcribe(
                [audio_path],
                source_lang='en',
                target_lang='en',
                batch_size=1
            )

            text = output[0].text
            duration = time.time() - start_time

            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_path
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"Error: {e}")
            return TranscriptionResult(
                text=f"[ERROR: {str(e)}]",
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_path
            )


def main():
    parser = argparse.ArgumentParser(description="Transcribe with Canary-1B-v2")
    parser.add_argument("--audio_dir", type=str, required=True)
    args = parser.parse_args()

    transcriber = Canary1Bv2Transcriber()
    audio_files = transcriber.get_audio_files(args.audio_dir)
    transcriber.transcribe_batch(audio_files)


if __name__ == "__main__":
    main()
