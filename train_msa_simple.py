#!/usr/bin/env python3
"""
Simple training runner for MSA fine-tuning
Just run: python train_msa_simple.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quran_muaalem.training.train_msa import main

if __name__ == "__main__":
    main()
