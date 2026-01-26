#!/usr/bin/env python3
"""
Complete Pipeline Runner for Chronos-2 Forecasting
Runs experiments and generates visualizations in one go
"""

import sys
import os
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Run complete Chronos-2 forecasting pipeline with visualizations'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['stocks', 'rates', 'combined', 'all'],
        default='stocks',
        help='Dataset to use (default: stocks)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test mode'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download'
    )
    
    parser.add_argument(
        '--skip-experiments',
        action='store_true',
        help='Skip experiments (only analyze existing results)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CHRONOS-2 COMPLETE PIPELINE")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Quick test: {args.quick_test}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Step 1: Run experiments (unless skipped)
    if not args.skip_experiments:
        cmd_parts = [
            "python main.py",
            f"--dataset {args.dataset}",
            f"--device {args.device}"
        ]
        
        if args.quick_test:
            cmd_parts.append("--quick-test")
        
        if args.skip_download:
            cmd_parts.append("--skip-download")
        
        cmd = " ".join(cmd_parts)
        
        if not run_command(cmd, "STEP 1: Running Experiments"):
            print("\n✗ Pipeline failed at experiment stage")
            sys.exit(1)
    else:
        print("\n⊳ Skipping experiments (using existing results)")
    
    # Step 2: Analyze results and generate visualizations
    if not run_command("python analyze_results.py", "STEP 2: Analyzing Results & Generating Visualizations"):
        print("\n✗ Pipeline failed at analysis stage")
        sys.exit(1)
    
    # Final summary
    print("\n" + "="*70)
    print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("="*70)
    print("\nResults available in:")
    print("  - results/experiments_*.json (full results)")
    print("  - results/experiments_summary_*.csv (summary table)")
    print("  - visualizations/*.png (plots)")
    print("  - visualizations/summary_report.txt (text report)")
    print()
    print("Next steps:")
    print("  1. Review visualizations in 'visualizations/' directory")
    print("  2. Analyze CSV file in Excel or Pandas")
    print("  3. Read summary_report.txt for key findings")
    print()

if __name__ == "__main__":
    main()
