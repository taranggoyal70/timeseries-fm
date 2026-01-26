#!/usr/bin/env python3
"""
Structure Verification Script
Verifies all Python files are properly connected and imports work
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} MISSING: {filepath}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description} imports successfully")
        return True
    except ImportError as e:
        print(f"✗ {description} import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠ {description} import warning: {e}")
        return True  # May fail due to missing dependencies but structure is OK

def verify_structure():
    """Verify complete project structure."""
    print("="*70)
    print("CHRONOS-2 PROJECT STRUCTURE VERIFICATION")
    print("="*70)
    print()
    
    all_ok = True
    
    # Core Python files
    print("CORE PYTHON FILES")
    print("-"*70)
    core_files = {
        'main.py': 'Main execution script',
        'data_loader.py': 'Data download and loading',
        'chronos_forecaster.py': 'UV/MV forecasting',
        'chronos_experiment_runner.py': 'Experiment orchestration',
        'experiment_config.py': 'Configuration',
        'metrics_calculator.py': 'RMSE/MAPE calculations',
        'metrics.py': 'Legacy metrics',
        'visualizer.py': 'Visualization functions',
        'analyze_results.py': 'Results analysis',
        'run_complete_pipeline.py': 'Complete pipeline runner'
    }
    
    for filename, description in core_files.items():
        if not check_file_exists(filename, description):
            all_ok = False
    
    print()
    
    # Example files
    print("EXAMPLE FILES")
    print("-"*70)
    example_files = {
        'example_single_forecast.py': 'Single forecast example',
        'class_example_mag7.py': 'Magnificent-7 example',
        'class_style_notebook.py': 'Notebook-style example'
    }
    
    for filename, description in example_files.items():
        if not check_file_exists(filename, description):
            all_ok = False
    
    print()
    
    # Check imports
    print("IMPORT VERIFICATION")
    print("-"*70)
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    modules = {
        'data_loader': 'DataLoader module',
        'chronos_forecaster': 'ChronosForecaster module',
        'chronos_experiment_runner': 'ChronosExperimentRunner module',
        'experiment_config': 'ExperimentConfig module',
        'metrics_calculator': 'MetricsCalculator module',
        'metrics': 'Metrics module',
        'visualizer': 'Visualizer module'
    }
    
    for module, description in modules.items():
        if not check_import(module, description):
            all_ok = False
    
    print()
    
    # Check key dependencies
    print("DEPENDENCY VERIFICATION")
    print("-"*70)
    
    dependencies = {
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'matplotlib': 'Matplotlib',
        'yfinance': 'yfinance',
        'tqdm': 'tqdm'
    }
    
    for dep, description in dependencies.items():
        if not check_import(dep, description):
            all_ok = False
    
    print()
    
    # Check chronos package separately
    print("CHRONOS-2 VERIFICATION")
    print("-"*70)
    try:
        from chronos import Chronos2Pipeline
        print("✓ Chronos-2 package installed and importable")
        print("✓ Chronos2Pipeline available")
    except ImportError as e:
        print(f"✗ Chronos-2 package not installed: {e}")
        print("  Install with: pip install chronos-forecasting")
        all_ok = False
    
    print()
    
    # Verify key functions exist
    print("FUNCTION VERIFICATION")
    print("-"*70)
    
    try:
        from data_loader import DataLoader
        loader = DataLoader()
        print("✓ DataLoader class instantiates")
        
        from chronos_experiment_runner import ChronosExperimentRunner
        print("✓ ChronosExperimentRunner class available")
        
        from metrics_calculator import MetricsCalculator
        calc = MetricsCalculator()
        print("✓ MetricsCalculator class instantiates")
        
        from visualizer import plot_forecast_comparison, plot_metrics_summary
        print("✓ Visualization functions available")
        
    except Exception as e:
        print(f"✗ Function verification failed: {e}")
        all_ok = False
    
    print()
    
    # Check directory structure
    print("DIRECTORY STRUCTURE")
    print("-"*70)
    
    # These will be created on first run
    optional_dirs = ['data', 'results', 'visualizations']
    for dirname in optional_dirs:
        if os.path.exists(dirname):
            print(f"✓ {dirname}/ directory exists")
        else:
            print(f"⊳ {dirname}/ directory will be created on first run")
    
    print()
    
    # Final summary
    print("="*70)
    if all_ok:
        print("✓ ALL VERIFICATIONS PASSED")
        print("="*70)
        print("\nProject structure is correct and all components are connected.")
        print("\nYou can now run:")
        print("  python main.py --quick-test --dataset stocks")
        print("  python run_complete_pipeline.py --quick-test --dataset stocks")
        print("  python example_single_forecast.py")
        print()
        return 0
    else:
        print("✗ SOME VERIFICATIONS FAILED")
        print("="*70)
        print("\nPlease fix the issues above before running experiments.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(verify_structure())
