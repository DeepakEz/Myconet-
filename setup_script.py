#!/usr/bin/env python3
"""
Setup Script for Contemplative MycoNet
=====================================

Setup and quick test script for the contemplative MycoNet system.
Handles dependencies, creates example configurations, and runs basic tests.
"""

import sys
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy',
        'dataclasses',  # Built-in for Python 3.7+
        'pathlib',      # Built-in for Python 3.4+
        'logging',      # Built-in
        'argparse',     # Built-in
        'collections',  # Built-in
        'typing',       # Built-in for Python 3.5+
        'enum',         # Built-in for Python 3.4+
        'time',         # Built-in
    ]
    
    optional_packages = [
        'torch',        # For neural network functionality
        'matplotlib',   # For visualization
        'seaborn',      # For advanced plotting
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ✗ {package} (REQUIRED)")
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"  ✗ {package} (optional)")
    
    if missing_required:
        print(f"\nERROR: Missing required packages: {missing_required}")
        print("Please install them with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nWARNING: Missing optional packages: {missing_optional}")
        print("For full functionality, install with: pip install " + " ".join(missing_optional))
        print("The system will work without them but with reduced capabilities.")
    
    print("\nDependency check completed!")
    return True

def create_example_configs():
    """Create example configuration files"""
    print("\nCreating example configuration files...")
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Basic configuration
    basic_config = {
        "experiment_name": "basic_contemplative_test",
        "environment_width": 25,
        "environment_height": 25,
        "initial_population": 10,
        "max_population": 50,
        "max_steps": 300,
        "save_interval": 50,
        "visualization_interval": 25,
        "enable_overmind": True,
        "overmind_intervention_frequency": 15,
        "brain_input_size": 16,
        "brain_hidden_size": 32,
        "brain_output_size": 8,
        "output_directory": "results_basic",
        "track_wisdom_propagation": True,
        "track_collective_behavior": True,
        "track_ethical_decisions": True,
        "contemplative_config": {
            "enable_contemplative_processing": True,
            "mindfulness_update_frequency": 10,
            "wisdom_signal_strength": 0.6,
            "collective_meditation_threshold": 0.7,
            "ethical_reasoning_depth": 3,
            "contemplative_memory_capacity": 500,
            "wisdom_sharing_radius": 3,
            "compassion_sensitivity": 0.8
        },
        "wisdom_signal_config": {
            "signal_types": [
                "ethical_insight",
                "suffering_alert", 
                "compassion_gradient",
                "wisdom_beacon",
                "meditation_sync",
                "cooperation_call",
                "mindfulness_wave"
            ],
            "base_diffusion_rate": 0.1,
            "base_decay_rate": 0.05,
            "propagation_distance": 5,
            "intensity_threshold": 0.1,
            "cross_signal_interference": True
        }
    }
    
    basic_file = configs_dir / "basic_test.json"
    with open(basic_file, 'w', encoding='utf-8') as f:
        json.dump(basic_config, f, indent=2)
    print(f"  Created: {basic_file}")
    
    # Advanced configuration for longer experiments
    advanced_config = basic_config.copy()
    advanced_config.update({
        "experiment_name": "advanced_contemplative_study",
        "environment_width": 40,
        "environment_height": 40,
        "initial_population": 20,
        "max_population": 100,
        "max_steps": 1000,
        "save_interval": 100,
        "output_directory": "results_advanced",
        "contemplative_config": {
            **basic_config["contemplative_config"],
            "wisdom_signal_strength": 0.8,
            "collective_meditation_threshold": 0.6,
            "ethical_reasoning_depth": 5,
            "contemplative_memory_capacity": 1000,
            "wisdom_sharing_radius": 5,
            "compassion_sensitivity": 0.9
        }
    })
    
    advanced_file = configs_dir / "advanced_study.json"
    with open(advanced_file, 'w', encoding='utf-8') as f:
        json.dump(advanced_config, f, indent=2)
    print(f"  Created: {advanced_file}")
    
    # Minimal configuration for quick testing
    minimal_config = {
        "experiment_name": "minimal_test",
        "environment_width": 15,
        "environment_height": 15,
        "initial_population": 5,
        "max_population": 20,
        "max_steps": 100,
        "save_interval": 25,
        "enable_overmind": False,  # Disable for speed
        "brain_input_size": 12,
        "brain_hidden_size": 24,
        "brain_output_size": 6,
        "output_directory": "results_minimal",
        "contemplative_config": {
            "enable_contemplative_processing": True,
            "mindfulness_update_frequency": 5,
            "wisdom_signal_strength": 0.5,
            "collective_meditation_threshold": 0.8,
            "ethical_reasoning_depth": 2,
            "contemplative_memory_capacity": 200,
            "wisdom_sharing_radius": 2,
            "compassion_sensitivity": 0.7
        }
    }
    
    minimal_file = configs_dir / "minimal_test.json"
    with open(minimal_file, 'w', encoding='utf-8') as f:
        json.dump(minimal_config, f, indent=2)
    print(f"  Created: {minimal_file}")
    
    print("Example configurations created!")

def create_directory_structure():
    """Create necessary directory structure"""
    print("\nCreating directory structure...")
    
    directories = [
        "configs",
        "results",
        "results_basic",
        "results_advanced", 
        "results_minimal",
        "contemplative_results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  Created: {directory}/")
    
    print("Directory structure created!")

def create_run_script():
    """Create convenient run script"""
    print("\nCreating run script...")
    
    run_script_content = '''#!/bin/bash
# Contemplative MycoNet Quick Run Script

echo "=== Contemplative MycoNet Quick Run ==="
echo

# Function to run experiment
run_experiment() {
    local config=$1
    local description=$2
    
    echo "Running $description..."
    echo "Configuration: $config"
    echo
    
    python myconet_contemplative_main.py \\
        --config-file "configs/$config.json" \\
        --verbose \\
        --seed 42
    
    echo
    echo "$description completed!"
    echo "Results saved to results_$config/"
    echo
}

# Check if specific experiment requested
if [ "$1" != "" ]; then
    case $1 in
        "minimal"|"min")
            run_experiment "minimal_test" "Minimal Test"
            ;;
        "basic")
            run_experiment "basic_test" "Basic Contemplative Experiment"
            ;;
        "advanced"|"adv")
            run_experiment "advanced_study" "Advanced Contemplative Study"
            ;;
        "list")
            echo "Available experiments:"
            echo "  minimal  - Quick test (5 agents, 100 steps)"
            echo "  basic    - Basic experiment (10 agents, 300 steps)"
            echo "  advanced - Full study (20 agents, 1000 steps)"
            echo
            echo "Usage: ./run.sh [experiment_name]"
            echo "Example: ./run.sh minimal"
            ;;
        *)
            echo "Unknown experiment: $1"
            echo "Use './run.sh list' to see available experiments"
            ;;
    esac
else
    echo "No experiment specified. Running minimal test..."
    echo
    run_experiment "minimal_test" "Minimal Test (Default)"
fi
'''
    
    run_script_file = Path("run.sh")
    with open(run_script_file, 'w', encoding='utf-8') as f:
        f.write(run_script_content)
    
    # Make executable on Unix systems
    import stat
    run_script_file.chmod(run_script_file.stat().st_mode | stat.S_IEXEC)
    
    print(f"  Created: {run_script_file}")
    
    # Also create Windows batch file
    windows_script_content = '''@echo off
REM Contemplative MycoNet Quick Run Script for Windows

echo === Contemplative MycoNet Quick Run ===
echo.

if "%1"=="" goto default
if "%1"=="minimal" goto minimal
if "%1"=="min" goto minimal
if "%1"=="basic" goto basic
if "%1"=="advanced" goto advanced
if "%1"=="adv" goto advanced
if "%1"=="list" goto list

echo Unknown experiment: %1
echo Use 'run.bat list' to see available experiments
goto end

:minimal
echo Running Minimal Test...
python myconet_contemplative_main.py --config-file "configs/minimal_test.json" --verbose --seed 42
goto end

:basic
echo Running Basic Contemplative Experiment...
python myconet_contemplative_main.py --config-file "configs/basic_test.json" --verbose --seed 42
goto end

:advanced
echo Running Advanced Contemplative Study...
python myconet_contemplative_main.py --config-file "configs/advanced_study.json" --verbose --seed 42
goto end

:list
echo Available experiments:
echo   minimal  - Quick test (5 agents, 100 steps)
echo   basic    - Basic experiment (10 agents, 300 steps)
echo   advanced - Full study (20 agents, 1000 steps)
echo.
echo Usage: run.bat [experiment_name]
echo Example: run.bat minimal
goto end

:default
echo No experiment specified. Running minimal test...
echo.
python myconet_contemplative_main.py --config-file "configs/minimal_test.json" --verbose --seed 42

:end
echo.
pause
'''
    
    windows_script_file = Path("run.bat")
    with open(windows_script_file, 'w', encoding='utf-8') as f:
        f.write(windows_script_content)
    
    print(f"  Created: {windows_script_file}")
    print("Run scripts created!")

def create_readme():
    """Create README file with usage instructions"""
    print("\nCreating README...")
    
    readme_content = '''# Contemplative MycoNet++

Enhanced MycoNet++ with contemplative AI capabilities, featuring mindfulness, wisdom propagation, ethical reasoning, and collective meditation.

## Quick Start

### 1. Setup
Run the setup script to check dependencies and create example configurations:
```bash
python setup_contemplative_myconet.py
```

### 2. Run Experiments

#### Using the convenient run scripts:
```bash
# Linux/Mac
./run.sh minimal     # Quick test (5 agents, 100 steps)
./run.sh basic       # Basic experiment (10 agents, 300 steps)
./run.sh advanced    # Full study (20 agents, 1000 steps)

# Windows
run.bat minimal
run.bat basic
run.bat advanced
```

#### Using the main script directly:
```bash
# Use preset configurations
python myconet_contemplative_main.py --config basic --verbose

# Use custom configuration file
python myconet_contemplative_main.py --config-file configs/basic_test.json --verbose

# Command line overrides
python myconet_contemplative_main.py --config basic --max-steps 500 --population 15
```

### 3. View Results
Results are saved to the specified output directory (e.g., `results_basic/`):
- `*_results.json`: Complete simulation data and analysis
- `checkpoint_*.json`: Periodic simulation checkpoints

## Key Features

### Contemplative Agents
- **Mindfulness Monitoring**: Tracks attention coherence and awareness levels
- **Wisdom Memory**: Stores and retrieves insights with temporal decay
- **Ethical Reasoning**: Multi-framework moral evaluation (consequentialist, deontological, virtue ethics, Buddhist ethics)
- **Contemplative States**: Ordinary, mindful, deep contemplation, collective meditation

### Wisdom Signal Network
- **Signal Types**: Suffering alerts, compassion gradients, wisdom beacons, meditation sync, cooperation calls
- **Propagation**: Chemical-signal-like diffusion with decay and amplification
- **Interference**: Cross-signal interactions (e.g., meditation amplifies wisdom)

### Contemplative Overmind
- **Network Dharma Compiler**: Ethical constraint enforcement for colony-level decisions
- **Collective Wisdom Processing**: Aggregates insights from across the network
- **Beneficial Interventions**: Coordinates help, triggers collective meditation, facilitates learning

### Evolution & Learning
- **Contemplative Trait Evolution**: Mindfulness capacity, wisdom thresholds, ethical sensitivity
- **Collective Behavior Emergence**: Network-wide meditation, cooperative resource sharing
- **Wisdom Propagation**: Insights spread through the network like nutrients in fungal networks

## Configuration

### Key Parameters

#### Contemplative Settings
- `enable_contemplative_processing`: Enable/disable contemplative features
- `wisdom_signal_strength`: Intensity of wisdom signal propagation
- `collective_meditation_threshold`: Threshold for triggering network meditation
- `ethical_reasoning_depth`: Complexity of ethical evaluation
- `compassion_sensitivity`: Responsiveness to suffering signals

#### Simulation Settings
- `environment_width/height`: World size
- `initial_population`: Starting number of agents
- `max_steps`: Simulation duration
- `enable_overmind`: Enable network-level AI coordination

## Analysis Metrics

### Population Metrics
- Survival rates, energy/health levels, generation diversity

### Contemplative Metrics
- Wisdom generation/propagation rates, mindfulness levels, collective harmony

### Ethical Metrics
- Ethical decision ratios, moral consistency, behavioral trends

### Network Metrics
- Signal diversity, network coherence, wisdom flow efficiency

## Example Configurations

### Minimal Test (configs/minimal_test.json)
- 5 agents, 15x15 environment, 100 steps
- No Overmind, basic contemplative features
- Quick functionality test

### Basic Experiment (configs/basic_test.json)
- 10 agents, 25x25 environment, 300 steps
- Overmind enabled, full contemplative features
- Standard research experiment

### Advanced Study (configs/advanced_study.json)
- 20 agents, 40x40 environment, 1000 steps
- Enhanced contemplative parameters
- Long-term behavioral evolution study

## Research Applications

### Studying Collective Intelligence
- How does wisdom propagate through agent networks?
- What triggers network-wide contemplative states?
- How do ethical behaviors emerge and spread?

### Human-AI Symbiosis Research
- Network integration of human and AI nodes
- Collaborative wisdom generation
- Mutual benefit optimization

### Beneficial AI Development
- Training AI systems with built-in ethical reasoning
- Network-level safety mechanisms
- Contemplative approach to AI alignment

## Dependencies

### Required
- Python 3.7+
- NumPy
- Standard library modules (json, pathlib, logging, etc.)

### Optional
- PyTorch (for neural network functionality)
- Matplotlib (for visualization)
- Seaborn (for advanced plotting)

## File Structure

```
contemplative_myconet/
├── myconet_contemplative_core.py      # Core contemplative processing
├── myconet_wisdom_signals.py          # Wisdom signal propagation system
├── myconet_contemplative_brains.py    # Enhanced neural architectures
├── myconet_contemplative_entities.py  # Contemplative agent classes
├── myconet_contemplative_overmind.py  # Network-level AI coordination
├── myconet_contemplative_main.py      # Main simulation script
├── setup_contemplative_myconet.py     # Setup and configuration helper
├── configs/                           # Configuration files
├── results/                           # Simulation outputs
└── README.md                          # This file
```

## Next Steps

1. **Run the minimal test** to verify everything works
2. **Experiment with configurations** to explore different behaviors
3. **Analyze results** to understand emergent contemplative behaviors
4. **Extend the system** with new contemplative features or metrics

---

*"True intelligence is wise intelligence, and true wisdom is efficient wisdom."*
'''
    
    readme_file = Path("README.md")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  Created: {readme_file}")
    print("README created!")

def run_quick_test():
    """Run a very quick test to verify system functionality"""
    print("\nRunning quick functionality test...")
    
    try:
        # Import all modules to check for syntax errors
        print("  Testing imports...")
        from myconet_contemplative_core import ContemplativeProcessor, WisdomInsight, WisdomType
        from myconet_wisdom_signals import WisdomSignalGrid, WisdomSignalConfig
        from myconet_contemplative_entities import ContemplativeNeuroAgent, ContemplativeConfig
        from myconet_contemplative_overmind import ContemplativeOvermind
        print("    ✓ All imports successful")
        
        # Test basic object creation
        print("  Testing object creation...")
        
        # Test contemplative processor
        processor = ContemplativeProcessor(agent_id=1, config={})
        print("    ✓ ContemplativeProcessor created")
        
        # Test wisdom signal grid
        signal_config = WisdomSignalConfig()
        signal_grid = WisdomSignalGrid(10, 10, signal_config)
        print("    ✓ WisdomSignalGrid created")
        
        # Test agent creation
        agent_config = {
            'contemplative_config': {
                'enable_contemplative_processing': True
            }
        }
        agent = ContemplativeNeuroAgent(agent_id=1, x=5, y=5, config=agent_config)
        print("    ✓ ContemplativeNeuroAgent created")
        
        # Test overmind creation  
        overmind = ContemplativeOvermind(
            colony_size=5, 
            environment_size=(10, 10),
            config={}
        )
        print("    ✓ ContemplativeOvermind created")
        
        print("  Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ✗ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Contemplative MycoNet++ Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\nSetup failed due to missing dependencies.")
        return False
    
    # Create directory structure
    create_directory_structure()
    
    # Create example configurations
    create_example_configs()
    
    # Create run scripts
    create_run_script()
    
    # Create README
    create_readme()
    
    # Run quick test
    if not run_quick_test():
        print("\nSetup completed but quick test failed.")
        print("Check the error messages above and ensure all modules are correctly implemented.")
        return False
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run a quick test:")
    print("   ./run.sh minimal    (Linux/Mac)")
    print("   run.bat minimal     (Windows)")
    print("   OR")
    print("   python myconet_contemplative_main.py --config basic --max-steps 50")
    print()
    print("2. View results in the results_*/ directories")
    print()
    print("3. Experiment with different configurations in configs/")
    print()
    print("4. Read README.md for detailed usage instructions")
    print("=" * 40)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)