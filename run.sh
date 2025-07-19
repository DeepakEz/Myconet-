#!/bin/bash
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
    
    python myconet_contemplative_main.py \
        --config-file "configs/$config.json" \
        --verbose \
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
