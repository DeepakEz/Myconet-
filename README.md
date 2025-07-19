# Contemplative MycoNet++

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
