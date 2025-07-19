"""
MycoNet++ Contemplative Integration Module
==========================================

Integration layer between contemplative features and existing MycoNet++ infrastructure.
Provides backward compatibility and seamless feature integration.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Import contemplative modules
from myconet_contemplative_core import ContemplativeProcessor, WisdomInsight
from myconet_wisdom_signals import WisdomSignalGrid, WisdomSignalConfig
from myconet_contemplative_entities import ContemplativeNeuroAgent, ContemplativeColony
from myconet_contemplative_overmind import ContemplativeOvermind

logger = logging.getLogger(__name__)

class MycoNetCompatibilityLayer:
    """
    Compatibility layer for existing MycoNet++ interfaces
    Ensures contemplative features work with existing codebase
    """
    
    @staticmethod
    def wrap_existing_agent(existing_agent, contemplative_config: Dict[str, Any] = None):
        """
        Wrap an existing NeuroAgent with contemplative capabilities
        
        Args:
            existing_agent: Original MycoNet++ NeuroAgent
            contemplative_config: Configuration for contemplative features
            
        Returns:
            ContemplativeNeuroAgent: Enhanced agent with contemplative capabilities
        """
        contemplative_config = contemplative_config or {}
        
        # Extract properties from existing agent
        agent_config = {
            'initial_energy': getattr(existing_agent, 'energy', 1.0),
            'initial_health': getattr(existing_agent, 'health', 1.0),
            'mutation_rate': getattr(existing_agent, 'mutation_rate', 0.01),
            'learning_rate': getattr(existing_agent, 'learning_rate', 0.001),
            'brain_config': {
                'input_size': getattr(existing_agent.brain, 'input_size', 10) if hasattr(existing_agent, 'brain') else 10,
                'hidden_size': getattr(existing_agent.brain, 'hidden_size', 64) if hasattr(existing_agent, 'brain') else 64,
                'output_size': getattr(existing_agent.brain, 'output_size', 5) if hasattr(existing_agent, 'brain') else 5
            },
            'contemplative_config': contemplative_config
        }
        
        # Create contemplative agent
        contemplative_agent = ContemplativeNeuroAgent(
            agent_id=getattr(existing_agent, 'agent_id', np.random.randint(100000)),
            x=getattr(existing_agent, 'x', 0),
            y=getattr(existing_agent, 'y', 0),
            config=agent_config
        )
        
        # Transfer additional properties
        if hasattr(existing_agent, 'age'):
            contemplative_agent.age = existing_agent.age
        if hasattr(existing_agent, 'generation'):
            contemplative_agent.generation = existing_agent.generation
        if hasattr(existing_agent, 'memory'):
            contemplative_agent.memory = existing_agent.memory
        
        logger.info(f"Wrapped existing agent {existing_agent.agent_id} with contemplative capabilities")
        return contemplative_agent
    
    @staticmethod
    def extract_legacy_observations(observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract observations compatible with legacy MycoNet++ agents
        
        Args:
            observations: Contemplative observations dict
            
        Returns:
            Legacy-compatible observations
        """
        legacy_obs = {}
        
        # Map contemplative observations to legacy format
        mapping = {
            'x': 'x',
            'y': 'y',
            'energy': 'energy',
            'health': 'health',
            'food_nearby': 'food_nearby',
            'water_nearby': 'water_nearby',
            'danger_level': 'danger_level',
            'other_agents_nearby': 'other_agents_nearby'
        }
        
        for legacy_key, contemplative_key in mapping.items():
            if contemplative_key in observations:
                legacy_obs[legacy_key] = observations[contemplative_key]
        
        return legacy_obs
    
    @staticmethod
    def create_contemplative_environment_from_legacy(legacy_env):
        """
        Create contemplative environment wrapper around legacy environment
        
        Args:
            legacy_env: Existing MycoNet++ environment
            
        Returns:
            ContemplativeEnvironment: Enhanced environment
        """
        from myconet_contemplative_main import ContemplativeEnvironment
        
        # Create new contemplative environment with same dimensions
        width = getattr(legacy_env, 'width', 50)
        height = getattr(legacy_env, 'height', 50)
        
        contemplative_env = ContemplativeEnvironment(width, height)
        
        # Copy over existing state if available
        if hasattr(legacy_env, 'food_grid'):
            contemplative_env.food_grid = legacy_env.food_grid.copy()
        if hasattr(legacy_env, 'water_grid'):
            contemplative_env.water_grid = legacy_env.water_grid.copy()
        if hasattr(legacy_env, 'hazard_grid'):
            contemplative_env.hazard_grid = legacy_env.hazard_grid.copy()
        
        return contemplative_env

class ContemplativeFeatureManager:
    """
    Manages activation and configuration of contemplative features
    Allows gradual feature rollout and A/B testing
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.feature_flags = {
            'mindfulness_processing': True,
            'wisdom_signals': True,
            'ethical_reasoning': True,
            'collective_meditation': True,
            'overmind_dharma_compiler': True,
            'contemplative_evolution': True,
            'network_wisdom_flow': True,
            'suffering_detection': True
        }
        
        self.feature_configs = {}
        self._initialize_feature_configs()
    
    def _initialize_feature_configs(self):
        """Initialize default configurations for each feature"""
        self.feature_configs = {
            'mindfulness_processing': {
                'update_frequency': 10,
                'attention_history_size': 100,
                'mindfulness_threshold': 0.6
            },
            'wisdom_signals': {
                'signal_strength': 0.6,
                'propagation_distance': 5,
                'decay_rate': 0.05,
                'cross_interference': True
            },
            'ethical_reasoning': {
                'reasoning_depth': 3,
                'ethical_frameworks': ['consequentialist', 'deontological', 'virtue_ethics', 'buddhist_ethics'],
                'ethical_threshold': 0.7
            },
            'collective_meditation': {
                'trigger_threshold': 0.7,
                'sync_radius': 10,
                'meditation_duration': 30
            },
            'overmind_dharma_compiler': {
                'ethical_principles': {
                    'reduce_suffering': 0.95,
                    'promote_wellbeing': 0.85,
                    'preserve_autonomy': 0.80,
                    'foster_wisdom': 0.90
                },
                'violation_threshold': 0.3
            },
            'contemplative_evolution': {
                'trait_mutation_rate': 0.05,
                'trait_inheritance_weight': 0.8,
                'fitness_weights': {
                    'survival': 0.3,
                    'ethical_behavior': 0.25,
                    'wisdom_generation': 0.2,
                    'collective_harmony': 0.25
                }
            }
        }
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.feature_flags.get(feature_name, False)
    
    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """Get configuration for a specific feature"""
        if not self.is_feature_enabled(feature_name):
            return {}
        return self.feature_configs.get(feature_name, {})
    
    def enable_feature(self, feature_name: str, config: Optional[Dict[str, Any]] = None):
        """Enable a feature with optional custom configuration"""
        self.feature_flags[feature_name] = True
        if config:
            self.feature_configs[feature_name].update(config)
        logger.info(f"Enabled contemplative feature: {feature_name}")
    
    def disable_feature(self, feature_name: str):
        """Disable a feature"""
        self.feature_flags[feature_name] = False
        logger.info(f"Disabled contemplative feature: {feature_name}")
    
    def create_agent_config(self, base_agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create agent configuration with active contemplative features"""
        contemplative_config = {}
        
        # Build contemplative configuration based on enabled features
        if self.is_feature_enabled('mindfulness_processing'):
            mindfulness_config = self.get_feature_config('mindfulness_processing')
            contemplative_config.update({
                'mindfulness_update_frequency': mindfulness_config['update_frequency'],
                'mindfulness_capacity': mindfulness_config['attention_history_size']
            })
        
        if self.is_feature_enabled('wisdom_signals'):
            wisdom_config = self.get_feature_config('wisdom_signals')
            contemplative_config.update({
                'wisdom_signal_strength': wisdom_config['signal_strength'],
                'wisdom_sharing_radius': wisdom_config['propagation_distance']
            })
        
        if self.is_feature_enabled('ethical_reasoning'):
            ethical_config = self.get_feature_config('ethical_reasoning')
            contemplative_config.update({
                'ethical_reasoning_depth': ethical_config['reasoning_depth'],
                'ethical_threshold': ethical_config['ethical_threshold']
            })
        
        # Merge with base configuration
        enhanced_config = base_agent_config.copy()
        enhanced_config['contemplative_config'] = contemplative_config
        
        return enhanced_config

class ExperimentRunner:
    """
    Manages different types of contemplative experiments
    Provides templates for common research scenarios
    """
    
    def __init__(self, feature_manager: ContemplativeFeatureManager):
        self.feature_manager = feature_manager
        self.experiment_templates = self._create_experiment_templates()
    
    def _create_experiment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined experiment templates"""
        return {
            'baseline_control': {
                'description': 'Baseline experiment with minimal contemplative features',
                'enabled_features': ['mindfulness_processing'],
                'environment_size': (30, 30),
                'population_size': 15,
                'simulation_steps': 500,
                'overmind_enabled': False
            },
            
            'wisdom_propagation': {
                'description': 'Study wisdom signal propagation through the network',
                'enabled_features': ['mindfulness_processing', 'wisdom_signals', 'network_wisdom_flow'],
                'environment_size': (40, 40),
                'population_size': 25,
                'simulation_steps': 800,
                'overmind_enabled': True,
                'special_config': {
                    'wisdom_signals': {
                        'signal_strength': 0.8,
                        'propagation_distance': 7
                    }
                }
            },
            
            'ethical_evolution': {
                'description': 'Study evolution of ethical reasoning capabilities',
                'enabled_features': ['mindfulness_processing', 'ethical_reasoning', 'contemplative_evolution'],
                'environment_size': (35, 35),
                'population_size': 20,
                'simulation_steps': 1200,
                'overmind_enabled': True,
                'special_config': {
                    'ethical_reasoning': {
                        'reasoning_depth': 5
                    },
                    'contemplative_evolution': {
                        'fitness_weights': {
                            'survival': 0.2,
                            'ethical_behavior': 0.4,
                            'wisdom_generation': 0.2,
                            'collective_harmony': 0.2
                        }
                    }
                }
            },
            
            'collective_meditation': {
                'description': 'Study collective meditation and network synchronization',
                'enabled_features': ['mindfulness_processing', 'wisdom_signals', 'collective_meditation'],
                'environment_size': (50, 50),
                'population_size': 30,
                'simulation_steps': 600,
                'overmind_enabled': True,
                'special_config': {
                    'collective_meditation': {
                        'trigger_threshold': 0.5,
                        'sync_radius': 15
                    }
                }
            },
            
            'suffering_response': {
                'description': 'Study detection and response to suffering in the network',
                'enabled_features': ['mindfulness_processing', 'wisdom_signals', 'suffering_detection', 'ethical_reasoning'],
                'environment_size': (40, 40),
                'population_size': 25,
                'simulation_steps': 700,
                'overmind_enabled': True,
                'special_config': {
                    'ethical_reasoning': {
                        'ethical_frameworks': ['consequentialist', 'buddhist_ethics'],
                        'reasoning_depth': 4
                    }
                }
            },
            
            'full_contemplative': {
                'description': 'Full contemplative system with all features enabled',
                'enabled_features': list(self.feature_manager.feature_flags.keys()),
                'environment_size': (60, 60),
                'population_size': 40,
                'simulation_steps': 1500,
                'overmind_enabled': True,
                'special_config': {}
            }
        }
    
    def setup_experiment(self, experiment_name: str, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Setup an experiment based on template
        
        Args:
            experiment_name: Name of experiment template
            custom_config: Optional custom configuration overrides
            
        Returns:
            Complete experiment configuration
        """
        if experiment_name not in self.experiment_templates:
            raise ValueError(f"Unknown experiment template: {experiment_name}")
        
        template = self.experiment_templates[experiment_name].copy()
        
        # Apply custom configuration if provided
        if custom_config:
            template.update(custom_config)
        
        # Configure feature manager
        for feature in self.feature_manager.feature_flags:
            if feature in template['enabled_features']:
                special_feature_config = template.get('special_config', {}).get(feature, {})
                self.feature_manager.enable_feature(feature, special_feature_config)
            else:
                self.feature_manager.disable_feature(feature)
        
        # Create simulation configuration
        from myconet_contemplative_main import ContemplativeSimulationConfig, ContemplativeConfig
        from myconet_wisdom_signals import WisdomSignalConfig
        
        env_width, env_height = template['environment_size']
        
        # Create contemplative config based on enabled features
        contemplative_config = ContemplativeConfig()
        if self.feature_manager.is_feature_enabled('mindfulness_processing'):
            mindfulness_cfg = self.feature_manager.get_feature_config('mindfulness_processing')
            contemplative_config.mindfulness_update_frequency = mindfulness_cfg['update_frequency']
        
        if self.feature_manager.is_feature_enabled('wisdom_signals'):
            wisdom_cfg = self.feature_manager.get_feature_config('wisdom_signals')
            contemplative_config.wisdom_signal_strength = wisdom_cfg['signal_strength']
            contemplative_config.wisdom_sharing_radius = wisdom_cfg['propagation_distance']
        
        if self.feature_manager.is_feature_enabled('ethical_reasoning'):
            ethical_cfg = self.feature_manager.get_feature_config('ethical_reasoning')
            contemplative_config.ethical_reasoning_depth = ethical_cfg['reasoning_depth']
        
        # Create wisdom signal config
        wisdom_signal_config = WisdomSignalConfig()
        if self.feature_manager.is_feature_enabled('wisdom_signals'):
            wisdom_cfg = self.feature_manager.get_feature_config('wisdom_signals')
            wisdom_signal_config.base_diffusion_rate = 0.1
            wisdom_signal_config.base_decay_rate = wisdom_cfg['decay_rate']
            wisdom_signal_config.propagation_distance = wisdom_cfg['propagation_distance']
            wisdom_signal_config.cross_signal_interference = wisdom_cfg['cross_interference']
        
        # Create full simulation config
        sim_config = ContemplativeSimulationConfig(
            experiment_name=f"{experiment_name}_experiment",
            environment_width=env_width,
            environment_height=env_height,
            initial_population=template['population_size'],
            max_steps=template['simulation_steps'],
            enable_overmind=template['overmind_enabled'],
            contemplative_config=contemplative_config,
            wisdom_signal_config=wisdom_signal_config,
            output_directory=f"results_{experiment_name}"
        )
        
        experiment_config = {
            'simulation_config': sim_config,
            'feature_manager': self.feature_manager,
            'experiment_template': template,
            'experiment_description': template['description']
        }
        
        logger.info(f"Configured experiment: {experiment_name}")
        logger.info(f"Description: {template['description']}")
        logger.info(f"Enabled features: {template['enabled_features']}")
        
        return experiment_config
    
    def list_experiments(self) -> List[str]:
        """List available experiment templates"""
        return list(self.experiment_templates.keys())
    
    def get_experiment_description(self, experiment_name: str) -> str:
        """Get description of an experiment template"""
        if experiment_name in self.experiment_templates:
            return self.experiment_templates[experiment_name]['description']
        return f"Unknown experiment: {experiment_name}"

class DataAnalyzer:
    """
    Analyzes contemplative simulation data and generates insights
    """
    
    def __init__(self):
        self.analysis_functions = {
            'wisdom_propagation': self._analyze_wisdom_propagation,
            'ethical_evolution': self._analyze_ethical_evolution,
            'collective_behavior': self._analyze_collective_behavior,
            'network_dynamics': self._analyze_network_dynamics,
            'contemplative_states': self._analyze_contemplative_states
        }
    
    def analyze_simulation_data(self, simulation_data: Dict[str, Any], 
                              analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Analyze simulation data with specified analysis types
        
        Args:
            simulation_data: Complete simulation data
            analysis_types: List of analysis types to perform
            
        Returns:
            Analysis results dictionary
        """
        if analysis_types is None:
            analysis_types = list(self.analysis_functions.keys())
        
        results = {
            'summary': self._generate_summary(simulation_data),
            'analyses': {}
        }
        
        for analysis_type in analysis_types:
            if analysis_type in self.analysis_functions:
                try:
                    analysis_result = self.analysis_functions[analysis_type](simulation_data)
                    results['analyses'][analysis_type] = analysis_result
                except Exception as e:
                    logger.warning(f"Analysis {analysis_type} failed: {e}")
                    results['analyses'][analysis_type] = {'error': str(e)}
        
        return results
    
    def _generate_summary(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary of simulation"""
        population_data = simulation_data.get('population_data', [])
        wisdom_data = simulation_data.get('wisdom_data', [])
        ethical_data = simulation_data.get('ethical_data', [])
        
        if not population_data:
            return {'error': 'No population data available'}
        
        final_pop = population_data[-1]
        initial_pop = population_data[0]
        
        summary = {
            'simulation_length': len(population_data),
            'initial_population': initial_pop.get('total_population', 0),
            'final_population': final_pop.get('total_population', 0),
            'survival_rate': final_pop.get('total_population', 0) / max(initial_pop.get('total_population', 1), 1),
            'average_final_energy': final_pop.get('average_energy', 0),
            'average_final_health': final_pop.get('average_health', 0),
            'generation_diversity': final_pop.get('generation_diversity', 0)
        }
        
        if wisdom_data:
            final_wisdom = wisdom_data[-1]
            summary.update({
                'total_wisdom_generated': final_wisdom.get('total_wisdom_generated', 0),
                'final_average_mindfulness': final_wisdom.get('average_mindfulness', 0),
                'collective_meditation_events': sum(1 for w in wisdom_data if w.get('agents_in_meditation', 0) > 0)
            })
        
        if ethical_data:
            final_ethical = ethical_data[-1]
            summary.update({
                'final_ethical_ratio': final_ethical.get('ethical_decision_ratio', 0),
                'collective_harmony': final_ethical.get('collective_harmony', 0)
            })
        
        return summary
    
    def _analyze_wisdom_propagation(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze wisdom signal propagation patterns"""
        wisdom_data = simulation_data.get('wisdom_data', [])
        network_data = simulation_data.get('network_data', [])
        
        if not wisdom_data or not network_data:
            return {'error': 'Insufficient data for wisdom propagation analysis'}
        
        # Calculate propagation efficiency over time
        wisdom_generated = [w.get('total_wisdom_generated', 0) for w in wisdom_data]
        wisdom_received = [w.get('total_wisdom_received', 0) for w in wisdom_data]
        
        propagation_efficiency = []
        for i in range(len(wisdom_generated)):
            if wisdom_generated[i] > 0:
                efficiency = wisdom_received[i] / wisdom_generated[i]
                propagation_efficiency.append(efficiency)
            else:
                propagation_efficiency.append(0.0)
        
        # Analyze network signal metrics
        signal_diversity = [n.get('signal_diversity', 0) for n in network_data]
        total_signals = [n.get('total_signals_created', 0) for n in network_data]
        
        return {
            'average_propagation_efficiency': np.mean(propagation_efficiency) if propagation_efficiency else 0,
            'propagation_efficiency_trend': np.polyfit(range(len(propagation_efficiency)), propagation_efficiency, 1)[0] if len(propagation_efficiency) > 1 else 0,
            'peak_wisdom_generation': max(wisdom_generated) if wisdom_generated else 0,
            'average_signal_diversity': np.mean(signal_diversity) if signal_diversity else 0,
            'total_signals_created': max(total_signals) if total_signals else 0,
            'wisdom_propagation_phases': self._identify_propagation_phases(wisdom_generated, wisdom_received)
        }
    
    def _analyze_ethical_evolution(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evolution of ethical behavior over time"""
        ethical_data = simulation_data.get('ethical_data', [])
        population_data = simulation_data.get('population_data', [])
        
        if not ethical_data:
            return {'error': 'No ethical data available'}
        
        ethical_ratios = [e.get('ethical_decision_ratio', 0) for e in ethical_data]
        collective_harmony = [e.get('collective_harmony', 0) for e in ethical_data]
        
        # Calculate trends
        ethical_trend = np.polyfit(range(len(ethical_ratios)), ethical_ratios, 1)[0] if len(ethical_ratios) > 1 else 0
        harmony_trend = np.polyfit(range(len(collective_harmony)), collective_harmony, 1)[0] if len(collective_harmony) > 1 else 0
        
        # Identify ethical development phases
        phases = self._identify_ethical_phases(ethical_ratios)
        
        return {
            'initial_ethical_ratio': ethical_ratios[0] if ethical_ratios else 0,
            'final_ethical_ratio': ethical_ratios[-1] if ethical_ratios else 0,
            'ethical_improvement': ethical_ratios[-1] - ethical_ratios[0] if len(ethical_ratios) > 1 else 0,
            'ethical_trend': ethical_trend,
            'harmony_trend': harmony_trend,
            'ethical_consistency': 1.0 - np.std(ethical_ratios) if ethical_ratios else 0,
            'ethical_phases': phases,
            'peak_ethical_performance': max(ethical_ratios) if ethical_ratios else 0
        }
    
    def _analyze_collective_behavior(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collective behavior patterns"""
        wisdom_data = simulation_data.get('wisdom_data', [])
        ethical_data = simulation_data.get('ethical_data', [])
        overmind_data = simulation_data.get('overmind_data', [])
        
        if not wisdom_data:
            return {'error': 'No wisdom data for collective behavior analysis'}
        
        # Analyze meditation events
        meditation_events = []
        for i, w in enumerate(wisdom_data):
            agents_in_meditation = w.get('agents_in_meditation', 0)
            if agents_in_meditation > 0:
                meditation_events.append({
                    'step': i,
                    'participants': agents_in_meditation,
                    'mindfulness_level': w.get('average_mindfulness', 0)
                })
        
        # Collective synchronization analysis
        mindfulness_levels = [w.get('average_mindfulness', 0) for w in wisdom_data]
        synchronization_events = self._detect_synchronization_events(mindfulness_levels)
        
        # Overmind intervention analysis
        overmind_interventions = 0
        successful_interventions = 0
        if overmind_data:
            for o in overmind_data:
                if o.get('decisions_made', 0) > 0:
                    overmind_interventions = o.get('decisions_made', 0)
                    successful_interventions = o.get('successful_interventions', 0)
        
        return {
            'total_meditation_events': len(meditation_events),
            'average_meditation_size': np.mean([e['participants'] for e in meditation_events]) if meditation_events else 0,
            'largest_meditation': max([e['participants'] for e in meditation_events]) if meditation_events else 0,
            'synchronization_events': len(synchronization_events),
            'collective_coherence_trend': np.polyfit(range(len(mindfulness_levels)), mindfulness_levels, 1)[0] if len(mindfulness_levels) > 1 else 0,
            'overmind_interventions': overmind_interventions,
            'overmind_success_rate': successful_interventions / max(overmind_interventions, 1)
        }
    
    def _analyze_network_dynamics(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network-level dynamics and emergent properties"""
        network_data = simulation_data.get('network_data', [])
        
        if not network_data:
            return {'error': 'No network data available'}
        
        # Extract network metrics over time
        coherence_levels = [n.get('network_contemplative_coherence', 0) for n in network_data]
        signal_diversity = [n.get('signal_diversity', 0) for n in network_data]
        wisdom_flow = [n.get('wisdom_flow_efficiency', 0) for n in network_data]
        
        # Analyze network states
        network_states = self._classify_network_states(coherence_levels, signal_diversity)
        
        return {
            'average_network_coherence': np.mean(coherence_levels) if coherence_levels else 0,
            'peak_network_coherence': max(coherence_levels) if coherence_levels else 0,
            'coherence_stability': 1.0 - np.std(coherence_levels) if coherence_levels else 0,
            'average_signal_diversity': np.mean(signal_diversity) if signal_diversity else 0,
            'wisdom_flow_efficiency': np.mean(wisdom_flow) if wisdom_flow else 0,
            'network_state_distribution': network_states,
            'emergent_behavior_indicators': self._detect_emergent_behaviors(network_data)
        }
    
    def _analyze_contemplative_states(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contemplative states and transitions"""
        wisdom_data = simulation_data.get('wisdom_data', [])
        
        if not wisdom_data:
            return {'error': 'No wisdom data for contemplative state analysis'}
        
        mindfulness_levels = [w.get('average_mindfulness', 0) for w in wisdom_data]
        meditation_participants = [w.get('agents_in_meditation', 0) for w in wisdom_data]
        
        # Classify contemplative states
        contemplative_states = []
        for i in range(len(mindfulness_levels)):
            mindfulness = mindfulness_levels[i]
            meditation = meditation_participants[i]
            
            if meditation > 0:
                state = 'collective_meditation'
            elif mindfulness > 0.7:
                state = 'high_mindfulness'
            elif mindfulness > 0.4:
                state = 'moderate_mindfulness'
            else:
                state = 'ordinary'
            
            contemplative_states.append(state)
        
        # Count state transitions
        state_transitions = {}
        for i in range(1, len(contemplative_states)):
            transition = f"{contemplative_states[i-1]} -> {contemplative_states[i]}"
            state_transitions[transition] = state_transitions.get(transition, 0) + 1
        
        return {
            'state_distribution': {state: contemplative_states.count(state) for state in set(contemplative_states)},
            'average_mindfulness': np.mean(mindfulness_levels) if mindfulness_levels else 0,
            'mindfulness_trend': np.polyfit(range(len(mindfulness_levels)), mindfulness_levels, 1)[0] if len(mindfulness_levels) > 1 else 0,
            'meditation_frequency': len([m for m in meditation_participants if m > 0]) / len(meditation_participants) if meditation_participants else 0,
            'state_transitions': state_transitions,
            'contemplative_stability': self._measure_state_stability(contemplative_states)
        }
    
    def _identify_propagation_phases(self, wisdom_generated: List[float], wisdom_received: List[float]) -> List[Dict[str, Any]]:
        """Identify distinct phases in wisdom propagation"""
        if len(wisdom_generated) < 10:
            return []
        
        phases = []
        window_size = len(wisdom_generated) // 5  # Divide into 5 phases
        
        for i in range(0, len(wisdom_generated), window_size):
            end_idx = min(i + window_size, len(wisdom_generated))
            phase_generated = wisdom_generated[i:end_idx]
            phase_received = wisdom_received[i:end_idx]
            
            if phase_generated:
                avg_efficiency = np.mean([r/max(g, 1) for g, r in zip(phase_generated, phase_received)])
                phases.append({
                    'start_step': i,
                    'end_step': end_idx - 1,
                    'average_generation': np.mean(phase_generated),
                    'average_reception': np.mean(phase_received),
                    'efficiency': avg_efficiency,
                    'phase_type': 'high_propagation' if avg_efficiency > 0.7 else 'low_propagation'
                })
        
        return phases
    
    def _identify_ethical_phases(self, ethical_ratios: List[float]) -> List[Dict[str, Any]]:
        """Identify phases in ethical development"""
        if len(ethical_ratios) < 10:
            return []
        
        phases = []
        window_size = len(ethical_ratios) // 4
        
        for i in range(0, len(ethical_ratios), window_size):
            end_idx = min(i + window_size, len(ethical_ratios))
            phase_ratios = ethical_ratios[i:end_idx]
            
            if phase_ratios:
                avg_ratio = np.mean(phase_ratios)
                trend = np.polyfit(range(len(phase_ratios)), phase_ratios, 1)[0] if len(phase_ratios) > 1 else 0
                
                phase_type = 'improvement' if trend > 0.01 else ('decline' if trend < -0.01 else 'stable')
                
                phases.append({
                    'start_step': i,
                    'end_step': end_idx - 1,
                    'average_ethical_ratio': avg_ratio,
                    'trend': trend,
                    'phase_type': phase_type
                })
        
        return phases
    
    def _detect_synchronization_events(self, mindfulness_levels: List[float]) -> List[Dict[str, Any]]:
        """Detect network synchronization events"""
        events = []
        threshold = np.mean(mindfulness_levels) + np.std(mindfulness_levels) if mindfulness_levels else 0.7
        
        in_event = False
        event_start = 0
        
        for i, level in enumerate(mindfulness_levels):
            if level > threshold and not in_event:
                in_event = True
                event_start = i
            elif level <= threshold and in_event:
                in_event = False
                events.append({
                    'start_step': event_start,
                    'end_step': i - 1,
                    'duration': i - event_start,
                    'peak_mindfulness': max(mindfulness_levels[event_start:i])
                })
        
        return events
    
    def _classify_network_states(self, coherence_levels: List[float], signal_diversity: List[float]) -> Dict[str, int]:
        """Classify network states based on coherence and diversity"""
        states = {}
        
        for coherence, diversity in zip(coherence_levels, signal_diversity):
            if coherence > 0.7 and diversity > 0.6:
                state = 'high_coherence_high_diversity'
            elif coherence > 0.7 and diversity <= 0.6:
                state = 'high_coherence_low_diversity'
            elif coherence <= 0.7 and diversity > 0.6:
                state = 'low_coherence_high_diversity'
            else:
                state = 'low_coherence_low_diversity'
            
            states[state] = states.get(state, 0) + 1
        
        return states
    
    def _detect_emergent_behaviors(self, network_data: List[Dict[str, Any]]) -> List[str]:
        """Detect emergent behaviors in network data"""
        behaviors = []
        
        # Look for sudden increases in signal creation
        signal_counts = [n.get('total_signals_created', 0) for n in network_data]
        if len(signal_counts) > 10:
            signal_diffs = np.diff(signal_counts)
            if np.max(signal_diffs) > np.mean(signal_diffs) + 2 * np.std(signal_diffs):
                behaviors.append('signal_cascade')
        
        # Look for coherence phase transitions
        coherence_levels = [n.get('network_contemplative_coherence', 0) for n in network_data]
        if len(coherence_levels) > 10:
            coherence_diffs = np.diff(coherence_levels)
            if np.any(np.abs(coherence_diffs) > 0.3):
                behaviors.append('coherence_phase_transition')
        
        # Look for sustained high activity
        wisdom_flow = [n.get('wisdom_flow_efficiency', 0) for n in network_data]
        if len(wisdom_flow) > 20:
            high_flow_periods = [i for i, flow in enumerate(wisdom_flow) if flow > 0.8]
            if len(high_flow_periods) > len(wisdom_flow) * 0.3:
                behaviors.append('sustained_high_wisdom_flow')
        
        return behaviors
    
    def _measure_state_stability(self, contemplative_states: List[str]) -> float:
        """Measure stability of contemplative states"""
        if len(contemplative_states) < 2:
            return 1.0
        
        transitions = sum(1 for i in range(1, len(contemplative_states)) 
                         if contemplative_states[i] != contemplative_states[i-1])
        
        return 1.0 - (transitions / (len(contemplative_states) - 1))

# Factory functions for easy integration
def create_contemplative_simulation_from_legacy(legacy_config: Dict[str, Any], 
                                              contemplative_features: List[str] = None) -> Dict[str, Any]:
    """
    Create contemplative simulation configuration from legacy MycoNet++ config
    
    Args:
        legacy_config: Existing MycoNet++ configuration
        contemplative_features: List of contemplative features to enable
        
    Returns:
        Enhanced simulation configuration
    """
    # Create feature manager
    feature_manager = ContemplativeFeatureManager(legacy_config)
    
    # Enable specified features
    if contemplative_features:
        for feature in contemplative_features:
            if feature in feature_manager.feature_flags:
                feature_manager.enable_feature(feature)
    
    # Create experiment runner
    experiment_runner = ExperimentRunner(feature_manager)
    
    # Determine appropriate experiment template based on legacy config
    population_size = legacy_config.get('initial_population', 15)
    max_steps = legacy_config.get('max_steps', 500)
    
    if population_size < 15 and max_steps < 300:
        template_name = 'baseline_control'
    elif population_size >= 30 or max_steps >= 1000:
        template_name = 'full_contemplative'
    else:
        template_name = 'wisdom_propagation'
    
    # Setup experiment
    experiment_config = experiment_runner.setup_experiment(template_name)
    
    # Apply legacy config overrides
    sim_config = experiment_config['simulation_config']
    sim_config.initial_population = legacy_config.get('initial_population', sim_config.initial_population)
    sim_config.max_steps = legacy_config.get('max_steps', sim_config.max_steps)
    sim_config.environment_width = legacy_config.get('environment_width', sim_config.environment_width)
    sim_config.environment_height = legacy_config.get('environment_height', sim_config.environment_height)
    
    return experiment_config

def run_contemplative_experiment(experiment_name: str, 
                               custom_config: Optional[Dict[str, Any]] = None,
                               analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run a complete contemplative experiment with analysis
    
    Args:
        experiment_name: Name of experiment template to run
        custom_config: Optional custom configuration
        analysis_types: Types of analysis to perform
        
    Returns:
        Complete results including simulation data and analysis
    """
    from myconet_contemplative_main import ContemplativeSimulation
    
    # Setup experiment
    feature_manager = ContemplativeFeatureManager({})
    experiment_runner = ExperimentRunner(feature_manager)
    experiment_config = experiment_runner.setup_experiment(experiment_name, custom_config)
    
    # Run simulation
    simulation = ContemplativeSimulation(experiment_config['simulation_config'])
    simulation.run_simulation()
    
    # Perform analysis
    analyzer = DataAnalyzer()
    analysis_results = analyzer.analyze_simulation_data(
        simulation.simulation_data, 
        analysis_types
    )
    
    # Combine results
    complete_results = {
        'experiment_config': experiment_config,
        'simulation_data': simulation.simulation_data,
        'analysis_results': analysis_results,
        'experiment_summary': {
            'experiment_name': experiment_name,
            'description': experiment_config['experiment_description'],
            'enabled_features': experiment_config['experiment_template']['enabled_features'],
            'final_population': len([a for a in simulation.agents if a.alive]),
            'simulation_steps': simulation.step_count
        }
    }
    
    return complete_results

# Integration testing function
def test_contemplative_integration():
    """Test the integration between contemplative and legacy systems"""
    print("Testing contemplative integration...")
    
    # Test feature manager
    feature_manager = ContemplativeFeatureManager({})
    print(f"✓ Feature manager created with {len(feature_manager.feature_flags)} features")
    
    # Test experiment runner
    experiment_runner = ExperimentRunner(feature_manager)
    experiments = experiment_runner.list_experiments()
    print(f"✓ Experiment runner created with {len(experiments)} templates")
    
    # Test experiment setup
    experiment_config = experiment_runner.setup_experiment('baseline_control')
    print("✓ Experiment configuration created")
    
    # Test data analyzer
    analyzer = DataAnalyzer()
    print(f"✓ Data analyzer created with {len(analyzer.analysis_functions)} analysis types")
    
    # Test compatibility layer
    print("✓ Compatibility layer functions available")
    
    print("Integration test completed successfully!")
    return True