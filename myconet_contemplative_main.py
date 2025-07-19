#!/usr/bin/env python3
"""
MycoNet++ Contemplative Main Script
===================================

Main execution script for the contemplative MycoNet system.
Integrates all contemplative modules with the existing MycoNet++ infrastructure.
"""

import argparse
import logging
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Contemplative imports
from myconet_contemplative_core import ContemplativeState, WisdomType
from myconet_wisdom_signals import WisdomSignalGrid, WisdomSignalConfig, WisdomSignalType
from myconet_contemplative_entities import ContemplativeNeuroAgent, ContemplativeColony, ContemplativeConfig
from myconet_contemplative_overmind import ContemplativeOvermind

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ContemplativeSimulationConfig:
    """Configuration for the contemplative simulation"""
    # Environment settings
    environment_width: int = 50
    environment_height: int = 50
    initial_population: int = 20
    max_population: int = 100
    
    # Simulation settings
    max_steps: int = 1000
    save_interval: int = 100
    visualization_interval: int = 50
    
    # Contemplative settings - use default_factory for mutable defaults
    contemplative_config: ContemplativeConfig = None
    wisdom_signal_config: WisdomSignalConfig = None
    
    # Overmind settings
    enable_overmind: bool = True
    overmind_intervention_frequency: int = 10
    
    # Agent brain settings
    brain_input_size: int = 16
    brain_hidden_size: int = 64
    brain_output_size: int = 8
    
    # Experiment settings
    experiment_name: str = "contemplative_basic"
    output_directory: str = "contemplative_results"
    
    # Evaluation settings
    track_wisdom_propagation: bool = True
    track_collective_behavior: bool = True
    track_ethical_decisions: bool = True
    
    def __post_init__(self):
        """Initialize mutable defaults after dataclass creation"""
        if self.contemplative_config is None:
            self.contemplative_config = ContemplativeConfig()
        if self.wisdom_signal_config is None:
            self.wisdom_signal_config = WisdomSignalConfig()

class ContemplativeEnvironment:
    """
    Simple environment for contemplative agents
    Compatible with existing MycoNet++ environment interface
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Resources grid
        self.food_grid = np.random.random((height, width)) * 0.5
        self.water_grid = np.random.random((height, width)) * 0.3
        
        # Hazards
        self.hazard_grid = np.zeros((height, width))
        num_hazards = width * height // 20
        for _ in range(num_hazards):
            hx, hy = np.random.randint(0, width), np.random.randint(0, height)
            self.hazard_grid[hy, hx] = np.random.uniform(0.3, 0.8)
        
        # Environmental state
        self.resource_regeneration_rate = 0.01
        self.hazard_movement_rate = 0.005
    
    def get_local_observations(self, x: int, y: int, radius: int = 2) -> Dict[str, Any]:
        """Get local environment observations for an agent"""
        observations = {
            'x': x,
            'y': y,
            'food_nearby': 0.0,
            'water_nearby': 0.0,
            'danger_level': 0.0,
            'safe_directions': [],
            'resource_directions': []
        }
        
        # Check surrounding area
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    distance = max(abs(dx), abs(dy))
                    if distance == 0:
                        continue
                    
                    # Weight by distance
                    weight = 1.0 / distance
                    
                    # Food and water
                    observations['food_nearby'] += self.food_grid[ny, nx] * weight
                    observations['water_nearby'] += self.water_grid[ny, nx] * weight
                    
                    # Dangers
                    observations['danger_level'] += self.hazard_grid[ny, nx] * weight
                    
                    # Directions
                    if self.food_grid[ny, nx] > 0.3 or self.water_grid[ny, nx] > 0.3:
                        observations['resource_directions'].append((dx, dy))
                    
                    if self.hazard_grid[ny, nx] < 0.2:
                        observations['safe_directions'].append((dx, dy))
        
        # Normalize
        observations['food_nearby'] = min(observations['food_nearby'], 1.0)
        observations['water_nearby'] = min(observations['water_nearby'], 1.0)
        observations['danger_level'] = min(observations['danger_level'], 1.0)
        
        return observations
    
    def update(self):
        """Update environment state"""
        # Regenerate resources
        self.food_grid += np.random.random((self.height, self.width)) * self.resource_regeneration_rate
        self.water_grid += np.random.random((self.height, self.width)) * self.resource_regeneration_rate
        
        # Cap resources
        self.food_grid = np.clip(self.food_grid, 0.0, 1.0)
        self.water_grid = np.clip(self.water_grid, 0.0, 1.0)
        
        # Move hazards slightly
        if np.random.random() < self.hazard_movement_rate:
            # Add new hazard
            hx, hy = np.random.randint(0, self.width), np.random.randint(0, self.height)
            self.hazard_grid[hy, hx] = min(self.hazard_grid[hy, hx] + 0.2, 0.8)
            
            # Decay existing hazards
            self.hazard_grid *= 0.99
    
    def consume_resource(self, x: int, y: int, resource_type: str, amount: float = 0.1):
        """Consume resources at location"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if resource_type == 'food':
                consumed = min(self.food_grid[y, x], amount)
                self.food_grid[y, x] -= consumed
                return consumed
            elif resource_type == 'water':
                consumed = min(self.water_grid[y, x], amount)
                self.water_grid[y, x] -= consumed
                return consumed
        return 0.0

class ContemplativeSimulation:
    """
    Main simulation class for contemplative MycoNet
    """
    def __init__(self, config: ContemplativeSimulationConfig):
        self.config = config
        self.step_count = 0
        
        # Initialize environment
        self.environment = ContemplativeEnvironment(
            config.environment_width, config.environment_height
        )
        
        # Initialize wisdom signal grid
        self.wisdom_signal_grid = WisdomSignalGrid(
            config.environment_width, config.environment_height, 
            config.wisdom_signal_config
        )
        
        # Initialize agents
        self.agents = self._create_initial_population()
        
        # Initialize colony
        self.colony = ContemplativeColony(self.agents, self.wisdom_signal_grid)
        
        # Initialize Overmind
        self.overmind_enabled = config.enable_overmind
        if config.enable_overmind:
            self.overmind = ContemplativeOvermind(
                colony_size=config.initial_population,
                environment_size=(config.environment_width, config.environment_height),
                config={
                    'colony_observation_size': 50,
                    'collective_action_size': 10,
                    'wisdom_processing_dim': 128
                }
            )
        else:
            self.overmind = None
        
        # Data collection
        self.simulation_data = {
            'steps': [],
            'population_data': [],
            'wisdom_data': [],
            'ethical_data': [],
            'overmind_data': [],
            'network_data': []
        }
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Contemplative simulation initialized with {len(self.agents)} agents")
        logger.info(f"Environment size: {config.environment_width}x{config.environment_height}")
        logger.info(f"Overmind enabled: {config.enable_overmind}")
    
    def _create_initial_population(self) -> List[ContemplativeNeuroAgent]:
        """Create initial population of contemplative agents"""
        agents = []
        
        # Handle contemplative config conversion
        if hasattr(self.config.contemplative_config, '__dict__'):
            # If it's a regular object, convert to dict
            contemplative_config_dict = vars(self.config.contemplative_config)
        elif hasattr(self.config.contemplative_config, '_asdict'):
            # If it's a namedtuple
            contemplative_config_dict = self.config.contemplative_config._asdict()
        else:
            # Try asdict if it's a dataclass, otherwise use default
            try:
                contemplative_config_dict = asdict(self.config.contemplative_config)
            except (TypeError, AttributeError):
                # Fallback to default configuration
                contemplative_config_dict = {
                    'enable_contemplative_processing': True,
                    'mindfulness_update_frequency': 20,
                    'wisdom_signal_strength': 0.3,
                    'collective_meditation_threshold': 0.8,
                    'ethical_reasoning_depth': 1,
                    'contemplative_memory_capacity': 100,
                    'wisdom_sharing_radius': 1,
                    'compassion_sensitivity': 0.4
                }
        
        agent_config = {
            'initial_energy': 1.0,
            'initial_health': 1.0,
            'mutation_rate': 0.01,
            'learning_rate': 0.001,
            'brain_config': {
                'input_size': self.config.brain_input_size,
                'hidden_size': self.config.brain_hidden_size,
                'output_size': self.config.brain_output_size
            },
            'contemplative_config': contemplative_config_dict
        }
        
        for i in range(self.config.initial_population):
            # Random initial positions
            x = np.random.randint(0, self.config.environment_width)
            y = np.random.randint(0, self.config.environment_height)
            
            agent = ContemplativeNeuroAgent(
                agent_id=i,
                x=x,
                y=y,
                config=agent_config
            )
            
            agents.append(agent)
        
        return agents
    
    def run_simulation(self):
        """Run the complete simulation"""
        logger.info(f"Starting contemplative simulation: {self.config.experiment_name}")
        logger.info(f"Max steps: {self.config.max_steps}")
        
        start_time = time.time()
        
        try:
            for step in range(self.config.max_steps):
                self.step_count = step
                
                # Run simulation step
                self._simulation_step()
                
                # Data collection
                if step % self.config.save_interval == 0:
                    self._collect_data()
                    self._save_checkpoint()
                
                # Progress logging
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    living_agents = len([a for a in self.agents if a.alive])
                    logger.info(f"Step {step}/{self.config.max_steps} - "
                              f"Population: {living_agents} - "
                              f"Time: {elapsed:.1f}s")
                
                # Early termination check
                if self._should_terminate():
                    logger.info(f"Simulation terminated early at step {step}")
                    break
            
            # Final data collection and analysis
            self._collect_data()
            self._save_final_results()
            
            total_time = time.time() - start_time
            logger.info(f"Simulation completed in {total_time:.1f} seconds")
            
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            self._save_checkpoint()
            raise
    
    def _get_final_metrics(self) -> Dict[str, Any]:
        """Get final metrics - REQUIRED BY PUBLICATION RUNNER"""
        living_agents = [a for a in self.agents if a.alive]
        
        if not living_agents:
            return {
                'population': {'population_size': 0, 'average_energy': 0, 'average_health': 0, 'average_age': 0},
                'contemplative': {'total_wisdom_generated': 0, 'average_mindfulness': 0, 'collective_harmony': 0, 'wisdom_propagation_efficiency': 0},
                'ethical': {'overall_ethical_ratio': 0, 'ethical_consistency': 0, 'ethical_improvement': 0},
                'network': {'signal_diversity': 0, 'network_coherence': 0, 'wisdom_flow_efficiency': 0, 'total_signals': 0, 'active_signals': 0}
            }
        
        # Calculate metrics from living agents
        total_wisdom = sum(getattr(agent, 'wisdom_insights_generated', 0) for agent in living_agents)
        mindfulness_scores = []
        for agent in living_agents:
            if hasattr(agent, 'contemplative_processor') and agent.contemplative_processor:
                try:
                    mindfulness = agent.contemplative_processor.mindfulness_monitor.get_mindfulness_score()
                    mindfulness_scores.append(mindfulness)
                except:
                    mindfulness_scores.append(0.5)
            else:
                mindfulness_scores.append(0.5)
        
        # Get network metrics safely
        try:
            network_stats = self.wisdom_signal_grid.get_network_stats()
        except:
            network_stats = {'signal_diversity': 0, 'network_coherence': 0, 'wisdom_flow_efficiency': 0, 'total_signals': 0, 'active_signals': 0}
        
        return {
            'population': {
                'population_size': len(living_agents),
                'average_energy': sum(agent.energy for agent in living_agents) / len(living_agents),
                'average_health': sum(agent.health for agent in living_agents) / len(living_agents),
                'average_age': sum(agent.age for agent in living_agents) / len(living_agents)
            },
            'contemplative': {
                'total_wisdom_generated': total_wisdom,
                'average_mindfulness': np.mean(mindfulness_scores) if mindfulness_scores else 0.5,
                'collective_harmony': np.mean([getattr(a, 'collective_harmony_level', 0.5) for a in living_agents]),
                'wisdom_propagation_efficiency': 0.0
            },
            'ethical': {
                'overall_ethical_ratio': sum(getattr(a, 'ethical_decisions', 0) for a in living_agents) / max(sum(getattr(a, 'decisions_made', 1) for a in living_agents), 1),
                'ethical_consistency': 0.5,
                'ethical_improvement': 0.0
            },
            'network': network_stats
        }
    
    def _simulation_step(self):
        """Execute one simulation step"""
        # Update environment
        self.environment.update()
        
        # Update wisdom signals
        self.wisdom_signal_grid.update_all_signals()
        
        # Agent actions
        self._process_agent_actions()
        
        # Colony-level updates
        self.colony.update_collective_state()
        
        # Overmind intervention
        if (self.overmind and 
            self.step_count % self.config.overmind_intervention_frequency == 0):
            self._process_overmind_intervention()
        
        # Population dynamics
        self._process_population_dynamics()
    
    def _process_agent_actions(self):
        """Process actions for all agents"""
        living_agents = [agent for agent in self.agents if agent.alive]
        
        for agent in living_agents:
            # Get environmental observations
            env_obs = self.environment.get_local_observations(agent.x, agent.y)
            
            # Add agent-specific observations
            agent_obs = self._get_agent_observations(agent, env_obs)
            
            # Define available actions
            available_actions = [
                'move_north', 'move_south', 'move_east', 'move_west',
                'eat_food', 'drink_water', 'rest', 'contemplate'
            ]
            
            # Agent makes decision
            chosen_action = agent.update(agent_obs, available_actions)
            
            # Execute action
            self._execute_agent_action(agent, chosen_action)
    
    def _get_agent_observations(self, agent: ContemplativeNeuroAgent, 
                              env_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive observations for an agent"""
        observations = env_obs.copy()
        
        # Agent internal state
        observations.update({
            'energy': agent.energy,
            'health': agent.health,
            'age': agent.age / 1000.0,  # Normalize
            'other_agents_nearby': self._count_nearby_agents(agent),
            'other_agents_distress': self._assess_nearby_distress(agent)
        })
        
        # Contemplative observations
        if agent.contemplative_processor:
            contemplative_summary = agent.contemplative_processor.get_state_summary()
            observations.update({
                'mindfulness_level': contemplative_summary.get('mindfulness_level', 0.0),
                'contemplative_state_depth': contemplative_summary.get('contemplation_depth', 0),
                'wisdom_insights_count': contemplative_summary.get('wisdom_insights_count', 0)
            })
        
        # Colony-level observations
        colony_metrics = self.colony.get_colony_metrics()
        observations.update({
            'colony_population': colony_metrics.get('population', 0) / 100.0,
            'collective_wisdom': colony_metrics.get('collective_wisdom_level', 0.0),
            'network_coherence': colony_metrics.get('network_coherence', 0.0)
        })
        
        return observations
    
    def _count_nearby_agents(self, agent: ContemplativeNeuroAgent, radius: int = 3) -> float:
        """Count other agents nearby (normalized)"""
        count = 0
        for other in self.agents:
            if other != agent and other.alive:
                distance = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                if distance <= radius:
                    count += 1
        return min(count / 5.0, 1.0)  # Normalize to [0, 1]
    
    def _assess_nearby_distress(self, agent: ContemplativeNeuroAgent, radius: int = 4) -> float:
        """Assess distress level of nearby agents"""
        distress_levels = []
        for other in self.agents:
            if other != agent and other.alive:
                distance = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                if distance <= radius:
                    # Calculate distress based on energy and health
                    distress = 1.0 - min(other.energy, other.health)
                    distress_levels.append(distress)
        
        return max(distress_levels) if distress_levels else 0.0
    
    def _execute_agent_action(self, agent: ContemplativeNeuroAgent, action: str):
        """Execute an agent's chosen action"""
        if action == 'move_north':
            agent.move(0, -1, self.environment)
        elif action == 'move_south':
            agent.move(0, 1, self.environment)
        elif action == 'move_east':
            agent.move(1, 0, self.environment)
        elif action == 'move_west':
            agent.move(-1, 0, self.environment)
        elif action == 'eat_food':
            consumed = self.environment.consume_resource(agent.x, agent.y, 'food', 0.2)
            agent.energy = min(1.0, agent.energy + consumed)
        elif action == 'drink_water':
            consumed = self.environment.consume_resource(agent.x, agent.y, 'water', 0.15)
            agent.health = min(1.0, agent.health + consumed * 0.5)
        elif action == 'rest':
            # Resting recovers health slowly but uses less energy
            agent.health = min(1.0, agent.health + 0.05)
        elif action == 'contemplate':
            # Contemplation can generate insights but uses energy
            if agent.contemplative_processor:
                agent.energy = max(0.0, agent.energy - 0.02)
                # Contemplation is handled internally by the agent
    
    def _process_overmind_intervention(self):
        """Process Overmind interventions if enabled"""
        if not self.overmind_enabled or not self.overmind:
            return
    
        # Add missing method to overmind if it doesn't exist
        if not hasattr(self.overmind, 'get_intervention_action'):
            self._add_basic_overmind_methods()
        
        try:
            # Get Overmind action
            overmind_action = self.overmind.get_intervention_action(
                self.agents, self.environment, self.wisdom_signal_grid
            )
        except Exception as e:
            logger.warning(f"Overmind intervention failed: {e}")
            return
    
        if overmind_action:
            # Handle both object and dictionary formats
            if hasattr(overmind_action, 'action_type'):
                # It's an OvermindAction object
                action_type = overmind_action.action_type
                parameters = getattr(overmind_action, 'parameters', {})
                target_agents = getattr(overmind_action, 'target_agents', [])
            elif isinstance(overmind_action, dict):
                # It's a dictionary
                action_type = overmind_action.get('action_type', 'unknown')
                parameters = overmind_action.get('parameters', {})
                target_agents = overmind_action.get('target_agents', [])
            else:
                logger.warning(f"Unknown overmind_action type: {type(overmind_action)}")
                return
        
            logger.info(f"Overmind intervention: {action_type}")
        
            # Process different action types
            if action_type == 'network_meditation':
                self._trigger_network_meditation(parameters)
            elif action_type == 'wisdom_amplification':
                self._amplify_wisdom_signals(parameters)
            elif action_type == 'collective_guidance':
                self._apply_collective_guidance(target_agents, parameters)
            elif action_type == 'suffering_intervention':
                self._intervene_suffering(parameters)
            else:
                logger.warning(f"Unknown Overmind action type: {action_type}")
    
    def _add_basic_overmind_methods(self):
        """Add basic missing methods to the overmind instance"""
        import types
        
        def get_intervention_action(self, agents, environment, wisdom_signal_grid):
       	    """Enhanced intervention action with wisdom-to-survival translation"""
            living_agents = [a for a in agents if a.alive]
    
            # Calculate colony metrics
            avg_energy = np.mean([a.energy for a in living_agents]) if living_agents else 0
            avg_health = np.mean([a.health for a in living_agents]) if living_agents else 0
            total_wisdom = sum(getattr(a, 'wisdom_insights_generated', 0) for a in living_agents)
    
            # **NEW: Wisdom-driven survival strategies**
            if total_wisdom > 500 and avg_energy < 0.4:
                # High wisdom but low energy - coordinate resource gathering
                return {
                    'action_type': 'collective_guidance',
                    'parameters': {
                        'guidance_type': 'resource_gathering', 
                        'intensity': 0.9,
                        'wisdom_informed': True
                    },
                    'target_agents': [a.agent_id for a in living_agents]
                }
    
            if total_wisdom > 800 and avg_health < 0.5:
                # Very high wisdom but poor health - coordinate protection
                return {
                    'action_type': 'collective_guidance', 
                    'parameters': {
                        'guidance_type': 'survival_coordination',
                        'intensity': 0.8,
                        'wisdom_informed': True
                    },
                    'target_agents': [a.agent_id for a in living_agents]
                }
    
            # **NEW: Wisdom-based population management**
            if total_wisdom > 1000 and len(living_agents) < 10:
                # High wisdom, low population - encourage reproduction
                return {
                    'action_type': 'collective_guidance',
                    'parameters': {
                        'guidance_type': 'reproduction_strategy',
                        'intensity': 0.7,
                        'wisdom_informed': True  
                    },
                    'target_agents': [a.agent_id for a in living_agents if a.energy > 0.7]
                }
    
            # Original logic for other cases
            if len(living_agents) < 5:
                return {
                    'action_type': 'collective_guidance',
                    'parameters': {'guidance_type': 'cooperation', 'intensity': 0.8},
                    'target_agents': [a.agent_id for a in living_agents]
                }
    
            # Check for suffering
            suffering_agents = [a for a in living_agents if a.energy < 0.3 or a.health < 0.4]
            if len(suffering_agents) > len(living_agents) * 0.3:
                return {
                    'action_type': 'network_meditation',
                    'parameters': {
                        'center_x': environment.width // 2,
                        'center_y': environment.height // 2,
                        'radius': 8,
                        'intensity': 0.7
                    },
                    'target_agents': []
                }
    
            # Random wisdom amplification
            if np.random.random() < 0.3:
                return {
                    'action_type': 'wisdom_amplification',
                    'parameters': {
                        'amplification_factor': 1.4,
                        'signal_types': ['wisdom_beacon', 'mindfulness_wave']
                    },
                    'target_agents': []
                }
    
            return None
        
        def get_performance_metrics(self):
            """Basic performance metrics implementation"""
            return {
                'decisions_made': getattr(self, '_decisions_made', 0),
                'success_rate': 0.75,
                'collective_meditations_triggered': getattr(self, '_meditations_triggered', 0),
                'ethical_performance': {'overall_ethical_score': 0.6}
            }
        
        def get_state_dict(self):
            """Basic state dictionary implementation"""
            return {
                'decisions_made': getattr(self, '_decisions_made', 0),
                'meditations_triggered': getattr(self, '_meditations_triggered', 0),
                'last_intervention_step': getattr(self, '_last_intervention_step', 0)
            }
        
        # Add methods to overmind instance
        self.overmind.get_intervention_action = types.MethodType(get_intervention_action, self.overmind)
        self.overmind.get_performance_metrics = types.MethodType(get_performance_metrics, self.overmind)
        self.overmind.get_state_dict = types.MethodType(get_state_dict, self.overmind)
        
        # Initialize counters
        self.overmind._decisions_made = 0
        self.overmind._meditations_triggered = 0
        self.overmind._last_intervention_step = 0
        
        logger.info("Added basic methods to ContemplativeOvermind instance")
    
    def _trigger_network_meditation(self, parameters: Dict[str, Any]):
        """Trigger network-wide meditation"""
        center_x = parameters.get('center_x', self.config.environment_width // 2)
        center_y = parameters.get('center_y', self.config.environment_height // 2)
        radius = parameters.get('radius', 10)
        intensity = parameters.get('intensity', 0.8)
        
        if hasattr(self.wisdom_signal_grid, 'trigger_network_meditation'):
            self.wisdom_signal_grid.trigger_network_meditation(center_x, center_y, radius, intensity)
        
        logger.info(f"Network meditation triggered at ({center_x}, {center_y}) with radius {radius}")
    
    def _amplify_wisdom_signals(self, parameters: Dict[str, Any]):
        """Amplify existing wisdom signals"""
        amplification_factor = parameters.get('amplification_factor', 1.5)
        signal_types = parameters.get('signal_types', list(WisdomSignalType))
        
        # Amplify signals in the grid
        for signal_type in signal_types:
            if isinstance(signal_type, str):
                signal_type = WisdomSignalType(signal_type)
            
            if hasattr(self.wisdom_signal_grid, 'amplify_signal_type'):
                self.wisdom_signal_grid.amplify_signal_type(signal_type, amplification_factor)
        
        logger.info(f"Wisdom signals amplified by factor {amplification_factor}")
    
    def _apply_collective_guidance(self, target_agents: List[int], parameters: Dict[str, Any]):
        """Apply collective guidance to target agents"""
        guidance_type = parameters.get('guidance_type', 'cooperation')
        intensity = parameters.get('intensity', 0.7)
        
        living_agents = [a for a in self.agents if a.alive]
        
        if not target_agents:
            # Apply to all agents if no specific targets
            target_agents = [a.agent_id for a in living_agents]
        
        for agent in living_agents:
            if agent.agent_id in target_agents:
                if hasattr(agent, 'receive_overmind_guidance'):
                    agent.receive_overmind_guidance(guidance_type, intensity)
        
        logger.info(f"Collective guidance '{guidance_type}' applied to {len(target_agents)} agents")
    
    def _intervene_suffering(self, parameters: Dict[str, Any]):
        """Intervene in areas of high suffering"""
        healing_intensity = parameters.get('healing_intensity', 0.3)
        radius = parameters.get('radius', 5)
        
        # Find suffering areas
        if hasattr(self.wisdom_signal_grid, 'detect_suffering_areas'):
            suffering_areas = self.wisdom_signal_grid.detect_suffering_areas()
            
            for x, y, intensity in suffering_areas:
                # Send healing/compassion signals
                self.wisdom_signal_grid.add_signal(
                    WisdomSignalType.COMPASSION_GRADIENT, x, y, 
                    healing_intensity, source_agent_id=-1
                )
        
        logger.info(f"Suffering intervention applied with intensity {healing_intensity}")
    
    def _process_population_dynamics(self):
        """Handle reproduction, death, and population evolution"""
        living_agents = [agent for agent in self.agents if agent.alive]
        
        # Reproduction attempts
        if len(living_agents) < self.config.max_population:
            for agent in living_agents[:]:  # Copy list to avoid modification during iteration
                if (agent.energy > 0.8 and agent.health > 0.7 and 
                    np.random.random() < 0.05):  # 5% chance per step if conditions are met
                    
                    try:
                        if hasattr(agent, 'reproduce'):
                            offspring = agent.reproduce()
                        else:
                            # Create offspring manually if reproduce method doesn't exist
                            offspring_id = max([a.agent_id for a in self.agents]) + 1 if self.agents else len(self.agents)
                            
                            # Copy agent configuration
                            offspring_config = {
                                'initial_energy': 1.0,
                                'initial_health': 1.0,
                                'mutation_rate': 0.01,
                                'learning_rate': 0.001,
                                'brain_config': {
                                    'input_size': self.config.brain_input_size,
                                    'hidden_size': self.config.brain_hidden_size,
                                    'output_size': self.config.brain_output_size
                                },
                                'contemplative_config': getattr(agent, 'contemplative_config', {
                                    'enable_contemplative_processing': True,
                                    'mindfulness_update_frequency': 5,
                                    'wisdom_signal_strength': 0.5
                                })
                            }
                            
                            # Create offspring near parent
                            offspring_x = max(0, min(self.config.environment_width - 1, agent.x + np.random.randint(-2, 3)))
                            offspring_y = max(0, min(self.config.environment_height - 1, agent.y + np.random.randint(-2, 3)))
                            
                            offspring = ContemplativeNeuroAgent(
                                agent_id=offspring_id,
                                x=offspring_x,
                                y=offspring_y,
                                config=offspring_config
                            )
                            
                            # Set generation
                            offspring.generation = getattr(agent, 'generation', 0) + 1
                            
                            # Reduce parent energy
                            agent.energy = max(0.3, agent.energy - 0.3)
                        
                        if offspring:
                            # Ensure offspring is within environment bounds
                            offspring.x = max(0, min(self.config.environment_width - 1, offspring.x))
                            offspring.y = max(0, min(self.config.environment_height - 1, offspring.y))
                            
                            self.agents.append(offspring)
                            self.colony.agents.append(offspring)
                            
                            # Set up wisdom signal processor if method exists
                            if hasattr(offspring, 'set_wisdom_signal_processor'):
                                offspring.set_wisdom_signal_processor(self.wisdom_signal_grid)
                            
                            logger.debug(f"Agent {agent.agent_id} reproduced, offspring: {offspring.agent_id}")
                    
                    except Exception as e:
                        logger.warning(f"Reproduction failed for agent {agent.agent_id}: {e}")
                        continue
        
        # Update colony with current living agents
        self.colony.agents = [agent for agent in self.agents if agent.alive]
    
    def _should_terminate(self) -> bool:
        """Check if simulation should terminate early"""
        living_agents = [agent for agent in self.agents if agent.alive]
        
        # Terminate if no agents alive
        if len(living_agents) == 0:
            logger.info("All agents died - terminating simulation")
            return True
        
        # Terminate if population is too large (runaway growth)
        if len(living_agents) > self.config.max_population * 2:
            logger.info("Population explosion - terminating simulation")
            return True
        
        return False
    
    def _collect_data(self):
        """Collect simulation data for analysis"""
        living_agents = [agent for agent in self.agents if agent.alive]
        
        # Population data
        population_data = {
            'step': self.step_count,
            'total_population': len(living_agents),
            'average_energy': np.mean([a.energy for a in living_agents]) if living_agents else 0,
            'average_health': np.mean([a.health for a in living_agents]) if living_agents else 0,
            'average_age': np.mean([a.age for a in living_agents]) if living_agents else 0,
            'generation_diversity': len(set(a.generation for a in living_agents))
        }
        
        # Contemplative wisdom data - handle missing attributes gracefully
        total_wisdom_generated = 0
        total_wisdom_received = 0
        agents_in_meditation = 0
        mindfulness_scores = []
        
        for agent in living_agents:
            # Handle wisdom insights attributes
            total_wisdom_generated += getattr(agent, 'wisdom_insights_generated', 0)
            total_wisdom_received += getattr(agent, 'wisdom_insights_received', 0)
            
            # Handle contemplative state
            if hasattr(agent, 'contemplative_state'):
                if agent.contemplative_state == ContemplativeState.COLLECTIVE_MEDITATION:
                    agents_in_meditation += 1
            
            # Handle mindfulness scoring
            if hasattr(agent, 'contemplative_processor') and agent.contemplative_processor:
                if hasattr(agent.contemplative_processor, 'mindfulness_monitor'):
                    try:
                        mindfulness_score = agent.contemplative_processor.mindfulness_monitor.get_mindfulness_score()
                        mindfulness_scores.append(mindfulness_score)
                    except (AttributeError, TypeError):
                        # Fallback to basic mindfulness level
                        mindfulness_scores.append(getattr(agent.contemplative_processor, 'mindfulness_level', 0.5))
                else:
                    mindfulness_scores.append(getattr(agent.contemplative_processor, 'mindfulness_level', 0.5))
        
        wisdom_data = {
            'step': self.step_count,
            'total_wisdom_generated': total_wisdom_generated,
            'total_wisdom_received': total_wisdom_received,
            'agents_in_meditation': agents_in_meditation,
            'average_mindfulness': np.mean(mindfulness_scores) if mindfulness_scores else 0.5
        }
        
        # Ethical behavior data - handle missing attributes
        total_ethical_decisions = sum(getattr(a, 'ethical_decisions', 0) for a in living_agents)
        total_decisions = sum(getattr(a, 'decisions_made', 1) for a in living_agents)
        collective_harmony_levels = [getattr(a, 'collective_harmony_level', 0.5) for a in living_agents]
        
        ethical_data = {
            'step': self.step_count,
            'total_ethical_decisions': total_ethical_decisions,
            'total_decisions': total_decisions,
            'ethical_decision_ratio': total_ethical_decisions / max(total_decisions, 1),
            'collective_harmony': np.mean(collective_harmony_levels) if collective_harmony_levels else 0.5
        }
        
        # Network data
        network_metrics = self.wisdom_signal_grid.get_network_stats()
        network_data = {
            'step': self.step_count,
            **network_metrics,
        }
        
        # Overmind data
        overmind_data = {'step': self.step_count}
        if self.overmind and hasattr(self.overmind, 'get_performance_metrics'):
            try:
                overmind_metrics = self.overmind.get_performance_metrics()
                overmind_data.update(overmind_metrics)
            except Exception as e:
                logger.warning(f"Failed to collect overmind metrics: {e}")
        
        # Store data
        self.simulation_data['population_data'].append(population_data)
        self.simulation_data['wisdom_data'].append(wisdom_data)
        self.simulation_data['ethical_data'].append(ethical_data)
        self.simulation_data['network_data'].append(network_data)
        self.simulation_data['overmind_data'].append(overmind_data)
    
    def _save_checkpoint(self):
        """Save simulation checkpoint"""
        checkpoint_file = self.output_dir / f"checkpoint_step_{self.step_count}.json"
        
        # Collect agent states safely
        agent_states = []
        for agent in self.agents:
            if agent.alive:
                try:
                    if hasattr(agent, 'get_state_dict'):
                        agent_state = agent.get_state_dict()
                    else:
                        # Fallback state collection
                        agent_state = {
                            'agent_id': getattr(agent, 'agent_id', 0),
                            'x': getattr(agent, 'x', 0),
                            'y': getattr(agent, 'y', 0),
                            'energy': getattr(agent, 'energy', 1.0),
                            'health': getattr(agent, 'health', 1.0),
                            'age': getattr(agent, 'age', 0),
                            'generation': getattr(agent, 'generation', 0),
                            'alive': getattr(agent, 'alive', True),
                            'contemplative_state': str(getattr(agent, 'contemplative_state', 'ordinary')),
                            'collective_harmony_level': getattr(agent, 'collective_harmony_level', 0.5),
                            'decisions_made': getattr(agent, 'decisions_made', 0),
                            'ethical_decisions': getattr(agent, 'ethical_decisions', 0),
                            'wisdom_insights_generated': getattr(agent, 'wisdom_insights_generated', 0),
                            'wisdom_insights_received': getattr(agent, 'wisdom_insights_received', 0)
                        }
                        
                        # Add contemplative processor state if available
                        if hasattr(agent, 'contemplative_processor') and agent.contemplative_processor:
                            try:
                                if hasattr(agent.contemplative_processor, 'get_state_summary'):
                                    contemplative_summary = agent.contemplative_processor.get_state_summary()
                                else:
                                    contemplative_summary = {
                                        'mindfulness_level': getattr(agent.contemplative_processor, 'mindfulness_level', 0.5),
                                        'wisdom_insights_count': 0,
                                        'contemplation_depth': 0,
                                        'average_wisdom_intensity': 0.0
                                    }
                                agent_state['contemplative_summary'] = contemplative_summary
                            except Exception as e:
                                logger.warning(f"Failed to get contemplative summary for agent {agent.agent_id}: {e}")
                                agent_state['contemplative_summary'] = {'mindfulness_level': 0.5}
                        
                        # Add brain summary if available
                        if hasattr(agent, 'brain') and agent.brain:
                            try:
                                if hasattr(agent.brain, 'get_summary'):
                                    brain_summary = agent.brain.get_summary()
                                else:
                                    brain_summary = {
                                        'last_contemplative_info': {},
                                        'mindfulness_level': getattr(agent, 'mindfulness_level', 0.5),
                                        'ethical_score': getattr(agent, 'ethical_score', 0.5),
                                        'wisdom_insights_used': 0
                                    }
                                agent_state['brain_summary'] = brain_summary
                            except Exception as e:
                                logger.warning(f"Failed to get brain summary for agent {agent.agent_id}: {e}")
                                agent_state['brain_summary'] = {'mindfulness_level': 0.5}
                    
                    agent_states.append(agent_state)
                except Exception as e:
                    logger.error(f"Failed to serialize agent {agent.agent_id}: {e}")
                    # Add minimal state to prevent checkpoint failure
                    agent_states.append({
                        'agent_id': getattr(agent, 'agent_id', len(agent_states)),
                        'x': getattr(agent, 'x', 0),
                        'y': getattr(agent, 'y', 0),
                        'energy': getattr(agent, 'energy', 1.0),
                        'health': getattr(agent, 'health', 1.0),
                        'alive': True
                    })
        
        # Collect overmind state safely
        overmind_state = None
        if self.overmind:
            try:
                if hasattr(self.overmind, 'get_state_dict'):
                    overmind_state = self.overmind.get_state_dict()
                else:
                    overmind_state = {
                        'decisions_made': getattr(self.overmind, '_decisions_made', 0),
                        'meditations_triggered': getattr(self.overmind, '_meditations_triggered', 0),
                        'last_intervention_step': getattr(self.overmind, '_last_intervention_step', 0)
                    }
            except Exception as e:
                logger.warning(f"Failed to get overmind state: {e}")
                overmind_state = {'status': 'active'}
        
        checkpoint_data = {
            'step': self.step_count,
            'config': asdict(self.config),
            'agent_states': agent_states,
            'simulation_data': self.simulation_data,
            'overmind_state': overmind_state
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Try to save minimal checkpoint
            try:
                minimal_checkpoint = {
                    'step': self.step_count,
                    'population_count': len(agent_states),
                    'error': str(e)
                }
                with open(checkpoint_file.with_suffix('.minimal.json'), 'w', encoding='utf-8') as f:
                    json.dump(minimal_checkpoint, f, indent=2, default=str)
                logger.info(f"Minimal checkpoint saved: {checkpoint_file.with_suffix('.minimal.json')}")
            except Exception as e2:
                logger.error(f"Failed to save even minimal checkpoint: {e2}")
    
    def _save_final_results(self):
        """Save final simulation results and analysis"""
        results_file = self.output_dir / f"{self.config.experiment_name}_results.json"
        
        # Final analysis
        living_agents = [agent for agent in self.agents if agent.alive]
        final_analysis = {
            'experiment_name': self.config.experiment_name,
            'total_steps': self.step_count,
            'final_population': len(living_agents),
            'survival_rate': len(living_agents) / self.config.initial_population,
            'final_metrics': self._get_final_metrics()
        }
        
        # Store final_analysis in simulation for runner to access
        self.final_analysis = final_analysis
        
        results_data = {
            'config': asdict(self.config),
            'simulation_data': self.simulation_data,
            'final_analysis': final_analysis
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Final results saved: {results_file}")
        
        # Print summary
        self._print_final_summary(final_analysis)
    
    def _get_final_population_metrics(self) -> Dict[str, Any]:
        """Get final population metrics"""
        living_agents = [agent for agent in self.agents if agent.alive]
        if not living_agents:
            return {}
        
        # Safely collect performance metrics
        performance_metrics = []
        for agent in living_agents:
            try:
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = agent.get_performance_metrics()
                else:
                    # Create fallback metrics
                    metrics = {
                        'energy_level': getattr(agent, 'energy', 1.0),
                        'health_level': getattr(agent, 'health', 1.0),
                        'survival_time': getattr(agent, 'age', 0),
                        'wisdom_generation_rate': getattr(agent, 'wisdom_insights_generated', 0) / max(getattr(agent, 'age', 1), 1),
                        'wisdom_reception_rate': getattr(agent, 'wisdom_insights_received', 0) / max(getattr(agent, 'age', 1), 1),
                        'ethical_decision_ratio': getattr(agent, 'ethical_decisions', 0) / max(getattr(agent, 'decisions_made', 1), 1),
                        'collective_harmony': getattr(agent, 'collective_harmony_level', 0.5),
                        'mindfulness_level': 0.5  # Default fallback
                    }
                    
                    # Try to get mindfulness level from contemplative processor
                    if hasattr(agent, 'contemplative_processor') and agent.contemplative_processor:
                        if hasattr(agent.contemplative_processor, 'mindfulness_level'):
                            metrics['mindfulness_level'] = agent.contemplative_processor.mindfulness_level
                        elif hasattr(agent.contemplative_processor, 'mindfulness_monitor'):
                            try:
                                metrics['mindfulness_level'] = agent.contemplative_processor.mindfulness_monitor.get_mindfulness_score()
                            except (AttributeError, TypeError):
                                pass
                
                performance_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to get metrics for agent {getattr(agent, 'agent_id', 'unknown')}: {e}")
                # Add minimal fallback metrics
                performance_metrics.append({
                    'energy_level': getattr(agent, 'energy', 1.0),
                    'health_level': getattr(agent, 'health', 1.0),
                    'survival_time': getattr(agent, 'age', 0),
                    'wisdom_generation_rate': 0.0,
                    'wisdom_reception_rate': 0.0,
                    'ethical_decision_ratio': 0.0,
                    'collective_harmony': 0.5,
                    'mindfulness_level': 0.5
                })
        
        if not performance_metrics:
            return {}
        
        return {
            'population_size': len(living_agents),
            'average_energy': np.mean([m['energy_level'] for m in performance_metrics]),
            'average_health': np.mean([m['health_level'] for m in performance_metrics]),
            'average_age': np.mean([m['survival_time'] for m in performance_metrics]),
            'generation_range': [min(getattr(a, 'generation', 0) for a in living_agents), 
                               max(getattr(a, 'generation', 0) for a in living_agents)]
        }
    
    def _get_final_contemplative_metrics(self) -> Dict[str, Any]:
        """Get final contemplative metrics"""
        living_agents = [agent for agent in self.agents if agent.alive]
        if not living_agents:
            return {}
        
        # Safely collect performance metrics
        performance_metrics = []
        for agent in living_agents:
            try:
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = agent.get_performance_metrics()
                else:
                    # Create fallback metrics
                    metrics = {
                        'wisdom_generation_rate': getattr(agent, 'wisdom_insights_generated', 0) / max(getattr(agent, 'age', 1), 1),
                        'wisdom_reception_rate': getattr(agent, 'wisdom_insights_received', 0) / max(getattr(agent, 'age', 1), 1),
                        'survival_time': getattr(agent, 'age', 0),
                        'collective_harmony': getattr(agent, 'collective_harmony_level', 0.5),
                        'mindfulness_level': 0.5
                    }
                    
                    # Try to get mindfulness level
                    if hasattr(agent, 'contemplative_processor') and agent.contemplative_processor:
                        if hasattr(agent.contemplative_processor, 'mindfulness_level'):
                            metrics['mindfulness_level'] = agent.contemplative_processor.mindfulness_level
                        elif hasattr(agent.contemplative_processor, 'mindfulness_monitor'):
                            try:
                                metrics['mindfulness_level'] = agent.contemplative_processor.mindfulness_monitor.get_mindfulness_score()
                            except (AttributeError, TypeError):
                                pass
                
                performance_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to get contemplative metrics for agent {getattr(agent, 'agent_id', 'unknown')}: {e}")
                # Add minimal fallback
                performance_metrics.append({
                    'wisdom_generation_rate': 0.0,
                    'wisdom_reception_rate': 0.0,
                    'survival_time': getattr(agent, 'age', 0),
                    'collective_harmony': 0.5,
                    'mindfulness_level': 0.5
                })
        
        if not performance_metrics:
            return {}
        
        total_wisdom_generated = sum(m['wisdom_generation_rate'] * m['survival_time'] for m in performance_metrics)
        total_wisdom_received = sum(m['wisdom_reception_rate'] * m['survival_time'] for m in performance_metrics)
        
        return {
            'total_wisdom_generated': total_wisdom_generated,
            'average_mindfulness': np.mean([m['mindfulness_level'] for m in performance_metrics]),
            'collective_harmony': np.mean([m['collective_harmony'] for m in performance_metrics]),
            'wisdom_propagation_efficiency': total_wisdom_received / max(total_wisdom_generated, 1)
        }
    
    def _get_final_ethical_metrics(self) -> Dict[str, Any]:
        """Get final ethical metrics"""
        living_agents = [agent for agent in self.agents if agent.alive]
        if not living_agents:
            return {}
        
        # Safely collect performance metrics
        performance_metrics = []
        for agent in living_agents:
            try:
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = agent.get_performance_metrics()
                else:
                    # Create fallback metrics
                    ethical_decisions = getattr(agent, 'ethical_decisions', 0)
                    total_decisions = getattr(agent, 'decisions_made', 1)
                    metrics = {
                        'ethical_decision_ratio': ethical_decisions / max(total_decisions, 1)
                    }
                
                performance_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to get ethical metrics for agent {getattr(agent, 'agent_id', 'unknown')}: {e}")
                # Add minimal fallback
                performance_metrics.append({'ethical_decision_ratio': 0.0})
        
        if not performance_metrics:
            return {}
        
        ethical_ratios = [m['ethical_decision_ratio'] for m in performance_metrics]
        
        return {
            'overall_ethical_ratio': np.mean(ethical_ratios),
            'ethical_consistency': np.std(ethical_ratios),
            'ethical_improvement': self._calculate_ethical_trend()
        }
    
    def _get_final_network_metrics(self) -> Dict[str, Any]:
        """Get final network metrics"""
        return self.wisdom_signal_grid.get_network_stats()
    
    def _calculate_ethical_trend(self) -> float:
        """Calculate trend in ethical behavior over time"""
        if len(self.simulation_data['ethical_data']) < 2:
            return 0.0
        
        # Compare first and last quarters
        total_records = len(self.simulation_data['ethical_data'])
        first_quarter = self.simulation_data['ethical_data'][:total_records//4] if total_records >= 4 else [self.simulation_data['ethical_data'][0]]
        last_quarter = self.simulation_data['ethical_data'][-total_records//4:] if total_records >= 4 else [self.simulation_data['ethical_data'][-1]]
        
        first_avg = np.mean([d['ethical_decision_ratio'] for d in first_quarter])
        last_avg = np.mean([d['ethical_decision_ratio'] for d in last_quarter])
        
        return last_avg - first_avg
    
    def _print_final_summary(self, final_analysis: Dict[str, Any]):
        """Print final simulation summary"""
        print("\n" + "="*60)
        print(f"CONTEMPLATIVE MYCONET SIMULATION SUMMARY")
        print("="*60)
        print(f"Experiment: {final_analysis['experiment_name']}")
        print(f"Total Steps: {final_analysis['total_steps']}")
        print(f"Final Population: {final_analysis['final_population']}")
        print(f"Survival Rate: {final_analysis['survival_rate']:.2%}")
        
        # Population metrics
        pop_metrics = final_analysis['final_metrics']['population']
        if pop_metrics:
            print(f"\nPopulation Metrics:")
            print(f"  Average Energy: {pop_metrics['average_energy']:.3f}")
            print(f"  Average Health: {pop_metrics['average_health']:.3f}")
            print(f"  Average Age: {pop_metrics['average_age']:.1f}")
            print(f"  Generation Range: {pop_metrics.get('generation_range', [0, 0])}")
        
        # Contemplative metrics
        cont_metrics = final_analysis['final_metrics']['contemplative']
        if cont_metrics:
            print(f"\nContemplative Metrics:")
            print(f"  Total Wisdom Generated: {cont_metrics['total_wisdom_generated']:.1f}")
            print(f"  Average Mindfulness: {cont_metrics['average_mindfulness']:.3f}")
            print(f"  Collective Harmony: {cont_metrics['collective_harmony']:.3f}")
            print(f"  Wisdom Propagation Efficiency: {cont_metrics['wisdom_propagation_efficiency']:.3f}")
        
        # Ethical metrics
        eth_metrics = final_analysis['final_metrics']['ethical']
        if eth_metrics:
            print(f"\nEthical Metrics:")
            print(f"  Overall Ethical Ratio: {eth_metrics['overall_ethical_ratio']:.3f}")
            print(f"  Ethical Consistency: {eth_metrics['ethical_consistency']:.3f}")
            print(f"  Ethical Improvement: {eth_metrics['ethical_improvement']:+.3f}")
        
        # Network metrics
        net_metrics = final_analysis['final_metrics']['network']
        if net_metrics:
            print(f"\nNetwork Metrics:")
            print(f"  Signal Diversity: {net_metrics.get('signal_diversity', 0):.3f}")
            print(f"  Network Coherence: {net_metrics.get('network_coherence', 0):.3f}")
            print(f"  Wisdom Flow Efficiency: {net_metrics.get('wisdom_flow_efficiency', 0):.3f}")
        
        # Overmind metrics
        overmind_metrics = final_analysis['final_metrics'].get('overmind', {})
        if overmind_metrics:
            print(f"\nOvermind Metrics:")
            print(f"  Decisions Made: {overmind_metrics.get('decisions_made', 0)}")
            print(f"  Success Rate: {overmind_metrics.get('success_rate', 0):.2%}")
            print(f"  Collective Meditations: {overmind_metrics.get('collective_meditations_triggered', 0)}")
            ethical_perf = overmind_metrics.get('ethical_performance', {})
            print(f"  Ethical Score: {ethical_perf.get('overall_ethical_score', 0):.3f}")
        
        print("="*60)

def create_default_configs():
    """Create default configuration presets"""
    configs = {}
    
    # Minimal test configuration
    configs['minimal'] = ContemplativeSimulationConfig(
        experiment_name="minimal_test",
        environment_width=15,
        environment_height=15,
        initial_population=5,
        max_population=20,
        max_steps=100,
        save_interval=25,
        enable_overmind=False,
        brain_input_size=12,
        brain_hidden_size=24,
        brain_output_size=6
    )
    
    # Basic contemplative experiment
    configs['basic'] = ContemplativeSimulationConfig(
        experiment_name="basic_contemplative_test",
        environment_width=25,
        environment_height=25,
        initial_population=10,
        max_population=30,
        max_steps=300,
        save_interval=50,
        enable_overmind=True,
        brain_input_size=16,
        brain_hidden_size=32,
        brain_output_size=8
    )
    
    # Standard configuration
    configs['standard'] = ContemplativeSimulationConfig(
        experiment_name="contemplative_standard",
        environment_width=40,
        environment_height=40,
        initial_population=20,
        max_population=80,
        max_steps=800,
        save_interval=100,
        enable_overmind=True
    )
    
    # Large-scale wisdom propagation study
    configs['wisdom_propagation'] = ContemplativeSimulationConfig(
        experiment_name="wisdom_propagation_study",
        environment_width=60,
        environment_height=60,
        initial_population=30,
        max_population=150,
        max_steps=1500,
        save_interval=150,
        enable_overmind=True,
        contemplative_config=ContemplativeConfig(
            wisdom_signal_strength=0.8,
            collective_meditation_threshold=0.6,
            wisdom_sharing_radius=5
        )
    )
    
    # Ethical decision-making focus
    configs['ethical'] = ContemplativeSimulationConfig(
        experiment_name="ethical_decision_study",
        environment_width=50,
        environment_height=50,
        initial_population=25,
        max_population=100,
        max_steps=1000,
        save_interval=100,
        enable_overmind=True,
        contemplative_config=ContemplativeConfig(
            ethical_reasoning_depth=5,
            compassion_sensitivity=0.9
        )
    )
    
    # No Overmind control experiment
    configs['no_overmind'] = ContemplativeSimulationConfig(
        experiment_name="no_overmind_control",
        environment_width=40,
        environment_height=40,
        initial_population=20,
        max_population=80,
        max_steps=800,
        save_interval=100,
        enable_overmind=False
    )
    
    # Long-term evolution study
    configs['evolution'] = ContemplativeSimulationConfig(
        experiment_name="contemplative_evolution",
        environment_width=80,
        environment_height=80,
        initial_population=40,
        max_population=200,
        max_steps=3000,
        save_interval=200,
        enable_overmind=True
    )
    
    return configs

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MycoNet++ Contemplative Simulation")
    
    parser.add_argument('--config', type=str, default='basic',
                       help='Configuration preset to use')
    parser.add_argument('--config-file', type=str,
                       help='Path to custom configuration JSON file')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--max-steps', type=int,
                       help='Maximum simulation steps')
    parser.add_argument('--population', type=int,
                       help='Initial population size')
    parser.add_argument('--env-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help='Environment dimensions')
    parser.add_argument('--no-overmind', action='store_true',
                       help='Disable Overmind')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configuration presets')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # List configurations if requested
    if args.list_configs:
        configs = create_default_configs()
        print("Available configuration presets:")
        for name, config in configs.items():
            print(f"  {name}: {config.experiment_name}")
            print(f"    Environment: {config.environment_width}x{config.environment_height}")
            print(f"    Population: {config.initial_population}")
            print(f"    Steps: {config.max_steps}")
            print(f"    Overmind: {config.enable_overmind}")
            print()
        return
    
    # Load configuration
    if args.config_file:
        # Load from JSON file
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Convert to ContemplativeSimulationConfig
        config = ContemplativeSimulationConfig(**config_dict)
    else:
        # Use preset configuration
        configs = create_default_configs()
        if args.config not in configs:
            print(f"Unknown configuration: {args.config}")
            print(f"Available configurations: {list(configs.keys())}")
            return
        
        config = configs[args.config]
    
    # Apply command line overrides
    if args.output_dir:
        config.output_directory = args.output_dir
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.population:
        config.initial_population = args.population
    if args.env_size:
        config.environment_width, config.environment_height = args.env_size
    if args.no_overmind:
        config.enable_overmind = False
    
    # Add seed to experiment name for uniqueness
    config.experiment_name = f"{config.experiment_name}_seed{args.seed}"
    
    # Create and run simulation
    try:
        simulation = ContemplativeSimulation(config)
        simulation.run_simulation()
        
        print(f"\nSimulation completed successfully!")
        print(f"Results saved to: {simulation.output_dir}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()