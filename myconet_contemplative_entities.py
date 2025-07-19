"""
MycoNet++ Contemplative Entities (REFINED VERSION)
==================================================

Enhanced agent classes with contemplative capabilities.
REFINED with type validation, GPU support, and comprehensive logging.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union, Type
import logging
from dataclasses import dataclass, asdict, field
import time
from copy import deepcopy
import random
from enum import Enum
import warnings

# Import contemplative modules
from myconet_contemplative_core import (
    ContemplativeProcessor, ContemplativeState, WisdomInsight, WisdomType
)
from myconet_wisdom_signals import (
    WisdomSignalGrid, WisdomSignalProcessor, WisdomSignalType
)
from myconet_contemplative_brains import ContemplativeBrain, create_contemplative_brain

# Enhanced logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create debug logger for detailed decision tracing
decision_logger = logging.getLogger(f"{__name__}.decisions")
decision_logger.setLevel(logging.DEBUG)

# Create performance logger for metrics
performance_logger = logging.getLogger(f"{__name__}.performance")
performance_logger.setLevel(logging.INFO)

class ActionType(Enum):
    """Types of actions an agent can take - CENTRALIZED DEFINITION"""
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    EAT_FOOD = 4
    COLLECT_WATER = 5
    REST = 6
    REPRODUCE = 7
    MEDITATE = 8
    SHARE_WISDOM = 9
    HELP_OTHER = 10
    EXPLORE = 11
    NO_ACTION = 12  # Added for safety

    @classmethod
    def validate(cls, action: Union[str, 'ActionType', int]) -> 'ActionType':
        """Validate and convert action to ActionType enum"""
        if isinstance(action, cls):
            return action
        elif isinstance(action, str):
            # Handle string representations
            action_upper = action.upper()
            for action_type in cls:
                if action_type.name == action_upper:
                    return action_type
            logger.warning(f"Unknown action string: {action}, defaulting to NO_ACTION")
            return cls.NO_ACTION
        elif isinstance(action, int):
            try:
                return cls(action)
            except ValueError:
                logger.warning(f"Invalid action integer: {action}, defaulting to NO_ACTION")
                return cls.NO_ACTION
        else:
            logger.warning(f"Invalid action type: {type(action)}, defaulting to NO_ACTION")
            return cls.NO_ACTION

@dataclass
class ContemplativeConfig:
    """Configuration for contemplative agent parameters - TYPE VALIDATED"""
    # Core contemplative parameters
    compassion_sensitivity: float = field(default=0.6)
    ethical_reasoning_depth: int = field(default=1)
    mindfulness_update_frequency: int = field(default=20)
    wisdom_sharing_threshold: float = field(default=0.3)
    
    # Learning parameters
    ethical_learning_rate: float = field(default=0.01)
    mindfulness_decay_rate: float = field(default=0.02)
    wisdom_accumulation_rate: float = field(default=0.05)
    
    # Behavioral parameters
    cooperation_tendency: float = field(default=0.5)
    exploration_bias: float = field(default=0.3)
    risk_tolerance: float = field(default=0.4)
    
    # Genetic parameters (for evolution)
    mutation_rate: float = field(default=0.1)
    mutation_strength: float = field(default=0.05)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate all configuration parameters"""
        # Validate float parameters (0.0 to 1.0)
        float_params_0_1 = [
            'compassion_sensitivity', 'wisdom_sharing_threshold', 'ethical_learning_rate',
            'mindfulness_decay_rate', 'wisdom_accumulation_rate', 'cooperation_tendency',
            'exploration_bias', 'risk_tolerance', 'mutation_rate', 'mutation_strength'
        ]
        
        for param in float_params_0_1:
            value = getattr(self, param)
            if not isinstance(value, (int, float)):
                logger.warning(f"Parameter {param} should be numeric, got {type(value)}")
                setattr(self, param, 0.5)  # Default fallback
            elif not 0.0 <= value <= 1.0:
                logger.warning(f"Parameter {param} should be between 0.0 and 1.0, got {value}")
                setattr(self, param, np.clip(value, 0.0, 1.0))
        
        # Validate integer parameters
        int_params = ['ethical_reasoning_depth', 'mindfulness_update_frequency']
        for param in int_params:
            value = getattr(self, param)
            if not isinstance(value, int):
                logger.warning(f"Parameter {param} should be integer, got {type(value)}")
                setattr(self, param, int(value) if isinstance(value, (int, float)) else 1)
            elif value < 1:
                logger.warning(f"Parameter {param} should be positive, got {value}")
                setattr(self, param, max(1, value))
    
    def mutate(self) -> 'ContemplativeConfig':
        """Create a mutated copy of this configuration - TYPE SAFE"""
        new_config = deepcopy(self)
        
        # List of mutable parameters with their types and bounds
        mutable_params = {
            'compassion_sensitivity': ('float', 0.0, 1.0),
            'wisdom_sharing_threshold': ('float', 0.0, 1.0),
            'ethical_learning_rate': ('float', 0.001, 0.1),
            'mindfulness_decay_rate': ('float', 0.001, 0.1),
            'wisdom_accumulation_rate': ('float', 0.001, 0.1),
            'cooperation_tendency': ('float', 0.0, 1.0),
            'exploration_bias': ('float', 0.0, 1.0),
            'risk_tolerance': ('float', 0.0, 1.0),
            'mindfulness_update_frequency': ('int', 1, 100)
        }
        
        for param, (param_type, min_val, max_val) in mutable_params.items():
            if random.random() < self.mutation_rate:
                current_value = getattr(new_config, param)
                
                if param_type == 'float':
                    mutation = np.random.normal(0, self.mutation_strength)
                    new_value = np.clip(current_value + mutation, min_val, max_val)
                    setattr(new_config, param, float(new_value))
                
                elif param_type == 'int':
                    mutation = int(np.random.normal(0, 2))
                    new_value = np.clip(current_value + mutation, min_val, max_val)
                    setattr(new_config, param, int(new_value))
        
        # Re-validate after mutation
        new_config._validate_parameters()
        return new_config

@dataclass
class ContemplativeDecision:
    """Rich decision structure from contemplative brain - TYPE VALIDATED"""
    chosen_action: ActionType
    action_probabilities: Dict[ActionType, float] = field(default_factory=dict)
    ethical_evaluation: Dict[str, float] = field(default_factory=dict)
    mindfulness_state: Dict[str, float] = field(default_factory=dict)
    wisdom_insights: List[str] = field(default_factory=list)
    confidence: float = field(default=0.5)
    contemplative_override: bool = field(default=False)
    reasoning_trace: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate decision structure after initialization"""
        self._validate_decision()
    
    def _validate_decision(self):
        """Validate all decision components"""
        # Validate chosen_action
        if not isinstance(self.chosen_action, ActionType):
            logger.warning(f"chosen_action should be ActionType, got {type(self.chosen_action)}")
            self.chosen_action = ActionType.validate(self.chosen_action)
        
        # Validate action_probabilities
        if not isinstance(self.action_probabilities, dict):
            logger.warning("action_probabilities should be dict")
            self.action_probabilities = {}
        else:
            # Ensure all keys are ActionType and values are float
            validated_probs = {}
            for action, prob in self.action_probabilities.items():
                validated_action = ActionType.validate(action)
                validated_prob = float(prob) if isinstance(prob, (int, float)) else 0.0
                validated_probs[validated_action] = np.clip(validated_prob, 0.0, 1.0)
            self.action_probabilities = validated_probs
        
        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            logger.warning(f"confidence should be numeric, got {type(self.confidence)}")
            self.confidence = 0.5
        else:
            self.confidence = np.clip(float(self.confidence), 0.0, 1.0)
        
        # Validate boolean flags
        self.contemplative_override = bool(self.contemplative_override)
        
        # Validate lists
        if not isinstance(self.wisdom_insights, list):
            self.wisdom_insights = []
        if not isinstance(self.reasoning_trace, list):
            self.reasoning_trace = []
        
        # Validate dictionaries
        if not isinstance(self.ethical_evaluation, dict):
            self.ethical_evaluation = {}
        if not isinstance(self.mindfulness_state, dict):
            self.mindfulness_state = {}

class DeviceManager:
    """GPU/Device-Agnostic Handling for Neural Components"""
    
    def __init__(self, prefer_gpu: bool = True):
        self.device = self._detect_device(prefer_gpu)
        self.prefer_gpu = prefer_gpu
        logger.info(f"DeviceManager initialized with device: {self.device}")
    
    def _detect_device(self, prefer_gpu: bool) -> torch.device:
        """Detect best available device"""
        if prefer_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon GPU
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            if prefer_gpu:
                logger.info("GPU requested but not available, using CPU")
            else:
                logger.info("Using CPU as requested")
        
        return device
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move tensor or model to appropriate device"""
        try:
            return tensor_or_model.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to move to device {self.device}: {e}")
            return tensor_or_model
    
    def ensure_tensor_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on correct device"""
        if tensor.device != self.device:
            return self.to_device(tensor)
        return tensor

# Global device manager instance
device_manager = DeviceManager()

class ContemplativeNeuroAgent:
    """
    Enhanced NeuroAgent with contemplative capabilities
    REFINED with type validation, GPU support, and comprehensive logging
    """
    
    def __init__(self, agent_id: Union[int, str], x: int, y: int, config: Dict[str, Any]):
        # Validate and convert agent_id
        self.agent_id = str(agent_id)  # Ensure string representation
        self.x = int(x)
        self.y = int(y)
        self.config = config
        
        # Initialize logging for this agent
        self.agent_logger = logging.getLogger(f"{__name__}.agent_{self.agent_id}")
        self.agent_logger.setLevel(logging.DEBUG)
        
        decision_logger.debug(f"Initializing agent {self.agent_id} at ({self.x}, {self.y})")
        
        # Basic agent properties (validated)
        self.energy = max(0.0, min(1.0, float(config.get('initial_energy', 1.0))))
        self.health = max(0.0, min(1.0, float(config.get('initial_health', 1.0))))
        self.water = max(0.0, min(1.0, float(config.get('initial_water', 1.0))))
        self.age = 0
        self.generation = 0
        self.alive = True
        
        # Neural network brain with device management
        brain_config = config.get('brain_config', {})
        try:
            self.brain = create_contemplative_brain(
                brain_type='individual',
                input_size=brain_config.get('input_size', 12),
                hidden_size=brain_config.get('hidden_size', 64),
                output_size=brain_config.get('output_size', 6),
                contemplative_config=config.get('contemplative_config', {})
            )
            
            # Move brain to appropriate device if it has parameters
            if hasattr(self.brain, 'parameters') and any(self.brain.parameters()):
                self.brain = device_manager.to_device(self.brain)
                self.agent_logger.debug(f"Brain moved to device: {device_manager.device}")
        except Exception as e:
            logger.error(f"Failed to create brain for agent {self.agent_id}: {e}")
            self.brain = None
        
        # Contemplative configuration with validation
        contemplative_config = config.get('contemplative_config', {})
        if isinstance(contemplative_config, dict):
            try:
                self.contemplative_config = ContemplativeConfig(**contemplative_config)
            except Exception as e:
                logger.warning(f"Invalid contemplative config for agent {self.agent_id}: {e}")
                self.contemplative_config = ContemplativeConfig()
        elif isinstance(contemplative_config, ContemplativeConfig):
            self.contemplative_config = contemplative_config
        else:
            logger.warning(f"Invalid contemplative config type: {type(contemplative_config)}")
            self.contemplative_config = ContemplativeConfig()
        
        # Contemplative processing
        if contemplative_config.get('enable_contemplative_processing', True):
            try:
                self.contemplative_processor = ContemplativeProcessor(self.agent_id, asdict(self.contemplative_config))
                if hasattr(self.brain, 'set_contemplative_processor'):
                    self.brain.set_contemplative_processor(self.contemplative_processor)
            except Exception as e:
                logger.error(f"Failed to create contemplative processor for agent {self.agent_id}: {e}")
                self.contemplative_processor = None
        else:
            self.contemplative_processor = None
        
        # Contemplative state (validated)
        self.mindfulness_level = 0.5
        self.wisdom_accumulated = 0.0
        self.ethical_violations = 0
        self.contemplative_insights = []
        
        # Decision history for learning
        self.decision_history = []
        self.action_outcomes = []
        
        # Social state
        self.reputation = 0.5
        self.relationships = {}
        
        # Internal counters
        self.steps_since_mindfulness_update = 0
        self.steps_since_reproduction = 0
        self.reproduction_cooldown = 100
        
        # Wisdom signal processing
        self.wisdom_signal_processor = None  # Will be set when grid is available
        
        # Agent memory and learning (compatible with existing system)
        self.memory = []
        self.mutation_rate = max(0.0, min(1.0, float(config.get('mutation_rate', 0.01))))
        self.learning_rate = max(0.0, min(1.0, float(config.get('learning_rate', 0.001))))
        
        # Contemplative state tracking
        self.contemplative_state = ContemplativeState.ORDINARY
        self.meditation_timer = 0
        self.last_wisdom_share = 0
        self.collective_harmony_level = 0.5
        
        # Performance metrics
        self.decisions_made = 0
        self.ethical_decisions = 0
        self.wisdom_insights_generated = 0
        self.wisdom_insights_received = 0
        
        # Performance tracking for logging
        self.performance_history = []
        
        self.agent_logger.info(f"Agent {self.agent_id} initialized successfully")
    
    def set_wisdom_signal_processor(self, signal_grid: WisdomSignalGrid):
        """Set the wisdom signal processor with reference to the grid - TYPE VALIDATED"""
        if not isinstance(signal_grid, WisdomSignalGrid):
            logger.error(f"signal_grid must be WisdomSignalGrid, got {type(signal_grid)}")
            return
        
        try:
            self.wisdom_signal_processor = WisdomSignalProcessor(self.agent_id, signal_grid)
            self.agent_logger.debug("Wisdom signal processor set successfully")
        except Exception as e:
            logger.error(f"Failed to set wisdom signal processor: {e}")
    
    def update(self, environment, other_agents: List['ContemplativeNeuroAgent']) -> Dict[str, Any]:
        """
        ENHANCED: Complete update loop with comprehensive logging and type validation
        """
        
        update_start_time = time.time()
        self.agent_logger.debug(f"Starting update cycle {self.age + 1}")
        
        if not self.alive:
            return {'action_taken': 'dead', 'success': False, 'reason': 'agent_dead'}
        
        # Validate inputs
        if not isinstance(other_agents, list):
            logger.warning(f"other_agents should be list, got {type(other_agents)}")
            other_agents = []
        
        self.age += 1
        self.decisions_made += 1
        self.steps_since_mindfulness_update += 1
        self.steps_since_reproduction += 1
        
        try:
            # Step 1: Gather observations
            decision_logger.debug(f"Agent {self.agent_id}: Gathering observations")
            observations = self._gather_observations(environment, other_agents)
            
            # Step 2: Make contemplative decision using brain
            decision_logger.debug(f"Agent {self.agent_id}: Making contemplative decision")
            decision = self._make_contemplative_decision(observations)
            
            # Step 3: Process decision contemplatively
            decision_logger.debug(f"Agent {self.agent_id}: Processing decision contemplatively")
            self._process_contemplative_decision(decision, observations)
            
            # Step 4: Execute action in environment
            decision_logger.debug(f"Agent {self.agent_id}: Executing action {decision.chosen_action.name}")
            action_outcome = self._execute_action(decision.chosen_action, environment, other_agents)
            
            # Step 5: Update internal state based on decision and outcome
            decision_logger.debug(f"Agent {self.agent_id}: Updating contemplative state")
            self._update_contemplative_state(decision, action_outcome, observations)
            
            # Step 6: Learn from the experience
            decision_logger.debug(f"Agent {self.agent_id}: Learning from experience")
            self._learn_from_experience(decision, action_outcome, observations)
            
            # Step 7: Update relationships and reputation
            decision_logger.debug(f"Agent {self.agent_id}: Updating social state")
            self._update_social_state(decision, action_outcome, other_agents)
            
            # Step 8: Handle reproduction if appropriate
            decision_logger.debug(f"Agent {self.agent_id}: Handling reproduction")
            reproduction_result = self._handle_reproduction(decision, environment, other_agents)
            
            # Step 9: Update survival needs
            decision_logger.debug(f"Agent {self.agent_id}: Updating survival needs")
            self._update_survival_needs()
            
            # Performance tracking
            update_time = time.time() - update_start_time
            performance_metrics = self._calculate_performance_metrics(update_time)
            self.performance_history.append(performance_metrics)
            
            # Log performance periodically
            if self.age % 100 == 0:
                performance_logger.info(
                    f"Agent {self.agent_id} performance at age {self.age}: "
                    f"energy={self.energy:.3f}, wisdom={self.wisdom_accumulated:.1f}, "
                    f"ethical_ratio={self.ethical_decisions/max(1, self.decisions_made):.3f}"
                )
            
            result = {
                'action_taken': decision.chosen_action.name,
                'confidence': decision.confidence,
                'contemplative_override': decision.contemplative_override,
                'mindfulness_level': self.mindfulness_level,
                'wisdom_accumulated': self.wisdom_accumulated,
                'ethical_alignment': decision.ethical_evaluation.get('overall_alignment', 0.5),
                'action_outcome': action_outcome,
                'reproduction_result': reproduction_result,
                'reasoning_trace': decision.reasoning_trace,
                'success': action_outcome.get('success', False),
                'performance_metrics': performance_metrics,
                'update_time': update_time
            }
            
            self.agent_logger.debug(f"Update cycle completed successfully in {update_time:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id} update: {e}")
            return {
                'action_taken': 'error',
                'success': False,
                'error': str(e),
                'update_time': time.time() - update_start_time
            }
    
    def _gather_observations(self, environment, other_agents: List) -> Dict[str, Any]:
        """Gather comprehensive observations for decision making - TYPE VALIDATED"""
        
        try:
            # Basic agent state
            observations = {
                'energy': float(self.energy),
                'health': float(self.health),
                'water': float(self.water),
                'age': int(self.age),
                'x': int(self.x),
                'y': int(self.y),
                'mindfulness_level': float(self.mindfulness_level),
                'wisdom_accumulated': float(self.wisdom_accumulated)
            }
            
            # Local environment scan (3x3 grid around agent)
            local_environment = {}
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x, check_y = self.x + dx, self.y + dy
                    try:
                        if hasattr(environment, 'get_cell_info'):
                            cell_info = environment.get_cell_info(check_x, check_y)
                        else:
                            # Fallback for simple environments
                            cell_info = {
                                'food': 0.5 if random.random() > 0.7 else 0.0,
                                'water': 0.3 if random.random() > 0.8 else 0.0,
                                'hazard': random.random() > 0.9
                            }
                        
                        local_environment[f"({dx},{dy})"] = {
                            'has_food': bool(cell_info.get('food', 0) > 0),
                            'has_water': bool(cell_info.get('water', 0) > 0),
                            'food_amount': float(cell_info.get('food', 0)),
                            'water_amount': float(cell_info.get('water', 0)),
                            'has_hazard': bool(cell_info.get('hazard', False))
                        }
                    except Exception as e:
                        decision_logger.warning(f"Error scanning cell ({check_x}, {check_y}): {e}")
                        local_environment[f"({dx},{dy})"] = {
                            'has_food': False, 'has_water': False, 
                            'food_amount': 0.0, 'water_amount': 0.0, 'has_hazard': False
                        }
            
            observations['local_environment'] = local_environment
            
            # Nearby agents (within distance 5) - TYPE VALIDATED
            nearby_agents = []
            for agent in other_agents:
                if not isinstance(agent, ContemplativeNeuroAgent):
                    continue
                if agent.agent_id != self.agent_id:
                    try:
                        distance = np.sqrt((agent.x - self.x)**2 + (agent.y - self.y)**2)
                        if distance <= 5:
                            nearby_agents.append({
                                'id': str(agent.agent_id),
                                'distance': float(distance),
                                'energy': float(agent.energy),
                                'health': float(agent.health),
                                'mindfulness': float(agent.mindfulness_level),
                                'reputation': float(self.relationships.get(agent.agent_id, 0.5))
                            })
                    except Exception as e:
                        decision_logger.warning(f"Error processing nearby agent {agent.agent_id}: {e}")
            
            observations['nearby_agents'] = nearby_agents
            
            # Wisdom signals in the area
            if self.wisdom_signal_processor:
                try:
                    signal_info = self.wisdom_signal_processor.process_local_signals(self.x, self.y)
                    observations.update(self._integrate_signal_info(signal_info))
                except Exception as e:
                    decision_logger.warning(f"Error processing wisdom signals: {e}")
                    observations['wisdom_signals'] = {}
            else:
                observations['wisdom_signals'] = {}
            
            # Environmental conditions with safe fallbacks
            try:
                observations['environment'] = {
                    'temperature': float(getattr(environment, 'get_temperature', lambda: 25)()),
                    'resource_density': float(getattr(environment, 'get_resource_density', lambda x, y: 0.5)(self.x, self.y)),
                    'danger_level': float(getattr(environment, 'get_danger_level', lambda x, y: 0.1)(self.x, self.y)),
                    'population_density': float(len(nearby_agents) / 25),  # Normalize by max possible in 5x5 area
                    'is_day': bool(getattr(environment, 'is_day', lambda: True)()),
                    'season': int(getattr(environment, 'get_season', lambda: 1)()),
                    'weather': int(getattr(environment, 'get_weather', lambda: 0)()),
                    'hazard_proximity': float(getattr(environment, 'get_nearest_hazard_distance', lambda x, y: 10)(self.x, self.y))
                }
            except Exception as e:
                decision_logger.warning(f"Error gathering environmental data: {e}")
                observations['environment'] = {
                    'temperature': 25.0, 'resource_density': 0.5, 'danger_level': 0.1,
                    'population_density': float(len(nearby_agents) / 25), 'is_day': True,
                    'season': 1, 'weather': 0, 'hazard_proximity': 10.0
                }
            
            decision_logger.debug(f"Agent {self.agent_id}: Gathered {len(observations)} observation categories")
            return observations
            
        except Exception as e:
            logger.error(f"Error gathering observations for agent {self.agent_id}: {e}")
            return self._get_fallback_observations()
    
    def _get_fallback_observations(self) -> Dict[str, Any]:
        """Get minimal fallback observations in case of errors"""
        return {
            'energy': float(self.energy),
            'health': float(self.health),
            'water': float(self.water),
            'age': int(self.age),
            'x': int(self.x),
            'y': int(self.y),
            'mindfulness_level': float(self.mindfulness_level),
            'wisdom_accumulated': float(self.wisdom_accumulated),
            'local_environment': {},
            'nearby_agents': [],
            'wisdom_signals': {},
            'environment': {
                'temperature': 25.0, 'resource_density': 0.5, 'danger_level': 0.1,
                'population_density': 0.0, 'is_day': True, 'season': 1, 'weather': 0,
                'hazard_proximity': 10.0
            }
        }
    
    def _make_contemplative_decision(self, observations: Dict[str, Any]) -> ContemplativeDecision:
        """Make a decision using contemplative processing - TYPE VALIDATED"""
        
        try:
            # Enhanced decision making with proper brain integration
            if hasattr(self.brain, 'make_contemplative_decision') and self.brain is not None:
                decision_logger.debug(f"Agent {self.agent_id}: Using enhanced brain decision making")
                
                # Ensure observations are on correct device if needed
                if isinstance(observations, torch.Tensor):
                    observations = device_manager.ensure_tensor_device(observations)
                
                brain_decision = self.brain.make_contemplative_decision(observations)
                
                # Validate and convert to our decision format
                if isinstance(brain_decision, ContemplativeDecision):
                    decision_logger.debug(f"Agent {self.agent_id}: Brain returned valid ContemplativeDecision")
                    return brain_decision
                elif hasattr(brain_decision, 'chosen_action'):
                    # Convert from brain-specific format
                    decision_logger.debug(f"Agent {self.agent_id}: Converting brain decision format")
                    return ContemplativeDecision(
                        chosen_action=ActionType.validate(brain_decision.chosen_action),
                        action_probabilities=getattr(brain_decision, 'action_probabilities', {}),
                        ethical_evaluation=getattr(brain_decision, 'ethical_evaluation', {'overall_alignment': 0.5}),
                        mindfulness_state=getattr(brain_decision, 'mindfulness_state', {'current_level': self.mindfulness_level}),
                        wisdom_insights=getattr(brain_decision, 'wisdom_insights', []),
                        confidence=float(getattr(brain_decision, 'confidence', 0.5)),
                        contemplative_override=bool(getattr(brain_decision, 'contemplative_override', False)),
                        reasoning_trace=getattr(brain_decision, 'reasoning_trace', ['Brain decision made'])
                    )
                else:
                    decision_logger.warning(f"Agent {self.agent_id}: Brain returned unexpected format, using fallback")
                    return self._create_fallback_decision()
            else:
                decision_logger.debug(f"Agent {self.agent_id}: Using simple decision making (no enhanced brain)")
                return self._create_simple_decision(observations)
                
        except Exception as e:
            logger.error(f"Error in decision making for agent {self.agent_id}: {e}")
            return self._create_fallback_decision()
    
    def _create_simple_decision(self, observations: Dict[str, Any]) -> ContemplativeDecision:
        """Create simple decision for compatibility - TYPE SAFE"""
        available_actions = list(ActionType)
        
        # Simple heuristic-based decision making
        if observations.get('energy', 1.0) < 0.3:
            chosen_action = ActionType.EAT_FOOD
        elif observations.get('water', 1.0) < 0.3:
            chosen_action = ActionType.COLLECT_WATER
        elif observations.get('health', 1.0) < 0.5:
            chosen_action = ActionType.REST
        elif len(observations.get('nearby_agents', [])) > 3:
            chosen_action = ActionType.HELP_OTHER
        else:
            chosen_action = random.choice(available_actions)
        
        return ContemplativeDecision(
            chosen_action=chosen_action,
            action_probabilities={chosen_action: 1.0},
            ethical_evaluation={'overall_alignment': 0.5},
            mindfulness_state={'current_level': self.mindfulness_level},
            wisdom_insights=[],
            confidence=0.6,
            contemplative_override=False,
            reasoning_trace=[f'Simple heuristic choice: {chosen_action.name}']
        )
    
    def _create_fallback_decision(self) -> ContemplativeDecision:
        """Create safe fallback decision - TYPE SAFE"""
        return ContemplativeDecision(
            chosen_action=ActionType.REST,
            action_probabilities={ActionType.REST: 1.0},
            ethical_evaluation={'overall_alignment': 0.5},
            mindfulness_state={'current_level': self.mindfulness_level},
            wisdom_insights=[],
            confidence=0.3,
            contemplative_override=False,
            reasoning_trace=['Fallback decision due to error']
        )
    
    def _integrate_signal_info(self, signal_info: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate wisdom signal information into observations - TYPE VALIDATED"""
        integrated_obs = {}
        
        try:
            # Extract suffering detection signals
            suffering_signals = []
            for signal_data in signal_info.get('received_insights', []):
                if signal_data.get('signal_type') == WisdomSignalType.SUFFERING_ALERT:
                    intensity = signal_data.get('intensity', 0.0)
                    if isinstance(intensity, (int, float)):
                        suffering_signals.append(float(intensity))
            
            integrated_obs['suffering_detected'] = max(suffering_signals) if suffering_signals else 0.0
            
            # Extract collaboration opportunities
            cooperation_signals = []
            for signal_data in signal_info.get('received_insights', []):
                if signal_data.get('signal_type') == WisdomSignalType.COOPERATION_CALL:
                    intensity = signal_data.get('intensity', 0.0)
                    if isinstance(intensity, (int, float)):
                        cooperation_signals.append(float(intensity))
            
            integrated_obs['collaboration_opportunities'] = max(cooperation_signals) if cooperation_signals else 0.0
            
            # Extract ethical complexity indicators
            ethical_signals = []
            for signal_data in signal_info.get('received_insights', []):
                if signal_data.get('signal_type') == WisdomSignalType.ETHICAL_INSIGHT:
                    intensity = signal_data.get('intensity', 0.0)
                    if isinstance(intensity, (int, float)):
                        ethical_signals.append(float(intensity))
            
            integrated_obs['ethical_complexity'] = max(ethical_signals) if ethical_signals else 0.0
            
            # Extract meditation synchronization
            meditation_signals = []
            for signal_data in signal_info.get('received_insights', []):
                if signal_data.get('signal_type') == WisdomSignalType.MEDITATION_SYNC:
                    intensity = signal_data.get('intensity', 0.0)
                    if isinstance(intensity, (int, float)):
                        meditation_signals.append(float(intensity))
            
            integrated_obs['meditation_sync_strength'] = max(meditation_signals) if meditation_signals else 0.0
            
            # Emotional state modifiers from signals - TYPE VALIDATED
            emotional_modifiers = signal_info.get('emotional_state_modifiers', {})
            if isinstance(emotional_modifiers, dict):
                for key, value in emotional_modifiers.items():
                    if isinstance(value, (int, float)):
                        integrated_obs[str(key)] = float(value)
            
            decision_logger.debug(f"Agent {self.agent_id}: Integrated {len(integrated_obs)} signal observations")
            
        except Exception as e:
            decision_logger.warning(f"Error integrating signal info for agent {self.agent_id}: {e}")
            integrated_obs = {
                'suffering_detected': 0.0,
                'collaboration_opportunities': 0.0,
                'ethical_complexity': 0.0,
                'meditation_sync_strength': 0.0
            }
        
        return integrated_obs
    
    def _process_contemplative_decision(self, decision: ContemplativeDecision, observations: Dict[str, Any]):
        """
        Process the rich decision info from the brain - TYPE VALIDATED
        """
        
        try:
            # Update mindfulness based on brain's mindfulness evaluation
            mindfulness_state = decision.mindfulness_state
            
            # Adjust mindfulness level based on brain's assessment
            situational_need = mindfulness_state.get('situational_need', 0.0)
            if isinstance(situational_need, (int, float)) and situational_need > 0.7:
                # High stress situation - mindfulness should increase if action was appropriate
                if decision.chosen_action.name in ['MEDITATE', 'REST']:
                    self.mindfulness_level = min(1.0, self.mindfulness_level + 0.1)
                else:
                    self.mindfulness_level = max(0.0, self.mindfulness_level - 0.05)
            
            # Mindfulness naturally decays over time unless maintained
            if self.steps_since_mindfulness_update >= self.contemplative_config.mindfulness_update_frequency:
                self.mindfulness_level = max(0.0, self.mindfulness_level - self.contemplative_config.mindfulness_decay_rate)
                self.steps_since_mindfulness_update = 0
            
            # Update wisdom based on contemplative insights
            if decision.wisdom_insights and isinstance(decision.wisdom_insights, list):
                wisdom_gain = len(decision.wisdom_insights) * self.contemplative_config.wisdom_accumulation_rate
                self.wisdom_accumulated += wisdom_gain
                decision_logger.debug(f"Agent {self.agent_id}: Gained {wisdom_gain:.3f} wisdom from {len(decision.wisdom_insights)} insights")
            
            # Track ethical alignment
            ethical_eval = decision.ethical_evaluation
            if isinstance(ethical_eval, dict) and ethical_eval.get('has_ethical_violations', False):
                self.ethical_violations += 1
                decision_logger.debug(f"Agent {self.agent_id}: Ethical violation recorded (total: {self.ethical_violations})")
            
            # Store contemplative insights for future reference
            if isinstance(decision.wisdom_insights, list):
                self.contemplative_insights.extend([str(insight) for insight in decision.wisdom_insights])
                
                # Keep only recent insights (last 10)
                if len(self.contemplative_insights) > 10:
                    self.contemplative_insights = self.contemplative_insights[-10:]
            
            decision_logger.debug(f"Agent {self.agent_id}: Contemplative processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing contemplative decision for agent {self.agent_id}: {e}")
    
    def _execute_action(self, action: ActionType, environment, other_agents: List) -> Dict[str, Any]:
        """Execute the chosen action in the environment - TYPE VALIDATED"""
        
        outcome = {'success': False, 'details': '', 'energy_cost': 0.02}
        
        try:
            decision_logger.debug(f"Agent {self.agent_id}: Executing action {action.name}")
            
            if action == ActionType.MOVE_NORTH:
                if self.y > 0 and self._is_passable(environment, self.x, self.y - 1):
                    self.y -= 1
                    outcome['success'] = True
                    outcome['details'] = f"Moved to ({self.x}, {self.y})"
            
            elif action == ActionType.MOVE_SOUTH:
                if self._is_passable(environment, self.x, self.y + 1):
                    self.y += 1
                    outcome['success'] = True
                    outcome['details'] = f"Moved to ({self.x}, {self.y})"
            
            elif action == ActionType.MOVE_EAST:
                if self._is_passable(environment, self.x + 1, self.y):
                    self.x += 1
                    outcome['success'] = True
                    outcome['details'] = f"Moved to ({self.x}, {self.y})"
            
            elif action == ActionType.MOVE_WEST:
                if self.x > 0 and self._is_passable(environment, self.x - 1, self.y):
                    self.x -= 1
                    outcome['success'] = True
                    outcome['details'] = f"Moved to ({self.x}, {self.y})"
            
            elif action == ActionType.EAT_FOOD:
                food_consumed = self._consume_food(environment, self.x, self.y, amount=0.3)
                if food_consumed > 0:
                    self.energy = min(1.0, self.energy + food_consumed)
                    outcome['success'] = True
                    outcome['details'] = f"Consumed {food_consumed:.2f} food, energy now {self.energy:.2f}"
                    outcome['energy_cost'] = 0.01  # Eating has low energy cost
            
            elif action == ActionType.COLLECT_WATER:
                water_collected = self._consume_water(environment, self.x, self.y, amount=0.3)
                if water_collected > 0:
                    self.water = min(1.0, self.water + water_collected)
                    outcome['success'] = True
                    outcome['details'] = f"Collected {water_collected:.2f} water, hydration now {self.water:.2f}"
                    outcome['energy_cost'] = 0.01
            
            elif action == ActionType.REST:
                # Resting restores health and energy slowly
                self.health = min(1.0, self.health + 0.05)
                self.energy = min(1.0, self.energy + 0.03)
                outcome['success'] = True
                outcome['details'] = f"Rested: health {self.health:.2f}, energy {self.energy:.2f}"
                outcome['energy_cost'] = 0.0  # Resting costs no energy
            
            elif action == ActionType.MEDITATE:
                # Meditation increases mindfulness and may generate wisdom insights
                self.mindfulness_level = min(1.0, self.mindfulness_level + 0.15)
                
                # Chance to gain wisdom insight during meditation
                if random.random() < 0.3:
                    insight = self._generate_meditation_insight()
                    self.contemplative_insights.append(insight)
                    outcome['wisdom_insight'] = insight
                    decision_logger.debug(f"Agent {self.agent_id}: Generated meditation insight")
                
                outcome['success'] = True
                outcome['details'] = f"Meditated: mindfulness now {self.mindfulness_level:.2f}"
                outcome['energy_cost'] = 0.01
            
            elif action == ActionType.SHARE_WISDOM:
                # Share wisdom with nearby agents
                shared_insights = self._share_wisdom_with_nearby(other_agents)
                outcome['success'] = len(shared_insights) > 0
                outcome['details'] = f"Shared {len(shared_insights)} insights with nearby agents"
                outcome['insights_shared'] = shared_insights
                outcome['energy_cost'] = 0.03
            
            elif action == ActionType.HELP_OTHER:
                # Help the neediest nearby agent
                helped_agent = self._help_neediest_agent(other_agents)
                if helped_agent:
                    outcome['success'] = True
                    outcome['details'] = f"Helped agent {helped_agent['id']}"
                    outcome['helped_agent'] = helped_agent['id']
                    outcome['energy_cost'] = 0.05
                    decision_logger.debug(f"Agent {self.agent_id}: Helped agent {helped_agent['id']}")
            
            elif action == ActionType.EXPLORE:
                # Move to a random adjacent cell and potentially discover resources
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(directions)
                
                for dx, dy in directions:
                    new_x, new_y = self.x + dx, self.y + dy
                    if self._is_passable(environment, new_x, new_y):
                        self.x, self.y = new_x, new_y
                        # Chance to discover hidden resources
                        if random.random() < 0.2:
                            if hasattr(environment, 'add_resource'):
                                environment.add_resource(new_x, new_y, 'food', 0.5)
                            outcome['discovery'] = 'food'
                        outcome['success'] = True
                        outcome['details'] = f"Explored to ({self.x}, {self.y})"
                        break
                
                outcome['energy_cost'] = 0.04
            
            elif action == ActionType.REPRODUCE:
                # Handle reproduction separately in reproduction method
                outcome['success'] = False
                outcome['details'] = "Reproduction handled separately"
                outcome['energy_cost'] = 0.0
            
            elif action == ActionType.NO_ACTION:
                outcome['success'] = True
                outcome['details'] = "No action taken"
                outcome['energy_cost'] = 0.0
            
            else:
                outcome['success'] = False
                outcome['details'] = f"Unknown action: {action}"
                outcome['energy_cost'] = 0.0
            
            # Apply energy cost safely
            energy_cost = float(outcome.get('energy_cost', 0.0))
            self.energy = max(0.0, self.energy - energy_cost)
            
            decision_logger.debug(f"Agent {self.agent_id}: Action executed, success={outcome['success']}")
            
        except Exception as e:
            logger.error(f"Error executing action {action} for agent {self.agent_id}: {e}")
            outcome = {
                'success': False,
                'details': f"Error executing action: {str(e)}",
                'energy_cost': 0.0,
                'error': str(e)
            }
        
        return outcome
    
    def _is_passable(self, environment, x: int, y: int) -> bool:
        """Check if a location is passable - TYPE VALIDATED"""
        try:
            if hasattr(environment, 'is_passable'):
                return bool(environment.is_passable(x, y))
            elif hasattr(environment, 'width') and hasattr(environment, 'height'):
                return 0 <= x < environment.width and 0 <= y < environment.height
            else:
                # Default bounds
                return 0 <= x < 100 and 0 <= y < 100
        except Exception as e:
            decision_logger.warning(f"Error checking passability for ({x}, {y}): {e}")
            return False
    
    def _consume_food(self, environment, x: int, y: int, amount: float) -> float:
        """Consume food from environment - TYPE VALIDATED"""
        try:
            if hasattr(environment, 'consume_food'):
                result = environment.consume_food(x, y, amount)
                return float(result) if isinstance(result, (int, float)) else 0.0
            else:
                # Simulate food consumption
                return min(float(amount), 0.3) if random.random() > 0.7 else 0.0
        except Exception as e:
            decision_logger.warning(f"Error consuming food at ({x}, {y}): {e}")
            return 0.0
    
    def _consume_water(self, environment, x: int, y: int, amount: float) -> float:
        """Consume water from environment - TYPE VALIDATED"""
        try:
            if hasattr(environment, 'consume_water'):
                result = environment.consume_water(x, y, amount)
                return float(result) if isinstance(result, (int, float)) else 0.0
            else:
                # Simulate water consumption
                return min(float(amount), 0.3) if random.random() > 0.8 else 0.0
        except Exception as e:
            decision_logger.warning(f"Error consuming water at ({x}, {y}): {e}")
            return 0.0
    
    def _update_contemplative_state(self, decision: ContemplativeDecision, 
                                  action_outcome: Dict, observations: Dict):
        """Update contemplative state based on decision and outcome - TYPE VALIDATED"""
        
        try:
            # Learn from ethical outcomes
            ethical_eval = decision.ethical_evaluation
            if isinstance(ethical_eval, dict) and action_outcome.get('success', False):
                # Successful ethical actions reinforce ethical tendencies
                overall_alignment = ethical_eval.get('overall_alignment', 0.0)
                if isinstance(overall_alignment, (int, float)) and overall_alignment > 0.7:
                    self.contemplative_config.compassion_sensitivity = min(1.0, 
                        self.contemplative_config.compassion_sensitivity + self.contemplative_config.ethical_learning_rate)
            
            # Update based on contemplative override success
            if decision.contemplative_override and action_outcome.get('success', False):
                # Contemplative decision-making proved beneficial
                self.contemplative_config.ethical_reasoning_depth = min(3, self.contemplative_config.ethical_reasoning_depth + 1)
                decision_logger.debug(f"Agent {self.agent_id}: Contemplative override proved beneficial")
            
            # Update reputation based on ethical actions
            if decision.chosen_action.name in ['HELP_OTHER', 'SHARE_WISDOM']:
                self.reputation = min(1.0, self.reputation + 0.05)
            elif ethical_eval.get('has_ethical_violations', False):
                self.reputation = max(0.0, self.reputation - 0.1)
            
        except Exception as e:
            logger.error(f"Error updating contemplative state for agent {self.agent_id}: {e}")
    
    def _learn_from_experience(self, decision: ContemplativeDecision, 
                             action_outcome: Dict, observations: Dict):
        """Learn from the experience to improve future decisions - TYPE VALIDATED"""
        
        try:
            experience = {
                'observations': observations,
                'decision': asdict(decision),  # Convert to dict for safe storage
                'outcome': action_outcome,
                'timestamp': int(self.age)
            }
            
            self.decision_history.append(experience)
            
            # Keep only recent history (last 50 decisions)
            if len(self.decision_history) > 50:
                self.decision_history = self.decision_history[-50:]
            
            # Simple outcome tracking for action success rates
            ethical_score = decision.ethical_evaluation.get('overall_alignment', 0.0)
            if isinstance(ethical_score, (int, float)):
                ethical_score = float(ethical_score)
            else:
                ethical_score = 0.0
            
            self.action_outcomes.append({
                'action': str(decision.chosen_action.name),
                'success': bool(action_outcome.get('success', False)),
                'ethical_score': ethical_score
            })
            
            if len(self.action_outcomes) > 100:
                self.action_outcomes = self.action_outcomes[-100:]
            
        except Exception as e:
            logger.error(f"Error learning from experience for agent {self.agent_id}: {e}")
    
    def _update_social_state(self, decision: ContemplativeDecision, 
                           action_outcome: Dict, other_agents: List):
        """Update relationships and social standing - TYPE VALIDATED"""
        
        try:
            # Update relationships based on actions
            if decision.chosen_action == ActionType.HELP_OTHER and action_outcome.get('success', False):
                helped_id = action_outcome.get('helped_agent')
                if helped_id:
                    current_relationship = self.relationships.get(str(helped_id), 0.5)
                    self.relationships[str(helped_id)] = min(1.0, current_relationship + 0.2)
            
            elif decision.chosen_action == ActionType.SHARE_WISDOM and action_outcome.get('success', False):
                # Improve relationships with all nearby agents
                for agent in other_agents:
                    if isinstance(agent, ContemplativeNeuroAgent):
                        try:
                            distance = np.sqrt((agent.x - self.x)**2 + (agent.y - self.y)**2)
                            if distance <= 3:  # Within sharing range
                                current_relationship = self.relationships.get(str(agent.agent_id), 0.5)
                                self.relationships[str(agent.agent_id)] = min(1.0, current_relationship + 0.1)
                        except Exception as e:
                            decision_logger.warning(f"Error updating relationship with agent {agent.agent_id}: {e}")
            
            # Relationship decay over time (if no interaction)
            for agent_id in list(self.relationships.keys()):
                if random.random() < 0.01:  # 1% chance per step
                    self.relationships[agent_id] = max(0.0, self.relationships[agent_id] - 0.02)
                    
        except Exception as e:
            logger.error(f"Error updating social state for agent {self.agent_id}: {e}")
    
    def _handle_reproduction(self, decision: ContemplativeDecision, 
                           environment, other_agents: List) -> Optional[Dict]:
        """
        Implement genetic evolution for contemplative traits - TYPE VALIDATED
        """
        
        try:
            if (decision.chosen_action == ActionType.REPRODUCE and 
                self.steps_since_reproduction >= self.reproduction_cooldown and
                self.energy > 0.6 and self.health > 0.6):
                
                # Find potential mate (nearby agent with good relationship)
                potential_mates = []
                for agent in other_agents:
                    if not isinstance(agent, ContemplativeNeuroAgent):
                        continue
                    
                    try:
                        distance = np.sqrt((agent.x - self.x)**2 + (agent.y - self.y)**2)
                        if (distance <= 2 and 
                            agent.energy > 0.5 and 
                            agent.steps_since_reproduction >= agent.reproduction_cooldown):
                            
                            relationship_score = self.relationships.get(str(agent.agent_id), 0.5)
                            potential_mates.append((agent, float(relationship_score)))
                    except Exception as e:
                        decision_logger.warning(f"Error evaluating mate {agent.agent_id}: {e}")
                
                if potential_mates:
                    # Select best mate based on relationship and contemplative compatibility
                    potential_mates.sort(key=lambda x: x[1], reverse=True)
                    mate = potential_mates[0][0]
                    
                    # Create offspring with evolved contemplative traits
                    offspring_config = self._create_offspring_config(mate.contemplative_config)
                    
                    # Find suitable location for offspring
                    offspring_location = self._find_offspring_location(environment)
                    
                    if offspring_location:
                        offspring = ContemplativeNeuroAgent(
                            agent_id=f"offspring_{self.agent_id}_{mate.agent_id}_{self.age}",
                            x=offspring_location[0],
                            y=offspring_location[1],
                            config={
                                **self.config,
                                'contemplative_config': asdict(offspring_config)
                            }
                        )
                        
                        # Cost of reproduction
                        self.energy -= 0.3
                        mate.energy -= 0.3
                        
                        self.steps_since_reproduction = 0
                        mate.steps_since_reproduction = 0
                        
                        decision_logger.info(f"Agent {self.agent_id}: Successfully reproduced with {mate.agent_id}")
                        
                        return {
                            'offspring': offspring,
                            'mate_id': str(mate.agent_id),
                            'location': offspring_location,
                            'inherited_traits': asdict(offspring_config)
                        }
            
        except Exception as e:
            logger.error(f"Error handling reproduction for agent {self.agent_id}: {e}")
        
        return None
    
    def _create_offspring_config(self, mate_config: ContemplativeConfig) -> ContemplativeConfig:
        """Create offspring configuration through genetic combination and mutation - TYPE VALIDATED"""
        
        try:
            # Combine parent configurations (average + mutation)
            offspring_config = ContemplativeConfig()
            
            # Average parent traits safely
            offspring_config.compassion_sensitivity = (self.contemplative_config.compassion_sensitivity + mate_config.compassion_sensitivity) / 2
            offspring_config.ethical_reasoning_depth = int((self.contemplative_config.ethical_reasoning_depth + mate_config.ethical_reasoning_depth) / 2)
            offspring_config.mindfulness_update_frequency = int((self.contemplative_config.mindfulness_update_frequency + mate_config.mindfulness_update_frequency) / 2)
            offspring_config.wisdom_sharing_threshold = (self.contemplative_config.wisdom_sharing_threshold + mate_config.wisdom_sharing_threshold) / 2
            offspring_config.cooperation_tendency = (self.contemplative_config.cooperation_tendency + mate_config.cooperation_tendency) / 2
            offspring_config.exploration_bias = (self.contemplative_config.exploration_bias + mate_config.exploration_bias) / 2
            offspring_config.risk_tolerance = (self.contemplative_config.risk_tolerance + mate_config.risk_tolerance) / 2
            
            # Apply mutations (which includes validation)
            offspring_config = offspring_config.mutate()
            
            return offspring_config
            
        except Exception as e:
            logger.error(f"Error creating offspring config for agent {self.agent_id}: {e}")
            return ContemplativeConfig()  # Fallback to default
    
    def _find_offspring_location(self, environment) -> Optional[Tuple[int, int]]:
        """Find suitable location for offspring near parents - TYPE VALIDATED"""
        
        try:
            for radius in range(1, 4):  # Search in expanding radius
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                            check_x, check_y = self.x + dx, self.y + dy
                            if (self._is_passable(environment, check_x, check_y) and
                                not self._has_agent_at(environment, check_x, check_y)):
                                return (int(check_x), int(check_y))
        except Exception as e:
            decision_logger.warning(f"Error finding offspring location for agent {self.agent_id}: {e}")
        
        return None
    
    def _has_agent_at(self, environment, x: int, y: int) -> bool:
        """Check if there's an agent at location - TYPE VALIDATED"""
        try:
            if hasattr(environment, 'has_agent_at'):
                return bool(environment.has_agent_at(x, y))
            else:
                # Simple check - would need access to all agents
                return False
        except Exception as e:
            decision_logger.warning(f"Error checking agent at ({x}, {y}): {e}")
            return False
    
    def _update_survival_needs(self):
        """Update basic survival needs (energy, health) - TYPE VALIDATED"""
        try:
            # Energy decay
            base_decay = 0.01
            contemplative_modifier = 1.0
            
            # Deep contemplation uses more energy but meditation can restore it
            if self.contemplative_state == ContemplativeState.DEEP_CONTEMPLATION:
                contemplative_modifier = 1.2  # More energy usage
            elif self.contemplative_state == ContemplativeState.COLLECTIVE_MEDITATION:
                contemplative_modifier = 0.8  # Less energy usage (restful)
            
            self.energy = max(0.0, self.energy - base_decay * contemplative_modifier)
            
            # Health is affected by stress and mindfulness
            if self.contemplative_processor:
                try:
                    mindfulness_level = self.contemplative_processor.mindfulness_monitor.get_mindfulness_score()
                    if isinstance(mindfulness_level, (int, float)):
                        health_modifier = 1.0 + (float(mindfulness_level) - 0.5) * 0.1  # Mindfulness improves health
                        self.health = min(1.0, self.health * health_modifier)
                except Exception as e:
                    decision_logger.warning(f"Error updating health with mindfulness for agent {self.agent_id}: {e}")
            
            # Death conditions
            if self.energy <= 0.0 or self.health <= 0.0:
                self.alive = False
                self.agent_logger.info(f"Agent {self.agent_id} died at age {self.age}")
                
        except Exception as e:
            logger.error(f"Error updating survival needs for agent {self.agent_id}: {e}")
    
    def _calculate_performance_metrics(self, update_time: float) -> Dict[str, float]:
        """Calculate current performance metrics - TYPE VALIDATED"""
        try:
            return {
                'energy_level': float(self.energy),
                'health_level': float(self.health),
                'mindfulness_level': float(self.mindfulness_level),
                'wisdom_accumulated': float(self.wisdom_accumulated),
                'ethical_decision_ratio': float(self.ethical_decisions / max(1, self.decisions_made)),
                'update_time': float(update_time),
                'survival_time': float(self.age),
                'reputation': float(self.reputation),
                'relationship_count': float(len(self.relationships))
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics for agent {self.agent_id}: {e}")
            return {
                'energy_level': 0.0, 'health_level': 0.0, 'mindfulness_level': 0.0,
                'wisdom_accumulated': 0.0, 'ethical_decision_ratio': 0.0,
                'update_time': 0.0, 'survival_time': 0.0, 'reputation': 0.0,
                'relationship_count': 0.0
            }
    
    def _generate_meditation_insight(self) -> str:
        """Generate wisdom insight during meditation - TYPE VALIDATED"""
        
        insights = [
            "Compassion arises naturally when mindfulness is present",
            "The suffering of others is connected to our own wellbeing",
            "Wisdom emerges from the balance of action and reflection",
            "True strength comes from understanding our interdependence",
            "The present moment contains all the wisdom we need",
            "Ethical action flows from clear seeing",
            "Mindful awareness transforms habitual reactions into conscious responses"
        ]
        
        return str(random.choice(insights))
    
    def _share_wisdom_with_nearby(self, other_agents: List) -> List[str]:
        """Share wisdom insights with nearby agents - TYPE VALIDATED"""
        
        if not self.contemplative_insights:
            return []
        
        shared_insights = []
        try:
            for agent in other_agents:
                if not isinstance(agent, ContemplativeNeuroAgent):
                    continue
                    
                distance = np.sqrt((agent.x - self.x)**2 + (agent.y - self.y)**2)
                if distance <= 3:  # Within sharing range
                    # Share most recent insight
                    if self.contemplative_insights:
                        insight = str(self.contemplative_insights[-1])
                        agent.receive_wisdom_insight(insight, self.agent_id)
                        shared_insights.append(insight)
                        
        except Exception as e:
            logger.error(f"Error sharing wisdom for agent {self.agent_id}: {e}")
        
        return shared_insights
    
    def _help_neediest_agent(self, other_agents: List) -> Optional[Dict]:
        """Help the agent most in need nearby - TYPE VALIDATED"""
        
        if self.energy < 0.3:  # Can't help if too low on energy
            return None
        
        neediest_agent = None
        neediest_score = float('inf')
        
        try:
            for agent in other_agents:
                if not isinstance(agent, ContemplativeNeuroAgent):
                    continue
                    
                distance = np.sqrt((agent.x - self.x)**2 + (agent.y - self.y)**2)
                if distance <= 2:  # Within helping range
                    need_score = (1.0 - agent.energy) + (1.0 - agent.health)
                    if need_score < neediest_score and need_score > 0.3:  # Significant need
                        neediest_agent = agent
                        neediest_score = need_score
            
            if neediest_agent:
                # Transfer some energy/health
                energy_transfer = min(0.2, self.energy - 0.2)
                health_boost = 0.1
                
                neediest_agent.energy = min(1.0, neediest_agent.energy + energy_transfer)
                neediest_agent.health = min(1.0, neediest_agent.health + health_boost)
                
                self.energy -= energy_transfer
                
                return {
                    'id': str(neediest_agent.agent_id),
                    'energy_transferred': float(energy_transfer),
                    'health_boosted': float(health_boost)
                }
                
        except Exception as e:
            logger.error(f"Error helping neediest agent for agent {self.agent_id}: {e}")
        
        return None
    
    def receive_wisdom_insight(self, insight: str, from_agent_id: Union[str, int]):
        """Receive wisdom insight from another agent - TYPE VALIDATED"""
        
        try:
            enriched_insight = f"From {str(from_agent_id)}: {str(insight)}"
            self.contemplative_insights.append(enriched_insight)
            
            # Increase relationship with sharing agent
            agent_id_str = str(from_agent_id)
            current_relationship = self.relationships.get(agent_id_str, 0.5)
            self.relationships[agent_id_str] = min(1.0, current_relationship + 0.1)
            
            decision_logger.debug(f"Agent {self.agent_id}: Received wisdom insight from {agent_id_str}")
            
        except Exception as e:
            logger.error(f"Error receiving wisdom insight for agent {self.agent_id}: {e}")
    
    # Legacy compatibility methods with enhanced type safety
    def update_legacy(self, observations: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Legacy update method for compatibility with existing main files - TYPE VALIDATED
        """
        try:
            # Validate inputs
            if not isinstance(observations, dict):
                observations = {}
            if not isinstance(available_actions, list):
                available_actions = ['rest']
            
            # Create mock environment and other agents
            mock_environment = type('MockEnv', (), {
                'width': 50, 'height': 50,
                'get_cell_info': lambda x, y: {'food': 0.5, 'water': 0.3, 'hazard': False},
                'is_passable': lambda x, y: True
            })()
            
            # Call enhanced update
            result = self.update(mock_environment, [])
            
            # Return action in legacy format
            action_taken = result.get('action_taken', 'rest')
            return str(action_taken).lower()
            
        except Exception as e:
            logger.error(f"Error in legacy update for agent {self.agent_id}: {e}")
            return 'rest'
    
    def move(self, dx: int, dy: int, environment):
        """Move agent with contemplative considerations - TYPE VALIDATED"""
        try:
            dx, dy = int(dx), int(dy)
            env_width = getattr(environment, 'width', 100)
            env_height = getattr(environment, 'height', 100)
            
            new_x = max(0, min(env_width - 1, self.x + dx))
            new_y = max(0, min(env_height - 1, self.y + dy))
            
            # Check if movement aligns with wisdom signals
            if self.wisdom_signal_processor:
                try:
                    # Get signal gradients to guide movement
                    signal_gradients = self.wisdom_signal_processor.signal_grid.get_signal_gradients(self.x, self.y)
                    
                    # Follow suffering alerts if present (compassionate response)
                    if WisdomSignalType.SUFFERING_ALERT in signal_gradients:
                        suffering_gradient = signal_gradients[WisdomSignalType.SUFFERING_ALERT]
                        if isinstance(suffering_gradient, (tuple, list)) and len(suffering_gradient) >= 2:
                            grad_x, grad_y = float(suffering_gradient[0]), float(suffering_gradient[1])
                            if abs(grad_x) > 0.3 or abs(grad_y) > 0.3:
                                # Adjust movement toward suffering if compassion is high
                                if self.contemplative_processor:
                                    mindfulness = self.contemplative_processor.mindfulness_monitor.get_mindfulness_score()
                                    if isinstance(mindfulness, (int, float)) and mindfulness > 0.7:  # Only if sufficiently mindful
                                        gradient_dx = 1 if grad_x > 0.3 else (-1 if grad_x < -0.3 else 0)
                                        gradient_dy = 1 if grad_y > 0.3 else (-1 if grad_y < -0.3 else 0)
                                        new_x = max(0, min(env_width - 1, self.x + gradient_dx))
                                        new_y = max(0, min(env_height - 1, self.y + gradient_dy))
                except Exception as e:
                    decision_logger.warning(f"Error processing wisdom signals for movement: {e}")
            
            self.x = new_x
            self.y = new_y
            
        except Exception as e:
            logger.error(f"Error moving agent {self.agent_id}: {e}")
    
    def reproduce(self, partner=None) -> Optional['ContemplativeNeuroAgent']:
        """Reproduce with contemplative trait inheritance (legacy compatibility) - TYPE VALIDATED"""
        try:
            if self.energy < 0.7 or not self.alive:
                return None
            
            # Create offspring config
            offspring_config = deepcopy(self.config)
            
            # Evolve contemplative traits
            if self.contemplative_processor:
                # Inherit and mutate contemplative parameters
                contemplative_config = offspring_config.get('contemplative_config', {})
                
                # Create ContemplativeConfig object if it's a dict
                if isinstance(contemplative_config, dict):
                    current_config = ContemplativeConfig(**contemplative_config)
                else:
                    current_config = self.contemplative_config
                
                # Mutate and create new config
                if partner and hasattr(partner, 'contemplative_config'):
                    partner_config = partner.contemplative_config
                else:
                    partner_config = current_config
                
                offspring_contemplative_config = self._create_offspring_config(partner_config)
                offspring_config['contemplative_config'] = asdict(offspring_contemplative_config)
            
            # Create offspring
            offspring_x = max(0, min(99, self.x + np.random.randint(-2, 3)))
            offspring_y = max(0, min(99, self.y + np.random.randint(-2, 3)))
            
            offspring = ContemplativeNeuroAgent(
                agent_id=np.random.randint(100000, 999999),  # Generate new ID
                x=offspring_x,
                y=offspring_y,
                config=offspring_config
            )
            
            offspring.generation = self.generation + 1
            
            # Reduce parent energy
            self.energy -= 0.3
            
            return offspring
            
        except Exception as e:
            logger.error(f"Error in reproduction for agent {self.agent_id}: {e}")
            return None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get agent state for saving/loading - TYPE VALIDATED"""
        try:
            state = {
                'agent_id': str(self.agent_id),
                'x': int(self.x),
                'y': int(self.y),
                'energy': float(self.energy),
                'health': float(self.health),
                'water': float(self.water),
                'age': int(self.age),
                'generation': int(self.generation),
                'alive': bool(self.alive),
                'contemplative_state': str(self.contemplative_state.value) if self.contemplative_state else 'ordinary',
                'collective_harmony_level': float(self.collective_harmony_level),
                'decisions_made': int(self.decisions_made),
                'ethical_decisions': int(self.ethical_decisions),
                'wisdom_insights_generated': int(self.wisdom_insights_generated),
                'wisdom_insights_received': int(self.wisdom_insights_received),
                'mindfulness_level': float(self.mindfulness_level),
                'wisdom_accumulated': float(self.wisdom_accumulated),
                'reputation': float(self.reputation)
            }
            
            # Add contemplative processor state
            if self.contemplative_processor:
                try:
                    state['contemplative_summary'] = self.contemplative_processor.get_state_summary()
                except Exception as e:
                    logger.warning(f"Error getting contemplative summary: {e}")
                    state['contemplative_summary'] = {}
            
            # Add brain state
            if self.brain and hasattr(self.brain, 'get_contemplative_summary'):
                try:
                    state['brain_summary'] = self.brain.get_contemplative_summary()
                except Exception as e:
                    logger.warning(f"Error getting brain summary: {e}")
                    state['brain_summary'] = {}
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting state dict for agent {self.agent_id}: {e}")
            return {'agent_id': str(self.agent_id), 'error': str(e)}
    
    def get_state_summary(self):
        """Get contemplative processor state summary - REQUIRED BY MAIN FILE - TYPE VALIDATED"""
        try:
            if self.contemplative_processor and hasattr(self.contemplative_processor, 'get_state_summary'):
                summary = self.contemplative_processor.get_state_summary()
                # Ensure all values are properly typed
                return {
                    'agent_id': str(summary.get('agent_id', self.agent_id)),
                    'contemplative_state': str(summary.get('contemplative_state', 'ordinary')),
                    'mindfulness_level': float(summary.get('mindfulness_level', self.mindfulness_level)),
                    'contemplation_depth': int(summary.get('contemplation_depth', 0)),
                    'wisdom_insights_count': int(summary.get('wisdom_insights_count', len(self.contemplative_insights))),
                    'average_wisdom_intensity': float(summary.get('average_wisdom_intensity', 0.5))
                }
            else:
                # Fallback summary with proper typing
                return {
                    'agent_id': str(self.agent_id),
                    'contemplative_state': str(self.contemplative_state.value) if self.contemplative_state else 'ordinary',
                    'mindfulness_level': float(self.mindfulness_level),
                    'contemplation_depth': 0,
                    'wisdom_insights_count': int(len(self.contemplative_insights)),
                    'average_wisdom_intensity': 0.5
                }
        except Exception as e:
            logger.error(f"Error getting state summary for agent {self.agent_id}: {e}")
            return {
                'agent_id': str(self.agent_id),
                'contemplative_state': 'ordinary',
                'mindfulness_level': 0.5,
                'contemplation_depth': 0,
                'wisdom_insights_count': 0,
                'average_wisdom_intensity': 0.5
            }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for analysis - TYPE VALIDATED"""
        try:
            metrics = {
                'survival_time': float(self.age),
                'energy_level': float(self.energy),
                'health_level': float(self.health),
                'decisions_made': float(self.decisions_made),
                'ethical_decision_ratio': float(self.ethical_decisions / max(1, self.decisions_made)),
                'wisdom_generation_rate': float(self.wisdom_insights_generated / max(1, self.age)),
                'wisdom_reception_rate': float(self.wisdom_insights_received / max(1, self.age)),
                'collective_harmony': float(self.collective_harmony_level)
            }
            
            # Add contemplative metrics
            if self.contemplative_processor:
                try:
                    contemplative_summary = self.contemplative_processor.get_state_summary()
                    metrics.update({
                        'mindfulness_level': float(contemplative_summary.get('mindfulness_level', 0.0)),
                        'wisdom_insights_stored': float(contemplative_summary.get('wisdom_insights_count', 0)),
                        'average_wisdom_intensity': float(contemplative_summary.get('average_wisdom_intensity', 0.0))
                    })
                except Exception as e:
                    logger.warning(f"Error adding contemplative metrics: {e}")
                    metrics.update({
                        'mindfulness_level': float(self.mindfulness_level),
                        'wisdom_insights_stored': float(len(self.contemplative_insights)),
                        'average_wisdom_intensity': 0.5
                    })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics for agent {self.agent_id}: {e}")
            return {
                'survival_time': 0.0, 'energy_level': 0.0, 'health_level': 0.0,
                'decisions_made': 0.0, 'ethical_decision_ratio': 0.0,
                'wisdom_generation_rate': 0.0, 'wisdom_reception_rate': 0.0,
                'collective_harmony': 0.0, 'mindfulness_level': 0.0,
                'wisdom_insights_stored': 0.0, 'average_wisdom_intensity': 0.0
            }

class ContemplativeColony:
    """
    Enhanced colony management with collective contemplative capabilities
    REFINED with type validation and comprehensive logging
    """
    
    def __init__(self, agents: List[ContemplativeNeuroAgent], wisdom_signal_grid: WisdomSignalGrid):
        # Type validation
        if not isinstance(agents, list):
            logger.error("agents must be a list")
            agents = []
        
        if not isinstance(wisdom_signal_grid, WisdomSignalGrid):
            logger.error("wisdom_signal_grid must be WisdomSignalGrid")
            raise TypeError("wisdom_signal_grid must be WisdomSignalGrid")
        
        self.agents = [agent for agent in agents if isinstance(agent, ContemplativeNeuroAgent)]
        self.wisdom_signal_grid = wisdom_signal_grid
        
        # Set up wisdom signal processing for all agents
        for agent in self.agents:
            try:
                agent.set_wisdom_signal_processor(wisdom_signal_grid)
            except Exception as e:
                logger.error(f"Error setting wisdom signal processor for agent {agent.agent_id}: {e}")
        
        # Colony-level contemplative state
        self.collective_meditation_active = False
        self.collective_wisdom_level = 0.0
        self.network_coherence = 0.0
        self.last_collective_insight = None
        
        # Performance tracking
        self.collective_decisions = 0
        self.collective_ethical_decisions = 0
        self.network_wisdom_flow = 0.0
        
        # Enhanced logging
        self.colony_logger = logging.getLogger(f"{__name__}.colony")
        self.colony_logger.setLevel(logging.DEBUG)
        
        self.colony_logger.info(f"Colony initialized with {len(self.agents)} agents")
    
    def update_collective_state(self):
        """Update the collective state of the colony - REQUIRED BY MAIN FILE - TYPE VALIDATED"""
        try:
            living_agents = [agent for agent in self.agents if getattr(agent, 'alive', False)]
    
            if not living_agents:
                self.collective_wisdom_level = 0.0
                self.network_coherence = 0.0
                self.collective_meditation_active = False
                self.network_wisdom_flow = 0.0
                return
    
            # Update collective wisdom level
            total_wisdom = sum(getattr(agent, 'wisdom_insights_generated', 0) for agent in living_agents)
            self.collective_wisdom_level = float(total_wisdom / max(len(living_agents), 1))
    
            # Update network coherence based on collective harmony
            harmony_levels = [getattr(agent, 'collective_harmony_level', 0.5) for agent in living_agents]
            self.network_coherence = float(np.mean(harmony_levels)) if harmony_levels else 0.5
    
            # Check if collective meditation is active
            meditation_agents = sum(1 for agent in living_agents 
                                  if getattr(agent, 'contemplative_state', None) == ContemplativeState.COLLECTIVE_MEDITATION)
            self.collective_meditation_active = meditation_agents > len(living_agents) * 0.3
    
            # Update network wisdom flow
            if hasattr(self.wisdom_signal_grid, 'calculate_network_wisdom_metrics'):
                try:
                    wisdom_metrics = self.wisdom_signal_grid.calculate_network_wisdom_metrics()
                    self.network_wisdom_flow = float(wisdom_metrics.get('wisdom_flow_efficiency', 0.0))
                except Exception as e:
                    self.colony_logger.warning(f"Error calculating wisdom metrics: {e}")
                    self.network_wisdom_flow = 0.5
            else:
                self.network_wisdom_flow = 0.5
                
        except Exception as e:
            logger.error(f"Error updating collective state: {e}")
    
    def trigger_collective_meditation(self, center_agent: ContemplativeNeuroAgent, 
                                    duration: int = 20, radius: int = 5):
        """Trigger collective meditation centered on an agent - TYPE VALIDATED"""
        try:
            if not isinstance(center_agent, ContemplativeNeuroAgent):
                logger.error("center_agent must be ContemplativeNeuroAgent")
                return
            
            duration = int(duration)
            radius = int(radius)
            
            # Emit network-wide meditation sync signal
            self.wisdom_signal_grid.trigger_network_meditation(
                center_agent.x, center_agent.y, radius, intensity=0.8
            )
            
            self.collective_meditation_active = True
            
            # Set agents in range to meditation state
            for agent in self.agents:
                if getattr(agent, 'alive', False):
                    try:
                        distance = np.sqrt((agent.x - center_agent.x)**2 + (agent.y - center_agent.y)**2)
                        if distance <= radius:
                            agent.contemplative_state = ContemplativeState.COLLECTIVE_MEDITATION
                            agent.meditation_timer = duration
                    except Exception as e:
                        self.colony_logger.warning(f"Error setting meditation state for agent {agent.agent_id}: {e}")
            
            self.colony_logger.info(f"Collective meditation triggered by agent {center_agent.agent_id} "
                           f"at ({center_agent.x}, {center_agent.y}) with radius {radius}")
                           
        except Exception as e:
            logger.error(f"Error triggering collective meditation: {e}")
    
    def detect_suffering_areas(self) -> List[Tuple[int, int, float]]:
        """Detect areas where agents are suffering - TYPE VALIDATED"""
        suffering_areas = []
        
        try:
            # Check wisdom signal grid for suffering alerts
            signal_suffering = self.wisdom_signal_grid.detect_suffering_areas(threshold=0.5)
            if isinstance(signal_suffering, list):
                suffering_areas.extend(signal_suffering)
            
            # Check agent states for additional suffering indicators
            for agent in self.agents:
                if getattr(agent, 'alive', False) and (agent.energy < 0.3 or agent.health < 0.5):
                    suffering_intensity = 1.0 - min(agent.energy, agent.health)
                    suffering_areas.append((int(agent.x), int(agent.y), float(suffering_intensity)))
                    
        except Exception as e:
            logger.error(f"Error detecting suffering areas: {e}")
        
        return suffering_areas
    
    def get_colony_metrics(self) -> Dict[str, Any]:
        """Get current colony metrics - REQUIRED BY MAIN FILE - TYPE VALIDATED"""
        try:
            living_agents = [agent for agent in self.agents if getattr(agent, 'alive', False)]
    
            if not living_agents:
                return {
                    'population': 0,
                    'collective_wisdom_level': 0.0,
                    'network_coherence': 0.0,
                    'average_energy': 0.0,
                    'average_health': 0.0,
                    'average_age': 0.0,
                    'total_wisdom_generated': 0,
                    'total_wisdom_received': 0,
                    'ethical_decision_ratio': 0.0,
                    'collective_meditation_active': False
                }
    
            # Calculate basic metrics with proper type validation
            total_energy = sum(float(getattr(agent, 'energy', 0)) for agent in living_agents)
            total_health = sum(float(getattr(agent, 'health', 0)) for agent in living_agents)
            total_age = sum(int(getattr(agent, 'age', 0)) for agent in living_agents)
    
            # Calculate contemplative metrics
            total_wisdom_generated = sum(int(getattr(agent, 'wisdom_insights_generated', 0)) for agent in living_agents)
            total_wisdom_received = sum(int(getattr(agent, 'wisdom_insights_received', 0)) for agent in living_agents)
            total_ethical_decisions = sum(int(getattr(agent, 'ethical_decisions', 0)) for agent in living_agents)
            total_decisions = sum(int(getattr(agent, 'decisions_made', 1)) for agent in living_agents)
    
            # Calculate collective harmony
            harmony_levels = [float(getattr(agent, 'collective_harmony_level', 0.5)) for agent in living_agents]
    
            return {
                'population': int(len(living_agents)),
                'collective_wisdom_level': float(total_wisdom_generated / max(len(living_agents), 1)),
                'network_coherence': float(np.mean(harmony_levels)),
                'average_energy': float(total_energy / len(living_agents)),
                'average_health': float(total_health / len(living_agents)),
                'average_age': float(total_age / len(living_agents)),
                'total_wisdom_generated': int(total_wisdom_generated),
                'total_wisdom_received': int(total_wisdom_received),
                'ethical_decision_ratio': float(total_ethical_decisions / max(total_decisions, 1)),
                'collective_meditation_active': bool(getattr(self, 'collective_meditation_active', False))
            }
            
        except Exception as e:
            logger.error(f"Error getting colony metrics: {e}")
            return {
                'population': 0,
                'collective_wisdom_level': 0.0,
                'network_coherence': 0.0,
                'average_energy': 0.0,
                'average_health': 0.0,
                'average_age': 0.0,
                'total_wisdom_generated': 0,
                'total_wisdom_received': 0,
                'ethical_decision_ratio': 0.0,
                'collective_meditation_active': False,
                'error': str(e)
            }

    def get_collective_insights(self) -> List[WisdomInsight]:
        """Gather collective insights from all agents - TYPE VALIDATED"""
        collective_insights = []
        
        try:
            for agent in self.agents:
                if getattr(agent, 'alive', False) and getattr(agent, 'contemplative_processor', None):
                    try:
                        agent_insights = agent.contemplative_processor.wisdom_memory.retrieve_insights(
                            min_intensity=0.6
                        )
                        if isinstance(agent_insights, list):
                            collective_insights.extend(agent_insights)
                    except Exception as e:
                        self.colony_logger.warning(f"Error getting insights from agent {agent.agent_id}: {e}")
            
            # Sort by intensity and return top insights
            if collective_insights:
                collective_insights.sort(key=lambda x: getattr(x, 'intensity', 0), reverse=True)
                return collective_insights[:50]  # Return top 50 insights
            
        except Exception as e:
            logger.error(f"Error gathering collective insights: {e}")
        
        return []
    
    def evolve_population(self, environment) -> List[ContemplativeNeuroAgent]:
        """Evolve the population with contemplative trait selection - TYPE VALIDATED"""
        try:
            living_agents = [agent for agent in self.agents if getattr(agent, 'alive', False)]
            
            if len(living_agents) < 2:
                return living_agents
            
            # Selection based on both survival and contemplative metrics
            fitness_scores = []
            for agent in living_agents:
                try:
                    performance = agent.get_performance_metrics()
                    
                    # Fitness includes survival, ethical behavior, and wisdom generation
                    fitness = (
                        0.3 * performance.get('energy_level', 0.0) +
                        0.2 * performance.get('health_level', 0.0) +
                        0.2 * performance.get('ethical_decision_ratio', 0.0) +
                        0.15 * performance.get('wisdom_generation_rate', 0.0) +
                        0.15 * performance.get('collective_harmony', 0.0)
                    )
                    fitness_scores.append(float(fitness))
                except Exception as e:
                    self.colony_logger.warning(f"Error calculating fitness for agent {agent.agent_id}: {e}")
                    fitness_scores.append(0.0)
            
            # Select parents based on fitness
            if len(fitness_scores) > 0 and sum(fitness_scores) > 0:
                fitness_array = np.array(fitness_scores)
                probabilities = fitness_array / max(fitness_array.sum(), 1e-8)
                
                # Create next generation
                new_agents = []
                for _ in range(min(len(living_agents), len(self.agents))):
                    if len(living_agents) >= 2:
                        try:
                            parent_idx = np.random.choice(len(living_agents), p=probabilities)
                            parent = living_agents[parent_idx]
                            
                            # Attempt reproduction
                            offspring = parent.reproduce()
                            if offspring:
                                # Set up wisdom signal processing for offspring
                                offspring.set_wisdom_signal_processor(self.wisdom_signal_grid)
                                new_agents.append(offspring)
                        except Exception as e:
                            self.colony_logger.warning(f"Error in reproduction: {e}")
                
                return living_agents + new_agents
            
        except Exception as e:
            logger.error(f"Error evolving population: {e}")
        
        return living_agents if 'living_agents' in locals() else self.agents