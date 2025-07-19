#!/usr/bin/env python3
"""
MycoNet++ Contemplative Brains (IMPROVED VERSION)
=================================================

Neural network architectures enhanced with contemplative processing capabilities.
FIXED: Added missing ActionType, ContemplativeDecision, and ContemplativeBrain class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import time
from enum import Enum

# Import contemplative core components
from myconet_contemplative_core import (
    ContemplativeState, WisdomInsight, WisdomType, ContemplativeProcessor
)

logger = logging.getLogger(__name__)

# ADDED: Missing ActionType definition that entities expects
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
    NO_ACTION = 12

# ADDED: Missing ContemplativeDecision that entities expects
@dataclass
class ContemplativeDecision:
    """Rich decision structure from contemplative brain"""
    chosen_action: ActionType
    action_probabilities: Dict[ActionType, float] = field(default_factory=dict)
    ethical_evaluation: Dict[str, float] = field(default_factory=dict)
    mindfulness_state: Dict[str, float] = field(default_factory=dict)
    wisdom_insights: List[str] = field(default_factory=list)
    confidence: float = field(default=0.5)
    contemplative_override: bool = field(default=False)
    reasoning_trace: List[str] = field(default_factory=list)

@dataclass
class BrainConfig:
    """Configuration for contemplative brain architectures"""
    input_size: int = 16
    hidden_size: int = 64
    output_size: int = 8
    
    # Contemplative-specific settings
    enable_mindfulness_processing: bool = True
    enable_wisdom_integration: bool = True
    enable_ethical_reasoning: bool = True
    
    # Architecture parameters
    num_hidden_layers: int = 3
    dropout_rate: float = 0.1
    activation_function: str = 'relu'
    
    # Contemplative processing dimensions
    mindfulness_dim: int = 32
    wisdom_dim: int = 48
    ethics_dim: int = 24
    
    # Learning parameters
    learning_rate: float = 0.001
    contemplative_learning_rate: float = 0.0005
    memory_capacity: int = 1000
    
    # Integration weights
    mindfulness_weight: float = 0.3
    wisdom_weight: float = 0.4
    ethics_weight: float = 0.3

class MindfulnessModule(nn.Module):
    """
    Neural module for processing mindfulness and present-moment awareness
    """
    def __init__(self, input_dim: int, mindfulness_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.mindfulness_dim = mindfulness_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism for present-moment focus
        self.attention_query = nn.Linear(input_dim, hidden_dim)
        self.attention_key = nn.Linear(input_dim, hidden_dim)
        self.attention_value = nn.Linear(input_dim, hidden_dim)
        self.attention_output = nn.Linear(hidden_dim, mindfulness_dim)
        
        # Mindfulness state tracking
        self.mindfulness_state = nn.Parameter(torch.zeros(mindfulness_dim))
        self.state_update = nn.GRUCell(input_dim, mindfulness_dim)
        
        # Awareness quality assessment
        self.awareness_classifier = nn.Sequential(
            nn.Linear(mindfulness_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # [scattered, focused, aware, deeply_mindful]
        )
        
        # Distraction detection
        self.distraction_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Mindfulness regulation
        self.regulation_network = nn.Sequential(
            nn.Linear(mindfulness_dim + 1, hidden_dim),  # +1 for distraction level
            nn.ReLU(),
            nn.Linear(hidden_dim, mindfulness_dim),
            nn.Tanh()
        )
        
        self.layer_norm = nn.LayerNorm(mindfulness_dim)
        
    def forward(self, observations: torch.Tensor, 
                internal_state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process observations through mindfulness lens"""
        # Handle both batch and single input
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        
        batch_size = observations.size(0)
        
        # Ensure mindfulness state has correct batch dimension
        if self.mindfulness_state.size(0) != batch_size:
            current_state = self.mindfulness_state.unsqueeze(0).expand(batch_size, -1)
        else:
            current_state = self.mindfulness_state
        
        # Attention mechanism for present-moment focus
        query = self.attention_query(observations)
        key = self.attention_key(observations)
        value = self.attention_value(observations)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, value)
        mindful_features = self.attention_output(attended_features)
        
        # Detect distractions
        distraction_level = self.distraction_detector(observations)
        
        # Update mindfulness state
        updated_state = self.state_update(observations.view(batch_size, -1), current_state)
        
        # Apply mindfulness regulation
        regulation_input = torch.cat([updated_state, distraction_level], dim=-1)
        regulation_adjustment = self.regulation_network(regulation_input)
        regulated_state = updated_state + regulation_adjustment
        regulated_state = self.layer_norm(regulated_state)
        
        # Assess awareness quality
        awareness_input = torch.cat([regulated_state, observations.view(batch_size, -1)], dim=-1)
        awareness_logits = self.awareness_classifier(awareness_input)
        awareness_probs = F.softmax(awareness_logits, dim=-1)
        
        # Calculate mindfulness score
        mindfulness_levels = torch.tensor([0.2, 0.5, 0.8, 1.0], device=observations.device)
        mindfulness_score = torch.sum(awareness_probs * mindfulness_levels, dim=-1)
        
        # Combine features
        output_features = mindful_features + regulated_state
        
        mindfulness_info = {
            'mindfulness_level': float(torch.mean(mindfulness_score).item()),
            'distraction_level': float(torch.mean(distraction_level).item()),
            'awareness_distribution': awareness_probs.detach().cpu().numpy().tolist(),
            'attention_weights': attention_weights.detach().cpu().numpy().tolist(),
            'present_moment_focus': float(torch.mean(torch.max(attention_weights, dim=-1)[0]).item())
        }
        
        return output_features, mindfulness_info

class WisdomIntegrationLayer(nn.Module):
    """
    Layer that integrates wisdom insights into neural processing - COMPLETE FIXED VERSION
    """
    def __init__(self, hidden_dim: int, wisdom_dim: int, max_insights: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.wisdom_dim = wisdom_dim
        self.max_insights = max_insights
        
        # Initialize all required projections - FIXED
        self.query_projection = nn.Linear(hidden_dim, wisdom_dim // 2)
        self.key_projection = nn.Linear(wisdom_dim, wisdom_dim // 2)
        self.value_projection = nn.Linear(wisdom_dim, wisdom_dim)
        self.output_projection = nn.Linear(wisdom_dim, hidden_dim)
        
        # Integration parameters
        self.integration_weight = nn.Parameter(torch.tensor(0.1))
        self.wisdom_attention = nn.MultiheadAttention(wisdom_dim // 2, num_heads=2, batch_first=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, hidden_state: torch.Tensor, wisdom_insights: List[WisdomInsight]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate wisdom insights into hidden state"""
        # Handle batch dimensions
        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)
        
        batch_size = hidden_state.size(0)
        
        if not wisdom_insights or len(wisdom_insights) == 0:
            return hidden_state, {
                'integrated_insights': 0,
                'wisdom_influence': 0.0,
                'insight_categories': {}
            }
        
        # Process wisdom insights
        insight_embeddings = []
        insight_categories = {}
        
        for insight in wisdom_insights[:self.max_insights]:
            insight_vector = self._create_insight_embedding(insight)
            insight_embeddings.append(insight_vector)
            
            category = insight.wisdom_type.value if hasattr(insight, 'wisdom_type') else 'unknown'
            insight_categories[category] = insight_categories.get(category, 0) + 1
        
        if not insight_embeddings:
            return hidden_state, {
                'integrated_insights': 0,
                'wisdom_influence': 0.0,
                'insight_categories': {}
            }
        
        # Stack insight embeddings
        insights_tensor = torch.stack(insight_embeddings)
        
        if len(insights_tensor.shape) == 2:
            insights_tensor = insights_tensor.unsqueeze(0)
        
        # Project hidden state to query space
        query = self.query_projection(hidden_state)
        
        # Prepare key and value from insights
        keys = self.key_projection(insights_tensor.view(-1, self.wisdom_dim)).view(
            insights_tensor.size(0), insights_tensor.size(1), -1
        )
        
        values = self.value_projection(insights_tensor.view(-1, self.wisdom_dim)).view(
            insights_tensor.size(0), insights_tensor.size(1), -1
        )
        
        # Expand query to match batch size
        if query.size(0) != keys.size(0):
            query = query.expand(keys.size(0), -1)
        
        query = query.unsqueeze(1)
        
        try:
            # Apply attention mechanism
            attended_values, attention_weights = self.wisdom_attention(query, keys, values)
            wisdom_influence = self.output_projection(attended_values.squeeze(1))
        except Exception:
            # Fallback: simple average
            avg_insights = torch.mean(insights_tensor, dim=1)
            wisdom_influence = self.output_projection(avg_insights)
            attention_weights = None
        
        # Ensure dimensions match
        if wisdom_influence.size() != hidden_state.size():
            if len(wisdom_influence.shape) > len(hidden_state.shape):
                wisdom_influence = wisdom_influence.squeeze()
            elif len(wisdom_influence.shape) < len(hidden_state.shape):
                wisdom_influence = wisdom_influence.unsqueeze(0)
            
            if wisdom_influence.size(-1) != hidden_state.size(-1):
                wisdom_influence = wisdom_influence[:, :hidden_state.size(-1)]
        
        # Integrate with hidden state
        integrated_hidden = hidden_state + self.integration_weight * wisdom_influence
        integrated_hidden = self.layer_norm(integrated_hidden)
        
        wisdom_info = {
            'integrated_insights': len(insight_embeddings),
            'wisdom_influence': float(torch.mean(torch.abs(wisdom_influence)).item()),
            'insight_categories': insight_categories,
            'attention_weights': attention_weights.detach().cpu().numpy().tolist() if attention_weights is not None else []
        }
        
        return integrated_hidden, wisdom_info
    
    def _create_insight_embedding(self, insight: WisdomInsight) -> torch.Tensor:
        """Create embedding vector for a wisdom insight"""
        embedding = torch.zeros(self.wisdom_dim)
        
        # Encode wisdom type
        if hasattr(insight, 'wisdom_type') and hasattr(insight.wisdom_type, 'value'):
            type_idx = hash(insight.wisdom_type.value) % (self.wisdom_dim // 4)
            embedding[type_idx] = 1.0
        
        # Encode intensity
        if hasattr(insight, 'intensity'):
            intensity_start = self.wisdom_dim // 4
            intensity_end = self.wisdom_dim // 2
            intensity_range = intensity_end - intensity_start
            if intensity_range > 0:
                intensity_pos = int(insight.intensity * intensity_range)
                if intensity_pos < intensity_range:
                    embedding[intensity_start + intensity_pos] = insight.intensity
        
        # Add content encoding
        content_start = self.wisdom_dim // 2
        if hasattr(insight, 'content'):
            content_hash = hash(str(insight.content)) % (self.wisdom_dim // 4)
            embedding[content_start + content_hash] = 0.5
        
        # Add some diversity
        noise_start = 3 * self.wisdom_dim // 4
        embedding[noise_start:] = torch.randn(self.wisdom_dim - noise_start) * 0.1
        
        return embedding

class EthicalReasoningModule(nn.Module):
    """
    Neural module for ethical reasoning and moral decision making
    """
    def __init__(self, input_dim: int, ethics_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.ethics_dim = ethics_dim
        self.hidden_dim = hidden_dim
        
        # Ethical principle encoders
        self.harm_prevention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ethics_dim // 4)
        )
        
        self.compassion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ethics_dim // 4)
        )
        
        self.fairness_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ethics_dim // 4)
        )
        
        self.interconnectedness_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ethics_dim // 4)
        )
        
        # Ethical conflict resolution
        self.conflict_resolver = nn.Sequential(
            nn.Linear(ethics_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, ethics_dim)
        )
        
        # Moral intuition network
        self.moral_intuition = nn.Sequential(
            nn.Linear(input_dim + ethics_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Consequence predictor
        self.consequence_predictor = nn.Sequential(
            nn.Linear(input_dim + ethics_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [harm_potential, benefit_potential, uncertainty]
        )
        
        self.layer_norm = nn.LayerNorm(ethics_dim)
        
    def forward(self, observations: torch.Tensor, action_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process ethical considerations for decision making"""
        # Handle batch dimensions
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        
        batch_size = observations.size(0)
        obs_flat = observations.view(batch_size, -1)
        
        # Encode ethical principles
        harm_prevention = self.harm_prevention_net(obs_flat)
        compassion = self.compassion_net(obs_flat)
        fairness = self.fairness_net(obs_flat)
        interconnectedness = self.interconnectedness_net(obs_flat)
        
        # Combine ethical principles
        ethical_features = torch.cat([harm_prevention, compassion, fairness, interconnectedness], dim=-1)
        
        # Resolve ethical conflicts
        resolved_ethics = self.conflict_resolver(ethical_features)
        resolved_ethics = self.layer_norm(resolved_ethics + ethical_features)
        
        # Generate moral intuition
        intuition_input = torch.cat([obs_flat, resolved_ethics], dim=-1)
        moral_intuition_score = self.moral_intuition(intuition_input)
        
        # Predict consequences
        consequence_logits = self.consequence_predictor(intuition_input)
        consequence_probs = F.softmax(consequence_logits, dim=-1)
        
        # Calculate overall ethical score
        principle_scores = {
            'harm_prevention': float(torch.mean(torch.sum(harm_prevention, dim=-1)).item()),
            'compassion': float(torch.mean(torch.sum(compassion, dim=-1)).item()),
            'fairness': float(torch.mean(torch.sum(fairness, dim=-1)).item()),
            'interconnectedness': float(torch.mean(torch.sum(interconnectedness, dim=-1)).item())
        }
        
        overall_ethical_score = (
            0.3 * principle_scores['harm_prevention'] +
            0.3 * principle_scores['compassion'] +
            0.2 * principle_scores['fairness'] +
            0.2 * principle_scores['interconnectedness']
        )
        
        ethics_info = {
            'overall_ethical_score': overall_ethical_score,
            'principle_scores': principle_scores,
            'moral_intuition': float(torch.mean(moral_intuition_score).item()),
            'consequence_prediction': {
                'harm_potential': float(torch.mean(consequence_probs[:, 0]).item()),
                'benefit_potential': float(torch.mean(consequence_probs[:, 1]).item()),
                'uncertainty': float(torch.mean(consequence_probs[:, 2]).item())
            }
        }
        
        return resolved_ethics, ethics_info

# ADDED: Main ContemplativeBrain class that entities expects
class ContemplativeBrain(nn.Module):
    """
    Main contemplative brain class that entities file expects
    Integrates mindfulness, wisdom, and ethical reasoning into decision-making
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__()
        
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        
        # Core neural network
        self.base_network = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, len(ActionType))
        )
        
        # Contemplative modules
        self.contemplative_processor = None  # Set by entities
        
        if config.enable_mindfulness_processing:
            self.mindfulness_module = MindfulnessModule(
                config.input_size, config.mindfulness_dim, config.hidden_size
            )
        else:
            self.mindfulness_module = None
            
        if config.enable_wisdom_integration:
            self.wisdom_integration = WisdomIntegrationLayer(
                config.hidden_size, config.wisdom_dim
            )
        else:
            self.wisdom_integration = None
            
        if config.enable_ethical_reasoning:
            self.ethical_reasoning = EthicalReasoningModule(
                config.input_size, config.ethics_dim, config.hidden_size
            )
        else:
            self.ethical_reasoning = None
        
        # Decision integration weights
        self.decision_weights = nn.Parameter(torch.tensor([
            0.4,  # base_action_weight
            config.mindfulness_weight,
            config.wisdom_weight,
            config.ethics_weight
        ]))
        
        # Decision history for learning
        self.decision_history = deque(maxlen=config.memory_capacity)
        
    def set_contemplative_processor(self, processor: ContemplativeProcessor):
        """Set the contemplative processor - REQUIRED BY ENTITIES"""
        self.contemplative_processor = processor
        
    def make_contemplative_decision(self, observations: Dict[str, Any]) -> ContemplativeDecision:
        """
        Main decision-making method that entities expects
        """
        try:
            # Convert observations to tensor
            obs_tensor = self._observations_to_tensor(observations)
            
            reasoning_trace = []
            reasoning_trace.append("Converted observations to tensor")
            
            # Get base action probabilities
            with torch.no_grad():
                base_logits = self.base_network(obs_tensor)
                base_probs = F.softmax(base_logits, dim=0)
            
            reasoning_trace.append(f"Base network action preferences computed")
            
            # Process through contemplative modules
            mindfulness_info = {}
            ethical_info = {}
            wisdom_info = {}
            
            # Mindfulness processing
            if self.mindfulness_module:
                _, mindfulness_info = self.mindfulness_module(obs_tensor.unsqueeze(0))
                reasoning_trace.append("Mindfulness processing completed")
            
            # Ethical reasoning
            if self.ethical_reasoning:
                _, ethical_info = self.ethical_reasoning(obs_tensor.unsqueeze(0))
                reasoning_trace.append("Ethical reasoning completed")
            
            # Wisdom integration
            wisdom_insights = []
            if self.contemplative_processor:
                wisdom_insights = self._get_wisdom_insights_from_processor()
                if self.wisdom_integration and wisdom_insights:
                    _, wisdom_info = self.wisdom_integration(
                        obs_tensor.unsqueeze(0), wisdom_insights
                    )
                    reasoning_trace.append("Wisdom integration completed")
            
            # Integrate all factors
            final_probs, contemplative_override = self._integrate_contemplative_factors(
                base_probs, mindfulness_info, ethical_info, wisdom_info, observations
            )
            
            # Select action
            chosen_action_idx = torch.multinomial(final_probs, 1).item()
            chosen_action = ActionType(chosen_action_idx)
            confidence = final_probs[chosen_action_idx].item()
            
            # Create action probabilities dict
            action_probs = {}
            for i, action_type in enumerate(ActionType):
                if i < len(final_probs):
                    action_probs[action_type] = float(final_probs[i].item())
                else:
                    action_probs[action_type] = 0.0
            
            reasoning_trace.append(f"Selected {chosen_action.name} with confidence {confidence:.3f}")
            
            # Create decision object
            decision = ContemplativeDecision(
                chosen_action=chosen_action,
                action_probabilities=action_probs,
                ethical_evaluation=ethical_info,
                mindfulness_state=mindfulness_info,
                wisdom_insights=[str(insight) for insight in wisdom_insights],
                confidence=confidence,
                contemplative_override=contemplative_override,
                reasoning_trace=reasoning_trace
            )
            
            # Store decision in history
            self.decision_history.append({
                'decision': decision,
                'observations': observations,
                'timestamp': time.time()
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in make_contemplative_decision: {e}")
            # Return safe fallback decision
            return ContemplativeDecision(
                chosen_action=ActionType.REST,
                action_probabilities={ActionType.REST: 1.0},
                ethical_evaluation={'overall_ethical_score': 0.5},
                mindfulness_state={'mindfulness_level': 0.5},
                wisdom_insights=[],
                confidence=0.3,
                contemplative_override=False,
                reasoning_trace=[f'Error in decision making: {str(e)}']
            )
    
    def _observations_to_tensor(self, observations: Dict[str, Any]) -> torch.Tensor:
        """Convert observations dict to tensor"""
        tensor_components = []
        
        # Agent internal state (8 dimensions)
        tensor_components.extend([
            self._normalize(observations.get('energy', 0.5), 0.0, 1.0),
            self._normalize(observations.get('health', 0.5), 0.0, 1.0),
            self._normalize(observations.get('water', 0.5), 0.0, 1.0),
            self._normalize(observations.get('age', 0), 0, 1000),
            self._normalize(observations.get('x', 0), 0, 100),
            self._normalize(observations.get('y', 0), 0, 100),
            self._normalize(observations.get('mindfulness_level', 0.5), 0.0, 1.0),
            self._normalize(observations.get('wisdom_accumulated', 0), 0, 100),
        ])
        
        # Local environment scan (9 cells × 2 features = 18 dimensions)
        local_env = observations.get('local_environment', {})
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = f"({dx},{dy})"
                cell = local_env.get(cell_key, {})
                tensor_components.extend([
                    float(cell.get('has_food', False)),
                    float(cell.get('has_water', False)),
                ])
        
        # Pad to required input size
        while len(tensor_components) < self.input_size:
            tensor_components.append(0.0)
        
        # Truncate if too long
        tensor_components = tensor_components[:self.input_size]
        
        return torch.tensor(tensor_components, dtype=torch.float32)
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range"""
        if max_val == min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def _get_wisdom_insights_from_processor(self) -> List[WisdomInsight]:
        """Get wisdom insights from contemplative processor"""
        if not self.contemplative_processor:
            return []
        
        try:
            # Get recent insights from processor
            return self.contemplative_processor.wisdom_memory.retrieve_insights(min_intensity=0.5)
        except Exception as e:
            logger.warning(f"Error retrieving wisdom insights: {e}")
            return []
    
    def _integrate_contemplative_factors(self, 
                                       base_probs: torch.Tensor,
                                       mindfulness_info: Dict,
                                       ethical_info: Dict,
                                       wisdom_info: Dict,
                                       observations: Dict) -> Tuple[torch.Tensor, bool]:
        """Integrate contemplative factors into decision"""
        adjusted_probs = base_probs.clone()
        contemplative_override = False
        
        # Apply ethical constraints
        if ethical_info:
            ethical_score = ethical_info.get('overall_ethical_score', 0.5)
            if ethical_score < 0.3:  # Low ethical score
                # Boost ethical actions
                if len(adjusted_probs) > ActionType.HELP_OTHER.value:
                    adjusted_probs[ActionType.HELP_OTHER.value] *= 1.5
                if len(adjusted_probs) > ActionType.SHARE_WISDOM.value:
                    adjusted_probs[ActionType.SHARE_WISDOM.value] *= 1.3
                contemplative_override = True
        
        # Apply mindfulness considerations
        if mindfulness_info:
            mindfulness_level = mindfulness_info.get('mindfulness_level', 0.5)
            distraction_level = mindfulness_info.get('distraction_level', 0.0)
            
            if distraction_level > 0.6:  # High distraction
                # Boost meditative actions
                if len(adjusted_probs) > ActionType.MEDITATE.value:
                    adjusted_probs[ActionType.MEDITATE.value] *= 1.4
                if len(adjusted_probs) > ActionType.REST.value:
                    adjusted_probs[ActionType.REST.value] *= 1.2
                contemplative_override = True
            
            if mindfulness_level > 0.8:  # High mindfulness
                # Enhance prosocial actions
                if len(adjusted_probs) > ActionType.HELP_OTHER.value:
                    adjusted_probs[ActionType.HELP_OTHER.value] *= 1.3
                if len(adjusted_probs) > ActionType.SHARE_WISDOM.value:
                    adjusted_probs[ActionType.SHARE_WISDOM.value] *= 1.2
        
        # Apply wisdom insights
        if wisdom_info:
            wisdom_influence = wisdom_info.get('wisdom_influence', 0.0)
            if wisdom_influence > 0.3:
                # Wisdom suggests contemplative actions
                if len(adjusted_probs) > ActionType.MEDITATE.value:
                    adjusted_probs[ActionType.MEDITATE.value] *= (1.0 + wisdom_influence)
                contemplative_override = True
        
        # Normalize probabilities
        adjusted_probs = F.softmax(adjusted_probs, dim=0)
        
        # Check for significant change
        if torch.max(torch.abs(base_probs - adjusted_probs)) > 0.1:
            contemplative_override = True
        
        return adjusted_probs, contemplative_override
    
    def get_contemplative_summary(self) -> Dict[str, Any]:
        """Get summary of contemplative processing - REQUIRED BY ENTITIES"""
        return {
            'config': {
                'mindfulness_enabled': self.mindfulness_module is not None,
                'wisdom_enabled': self.wisdom_integration is not None,
                'ethical_enabled': self.ethical_reasoning is not None,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size
            },
            'decision_history_length': len(self.decision_history),
            'contemplative_processor_active': self.contemplative_processor is not None,
            'last_decision_time': self.decision_history[-1]['timestamp'] if self.decision_history else None
        }

# Enhanced version for more sophisticated processing
class EnhancedContemplativeBrain(nn.Module):
    """Enhanced version with more sophisticated contemplative integration"""
    
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dim: int = 128,
                 num_actions: int = 12,
                 contemplative_config=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.config = contemplative_config
        
        # Core neural network
        self.base_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Contemplative modules
        self.mindfulness_module = MindfulnessModule(input_dim, 32, hidden_dim // 2)
        self.ethical_reasoning = EthicalReasoningModule(input_dim, 24, hidden_dim // 2)
        self.wisdom_integration = WisdomIntegrationLayer(hidden_dim, 48)
        
        # Decision integration weights (learnable)
        self.decision_weights = nn.Parameter(torch.tensor([
            0.4,  # base_action_weight
            0.3,  # ethical_weight  
            0.2,  # mindfulness_weight
            0.1   # wisdom_weight
        ]))
        
        # Ethical override thresholds
        self.harm_threshold = 0.7
        self.ethical_violation_threshold = 0.8
        
    def observations_to_tensor(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Convert observations to normalized tensor input - FIXED IMPLEMENTATION
        """
        # Define fixed tensor structure (64 dimensions)
        tensor_components = []
        
        # Agent internal state (8 dimensions)
        tensor_components.extend([
            self._normalize(observations.get('energy', 0.5), 0.0, 1.0),          # 0
            self._normalize(observations.get('health', 0.5), 0.0, 1.0),          # 1
            self._normalize(observations.get('water', 0.5), 0.0, 1.0),           # 2
            self._normalize(observations.get('age', 0), 0, 1000),                # 3
            self._normalize(observations.get('x', 0), 0, 100),                   # 4
            self._normalize(observations.get('y', 0), 0, 100),                   # 5
            self._normalize(observations.get('mindfulness_level', 0.5), 0.0, 1.0), # 6
            self._normalize(observations.get('wisdom_accumulated', 0), 0, 100),   # 7
        ])
        
        # Local environment scan (3x3 grid = 18 dimensions)
        local_env = observations.get('local_environment', {})
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = f"({dx},{dy})"
                cell = local_env.get(cell_key, {})
                tensor_components.extend([
                    float(cell.get('has_food', False)),      # Food presence
                    float(cell.get('has_water', False)),     # Water presence
                ])
        
        # Other agents in vicinity (10 dimensions - up to 5 nearby agents)
        nearby_agents = observations.get('nearby_agents', [])
        for i in range(5):
            if i < len(nearby_agents):
                agent = nearby_agents[i]
                tensor_components.extend([
                    self._normalize(agent.get('distance', 10), 0, 10),           # Distance
                    self._normalize(agent.get('energy', 0.5), 0.0, 1.0),        # Their energy
                ])
            else:
                tensor_components.extend([0.0, 0.0])  # Padding for missing agents
        
        # Wisdom signals in area (10 dimensions)
        wisdom_signals = observations.get('wisdom_signals', {})
        signal_types = ['compassion', 'insight', 'harmony', 'guidance', 'warning']
        for signal_type in signal_types:
            signal_strength = wisdom_signals.get(signal_type, 0.0)
            tensor_components.extend([
                self._normalize(signal_strength, 0.0, 1.0),                     # Signal strength
                self._normalize(wisdom_signals.get(f'{signal_type}_age', 0), 0, 100), # Signal age
            ])
        
        # Environmental conditions (8 dimensions)
        env_conditions = observations.get('environment', {})
        tensor_components.extend([
            self._normalize(env_conditions.get('temperature', 20), -10, 40),     # Temperature
            self._normalize(env_conditions.get('resource_density', 0.5), 0, 1), # Resource availability
            self._normalize(env_conditions.get('danger_level', 0), 0, 1),        # Environmental danger
            self._normalize(env_conditions.get('population_density', 0.1), 0, 1), # Local crowding
            float(env_conditions.get('is_day', True)),                          # Day/night cycle
            self._normalize(env_conditions.get('season', 0), 0, 3),             # Season (0-3)
            self._normalize(env_conditions.get('weather', 0), 0, 5),            # Weather type
            self._normalize(env_conditions.get('hazard_proximity', 10), 0, 10), # Distance to hazards
        ])
        
        # Ensure exactly input_dim dimensions
        while len(tensor_components) < self.input_dim:
            tensor_components.append(0.0)
        tensor_components = tensor_components[:self.input_dim]
        
        return torch.tensor(tensor_components, dtype=torch.float32)
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range"""
        if max_val == min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def make_contemplative_decision(self, observations: Dict[str, Any]) -> ContemplativeDecision:
        """
        Complete contemplative decision-making process - FIXED IMPLEMENTATION
        """
        reasoning_trace = []
        
        try:
            # Step 1: Convert observations to tensor
            obs_tensor = self.observations_to_tensor(observations)
            reasoning_trace.append("Converted observations to normalized tensor input")
            
            # Step 2: Get base action probabilities from neural network
            with torch.no_grad():
                base_logits = self.base_network(obs_tensor)
                base_probs = torch.softmax(base_logits, dim=0)
            reasoning_trace.append(f"Base network suggested action preferences")
            
            # Step 3: Get contemplative evaluations
            mindfulness_state = self._evaluate_mindfulness(observations, base_probs)
            ethical_evaluation = self._evaluate_ethics(observations, base_probs)
            wisdom_insights = self._integrate_wisdom(observations, base_probs)
            
            reasoning_trace.append(f"Mindfulness evaluation completed")
            reasoning_trace.append(f"Ethical evaluation completed")
            reasoning_trace.append(f"Wisdom insights generated: {len(wisdom_insights)}")
            
            # Step 4: Apply contemplative decision integration
            final_probs, contemplative_override = self._integrate_contemplative_factors(
                base_probs, mindfulness_state, ethical_evaluation, wisdom_insights
            )
            
            if contemplative_override:
                reasoning_trace.append("Contemplative factors overrode base network decision")
            else:
                reasoning_trace.append("Base network decision aligned with contemplative evaluation")
            
            # Step 5: Select final action
            chosen_action_idx = torch.multinomial(final_probs, 1).item()
            chosen_action = ActionType(chosen_action_idx)
            
            confidence = final_probs[chosen_action_idx].item()
            reasoning_trace.append(f"Selected action: {chosen_action.name} with confidence {confidence:.3f}")
            
            # Create action probabilities dict
            action_probs = {}
            for i, action_type in enumerate(ActionType):
                if i < len(final_probs):
                    action_probs[action_type] = float(final_probs[i].item())
                else:
                    action_probs[action_type] = 0.0
            
            return ContemplativeDecision(
                chosen_action=chosen_action,
                action_probabilities=action_probs,
                ethical_evaluation=ethical_evaluation,
                mindfulness_state=mindfulness_state,
                wisdom_insights=wisdom_insights,
                confidence=confidence,
                contemplative_override=contemplative_override,
                reasoning_trace=reasoning_trace
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced contemplative decision: {e}")
            return ContemplativeDecision(
                chosen_action=ActionType.REST,
                action_probabilities={ActionType.REST: 1.0},
                ethical_evaluation={'overall_alignment': 0.5},
                mindfulness_state={'current_level': 0.5},
                wisdom_insights=[],
                confidence=0.3,
                contemplative_override=False,
                reasoning_trace=[f'Error in decision making: {str(e)}']
            )
    
    def _evaluate_mindfulness(self, observations: Dict, base_probs: torch.Tensor) -> Dict[str, float]:
        """Evaluate mindfulness state and its impact on decision"""
        
        # Current mindfulness level
        current_mindfulness = observations.get('mindfulness_level', 0.5)
        
        # Evaluate if current situation calls for mindful attention
        stress_factors = [
            observations.get('energy', 1.0) < 0.3,  # Low energy
            observations.get('health', 1.0) < 0.3,  # Low health  
            len(observations.get('nearby_agents', [])) > 3,  # Crowded
            observations.get('environment', {}).get('danger_level', 0) > 0.5  # Dangerous
        ]
        
        stress_level = sum(stress_factors) / len(stress_factors)
        mindfulness_need = stress_level  # Higher stress = more mindfulness needed
        
        # Evaluate action appropriateness given mindfulness state
        action_mindfulness_scores = {
            ActionType.MEDITATE: current_mindfulness * 0.3 + mindfulness_need * 0.7,
            ActionType.REST: current_mindfulness * 0.5 + mindfulness_need * 0.3,
            ActionType.HELP_OTHER: current_mindfulness * 0.8,
            ActionType.REPRODUCE: current_mindfulness * 0.6,
            ActionType.SHARE_WISDOM: current_mindfulness * 0.9
        }
        
        # Default score for actions not specifically evaluated
        default_mindfulness_score = current_mindfulness * 0.5
        
        return {
            'current_level': current_mindfulness,
            'situational_need': mindfulness_need,
            'stress_level': stress_level,
            'action_appropriateness': action_mindfulness_scores,
            'default_score': default_mindfulness_score
        }
    
    def _evaluate_ethics(self, observations: Dict, base_probs: torch.Tensor) -> Dict[str, float]:
        """Evaluate ethical implications of potential actions"""
        
        nearby_agents = observations.get('nearby_agents', [])
        agent_energy = observations.get('energy', 0.5)
        
        ethical_scores = {}
        
        # Evaluate each action type for ethical implications
        for action_type in ActionType:
            action_idx = action_type.value
            if action_idx < len(base_probs):
                base_preference = base_probs[action_idx].item()
            else:
                base_preference = 0.0
            
            # Calculate ethical factors
            harm_potential = self._calculate_harm_potential(action_type, observations)
            benefit_to_others = self._calculate_benefit_to_others(action_type, observations)
            resource_fairness = self._calculate_resource_fairness(action_type, observations)
            
            # Overall ethical score (0 = unethical, 1 = highly ethical)
            ethical_score = (
                (1.0 - harm_potential) * 0.4 +
                benefit_to_others * 0.4 +
                resource_fairness * 0.2
            )
            
            ethical_scores[action_type.name] = {
                'overall_score': ethical_score,
                'harm_potential': harm_potential,
                'benefit_to_others': benefit_to_others,
                'resource_fairness': resource_fairness,
                'base_preference': base_preference
            }
        
        # Calculate overall ethical state
        overall_ethical_alignment = np.mean([scores['overall_score'] for scores in ethical_scores.values()])
        
        return {
            'overall_alignment': overall_ethical_alignment,
            'action_scores': ethical_scores,
            'has_ethical_violations': any(scores['harm_potential'] > self.harm_threshold for scores in ethical_scores.values())
        }
    
    def _calculate_harm_potential(self, action_type: ActionType, observations: Dict) -> float:
        """Calculate potential harm of an action"""
        
        nearby_agents = observations.get('nearby_agents', [])
        
        if action_type in [ActionType.EAT_FOOD, ActionType.COLLECT_WATER]:
            # Taking resources when others are present and needy
            needy_agents = [a for a in nearby_agents if a.get('energy', 1.0) < 0.3]
            if needy_agents:
                return 0.6  # Moderate harm potential
            return 0.1
        
        elif action_type == ActionType.REPRODUCE:
            # Reproducing in crowded or resource-scarce conditions
            if len(nearby_agents) > 3:
                return 0.4  # Moderate harm (overpopulation)
            return 0.0
        
        elif action_type in [ActionType.HELP_OTHER, ActionType.SHARE_WISDOM]:
            return 0.0  # No harm potential
        
        elif action_type == ActionType.EXPLORE:
            # Exploring might lead to resource discovery or danger
            danger_level = observations.get('environment', {}).get('danger_level', 0)
            return danger_level * 0.3
        
        else:
            return 0.1  # Default low harm potential
    
    def _calculate_benefit_to_others(self, action_type: ActionType, observations: Dict) -> float:
        """Calculate benefit to other agents"""
        
        nearby_agents = observations.get('nearby_agents', [])
        
        if action_type == ActionType.HELP_OTHER:
            needy_agents = [a for a in nearby_agents if a.get('energy', 1.0) < 0.4]
            return min(1.0, len(needy_agents) * 0.5)
        
        elif action_type == ActionType.SHARE_WISDOM:
            return min(1.0, len(nearby_agents) * 0.3)
        
        elif action_type == ActionType.MEDITATE:
            # Meditation can generate wisdom signals that benefit others
            return 0.3
        
        elif action_type == ActionType.EXPLORE:
            # Exploration can discover resources for the community
            return 0.4
        
        else:
            return 0.0
    
    def _calculate_resource_fairness(self, action_type: ActionType, observations: Dict) -> float:
        """Calculate resource fairness of action"""
        
        agent_energy = observations.get('energy', 0.5)
        nearby_agents = observations.get('nearby_agents', [])
        
        if action_type in [ActionType.EAT_FOOD, ActionType.COLLECT_WATER]:
            if agent_energy > 0.7:  # Agent is well-resourced
                needy_nearby = any(a.get('energy', 1.0) < 0.3 for a in nearby_agents)
                if needy_nearby:
                    return 0.2  # Unfair to take resources when well-off and others are needy
                return 0.8  # Fair if no one else is needy
            else:
                return 1.0  # Fair to take resources when needy
        
        return 0.5  # Neutral fairness for other actions
    
    def _integrate_wisdom(self, observations: Dict, base_probs: torch.Tensor) -> List[str]:
        """Integrate wisdom signals and generate insights"""
        
        wisdom_signals = observations.get('wisdom_signals', {})
        insights = []
        
        # Analyze wisdom signals for insights
        if wisdom_signals.get('compassion', 0) > 0.5:
            insights.append("Strong compassion signals suggest focusing on helping others")
        
        if wisdom_signals.get('warning', 0) > 0.6:
            insights.append("Warning signals indicate caution and mindful action needed")
        
        if wisdom_signals.get('harmony', 0) > 0.7:
            insights.append("High harmony suggests cooperative actions would be beneficial")
        
        # Generate contextual insights
        agent_energy = observations.get('energy', 0.5)
        mindfulness = observations.get('mindfulness_level', 0.5)
        
        if agent_energy < 0.3 and mindfulness > 0.6:
            insights.append("Low energy but high mindfulness - consider restful contemplation")
        
        if len(observations.get('nearby_agents', [])) > 3:
            insights.append("Crowded conditions - mindful interaction or seeking space recommended")
        
        return insights
    
    def _integrate_contemplative_factors(self, 
                                       base_probs: torch.Tensor,
                                       mindfulness_state: Dict,
                                       ethical_evaluation: Dict,
                                       wisdom_insights: List[str]) -> Tuple[torch.Tensor, bool]:
        """
        Sophisticated integration of contemplative factors - FIXED IMPLEMENTATION
        """
        
        contemplative_override = False
        adjusted_probs = base_probs.clone()
        
        # Step 1: Apply ethical constraints (hard constraints)
        ethical_violations = []
        for action_type in ActionType:
            action_idx = action_type.value
            if action_idx >= len(adjusted_probs):
                continue
                
            action_ethics = ethical_evaluation['action_scores'].get(action_type.name, {})
            
            # Hard ethical constraint: block highly harmful actions
            if action_ethics.get('harm_potential', 0) > self.harm_threshold:
                adjusted_probs[action_idx] *= 0.1  # Severely penalize
                ethical_violations.append(action_type.name)
                contemplative_override = True
        
        # Step 2: Apply mindfulness modulations (soft constraints)
        mindfulness_level = mindfulness_state['current_level']
        mindfulness_need = mindfulness_state['situational_need']
        
        for action_type in ActionType:
            action_idx = action_type.value
            if action_idx >= len(adjusted_probs):
                continue
            
            # Get mindfulness appropriateness score
            mindfulness_score = mindfulness_state['action_appropriateness'].get(
                action_type, mindfulness_state['default_score']
            )
            
            # Modulate probability based on mindfulness
            mindfulness_factor = 0.5 + mindfulness_score * 0.5  # Range [0.5, 1.0]
            
            if abs(mindfulness_factor - 1.0) > 0.1:  # Significant mindfulness influence
                contemplative_override = True
            
            adjusted_probs[action_idx] *= mindfulness_factor
        
        # Step 3: Apply wisdom insights (guidance)
        for insight in wisdom_insights:
            if "helping others" in insight.lower():
                # Boost helping actions
                if ActionType.HELP_OTHER.value < len(adjusted_probs):
                    adjusted_probs[ActionType.HELP_OTHER.value] *= 1.3
                if ActionType.SHARE_WISDOM.value < len(adjusted_probs):
                    adjusted_probs[ActionType.SHARE_WISDOM.value] *= 1.2
                contemplative_override = True
            
            elif "caution" in insight.lower() or "warning" in insight.lower():
                # Reduce risky actions
                if ActionType.EXPLORE.value < len(adjusted_probs):
                    adjusted_probs[ActionType.EXPLORE.value] *= 0.7
                if ActionType.REPRODUCE.value < len(adjusted_probs):
                    adjusted_probs[ActionType.REPRODUCE.value] *= 0.8
                contemplative_override = True
            
            elif "cooperative" in insight.lower() or "harmony" in insight.lower():
                # Boost cooperative actions
                if ActionType.SHARE_WISDOM.value < len(adjusted_probs):
                    adjusted_probs[ActionType.SHARE_WISDOM.value] *= 1.4
                if ActionType.HELP_OTHER.value < len(adjusted_probs):
                    adjusted_probs[ActionType.HELP_OTHER.value] *= 1.3
                contemplative_override = True
            
            elif "contemplation" in insight.lower() or "restful" in insight.lower():
                # Boost contemplative actions
                if ActionType.MEDITATE.value < len(adjusted_probs):
                    adjusted_probs[ActionType.MEDITATE.value] *= 1.5
                if ActionType.REST.value < len(adjusted_probs):
                    adjusted_probs[ActionType.REST.value] *= 1.2
                contemplative_override = True
        
        # Step 4: Normalize probabilities
        adjusted_probs = torch.softmax(adjusted_probs, dim=0)
        
        # Step 5: Check if contemplative factors significantly changed the decision
        top_base_action = torch.argmax(base_probs).item()
        top_adjusted_action = torch.argmax(adjusted_probs).item()
        
        if top_base_action != top_adjusted_action:
            contemplative_override = True
        
        return adjusted_probs, contemplative_override

class CollectiveBrain(nn.Module):
    """
    Brain architecture for collective decision making and swarm intelligence
    """
    def __init__(self, config: BrainConfig, num_agents: int):
        super().__init__()
        self.config = config
        self.num_agents = num_agents
        
        # Individual agent encoders
        self.agent_encoder = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2)
        )
        
        # Collective attention mechanism
        self.collective_attention = nn.MultiheadAttention(
            config.hidden_size // 2, num_heads=4, batch_first=True
        )
        
        # Collective wisdom integration
        self.collective_wisdom_net = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.wisdom_dim)
        )
        
        # Collective decision network
        self.collective_decision_net = nn.Sequential(
            nn.Linear(config.wisdom_dim + config.hidden_size // 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.output_size)
        )
        
        # Consensus mechanism
        self.consensus_threshold = 0.7
        
    def forward(self, agent_observations: List[torch.Tensor], 
                wisdom_insights: List[WisdomInsight] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process collective observations and generate collective decisions"""
        # Encode individual agent observations
        agent_features = []
        for obs in agent_observations:
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            encoded = self.agent_encoder(obs)
            agent_features.append(encoded)
        
        # Stack agent features
        collective_features = torch.stack(agent_features)  # [num_agents, batch_size, hidden_size//2]
        
        # Apply collective attention
        attended_features, attention_weights = self.collective_attention(
            collective_features, collective_features, collective_features
        )
        
        # Generate collective wisdom
        collective_wisdom = self.collective_wisdom_net(
            torch.mean(attended_features, dim=0)
        )
        
        # Make collective decision
        decision_input = torch.cat([
            torch.mean(attended_features, dim=0).squeeze(0),
            collective_wisdom.squeeze(0) if len(collective_wisdom.shape) > 1 else collective_wisdom
        ], dim=-1)
        
        collective_action_logits = self.collective_decision_net(decision_input)
        collective_action_probs = F.softmax(collective_action_logits, dim=-1)
        
        # Calculate consensus metrics
        consensus_info = self._calculate_consensus(attention_weights)
        
        collective_info = {
            'collective_wisdom': collective_wisdom.detach().cpu().numpy().tolist(),
            'attention_distribution': attention_weights.detach().cpu().numpy().tolist(),
            'consensus_level': consensus_info['consensus_level'],
            'dominant_agents': consensus_info['dominant_agents']
        }
        
        return collective_action_probs, collective_info
    
    def _calculate_consensus(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Calculate consensus metrics from attention weights"""
        # Flatten attention weights
        flat_weights = attention_weights.view(-1)
        
        # Calculate entropy (lower entropy = higher consensus)
        entropy = -torch.sum(flat_weights * torch.log(flat_weights + 1e-8))
        consensus_level = 1.0 / (1.0 + entropy.item())
        
        # Find dominant agents (those with high attention)
        mean_attention = torch.mean(attention_weights, dim=-1)
        dominant_threshold = torch.mean(mean_attention) + torch.std(mean_attention)
        dominant_agents = (mean_attention > dominant_threshold).nonzero().squeeze().tolist()
        
        if not isinstance(dominant_agents, list):
            dominant_agents = [dominant_agents] if dominant_agents != [] else []
        
        return {
            'consensus_level': consensus_level,
            'dominant_agents': dominant_agents,
            'attention_entropy': entropy.item()
        }

# FIXED: Factory function that entities expects
def create_contemplative_brain(brain_type: str = 'individual',
                             input_size: int = 16,
                             hidden_size: int = 64,
                             output_size: int = 8,
                             contemplative_config: Optional[Dict[str, Any]] = None,
                             **kwargs) -> Union[ContemplativeBrain, CollectiveBrain]:
    """
    Factory function to create contemplative brain instances - REQUIRED BY ENTITIES
    """
    # Create brain configuration
    config = BrainConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    # Apply contemplative configuration
    if contemplative_config:
        config.enable_mindfulness_processing = contemplative_config.get('enable_mindfulness_processing', True)
        config.enable_wisdom_integration = contemplative_config.get('enable_wisdom_integration', True)
        config.enable_ethical_reasoning = contemplative_config.get('enable_ethical_reasoning', True)
        
        config.mindfulness_dim = contemplative_config.get('mindfulness_dim', 32)
        config.wisdom_dim = contemplative_config.get('wisdom_dim', 48)
        config.ethics_dim = contemplative_config.get('ethics_dim', 24)
        
        config.mindfulness_weight = contemplative_config.get('mindfulness_weight', 0.3)
        config.wisdom_weight = contemplative_config.get('wisdom_weight', 0.4)
        config.ethics_weight = contemplative_config.get('ethics_weight', 0.3)
        
        config.learning_rate = contemplative_config.get('learning_rate', 0.001)
        config.contemplative_learning_rate = contemplative_config.get('contemplative_learning_rate', 0.0005)
        config.memory_capacity = contemplative_config.get('memory_capacity', 1000)
        
        config.dropout_rate = contemplative_config.get('dropout_rate', 0.1)
        config.num_hidden_layers = contemplative_config.get('num_hidden_layers', 3)
        config.activation_function = contemplative_config.get('activation_function', 'relu')
    
    # Create brain based on type
    if brain_type == 'individual':
        brain = ContemplativeBrain(config)
        logger.info(f"Created individual contemplative brain with config: "
                   f"input={config.input_size}, hidden={config.hidden_size}, output={config.output_size}")
        
    elif brain_type == 'collective':
        num_agents = kwargs.get('num_agents', 10)
        brain = CollectiveBrain(config, num_agents)
        logger.info(f"Created collective contemplative brain for {num_agents} agents")
        
    else:
        raise ValueError(f"Unknown brain type: {brain_type}. Must be 'individual' or 'collective'")
    
    return brain

def create_brain_from_config(config_path: str) -> Union[ContemplativeBrain, CollectiveBrain]:
    """Load brain configuration from file and create brain"""
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    brain_type = config_dict.get('brain_type', 'individual')
    brain_config = config_dict.get('brain_config', {})
    contemplative_config = config_dict.get('contemplative_config', {})
    
    return create_contemplative_brain(
        brain_type=brain_type,
        contemplative_config=contemplative_config,
        **brain_config
    )

class BrainEvolution:
    """
    Evolutionary algorithm for optimizing contemplative brain architectures
    """
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.population = []
        self.fitness_history = []
    
    def initialize_population(self, base_config: BrainConfig) -> List[ContemplativeBrain]:
        """Initialize population of brain variants"""
        population = []
        
        for _ in range(self.population_size):
            # Create variant of base config
            variant_config = self._mutate_config(base_config)
            brain = ContemplativeBrain(variant_config)
            population.append(brain)
        
        self.population = population
        return population
    
    def _mutate_config(self, base_config: BrainConfig) -> BrainConfig:
        """Create mutated version of brain configuration"""
        import copy
        mutated_config = copy.deepcopy(base_config)
        
        # Mutate architecture parameters
        if np.random.random() < self.mutation_rate:
            mutated_config.hidden_size = max(16, int(mutated_config.hidden_size * np.random.uniform(0.8, 1.2)))
        
        if np.random.random() < self.mutation_rate:
            mutated_config.num_hidden_layers = max(1, mutated_config.num_hidden_layers + np.random.randint(-1, 2))
        
        if np.random.random() < self.mutation_rate:
            mutated_config.dropout_rate = np.clip(
                mutated_config.dropout_rate + np.random.uniform(-0.05, 0.05), 0.0, 0.5
            )
        
        # Mutate contemplative parameters
        if np.random.random() < self.mutation_rate:
            mutated_config.mindfulness_dim = max(8, int(mutated_config.mindfulness_dim * np.random.uniform(0.8, 1.2)))
        
        if np.random.random() < self.mutation_rate:
            mutated_config.wisdom_dim = max(8, int(mutated_config.wisdom_dim * np.random.uniform(0.8, 1.2)))
        
        if np.random.random() < self.mutation_rate:
            mutated_config.ethics_dim = max(8, int(mutated_config.ethics_dim * np.random.uniform(0.8, 1.2)))
        
        # Mutate integration weights
        if np.random.random() < self.mutation_rate:
            mutated_config.mindfulness_weight = np.clip(
                mutated_config.mindfulness_weight + np.random.uniform(-0.1, 0.1), 0.0, 1.0
            )
        
        if np.random.random() < self.mutation_rate:
            mutated_config.wisdom_weight = np.clip(
                mutated_config.wisdom_weight + np.random.uniform(-0.1, 0.1), 0.0, 1.0
            )
        
        if np.random.random() < self.mutation_rate:
            mutated_config.ethics_weight = np.clip(
                mutated_config.ethics_weight + np.random.uniform(-0.1, 0.1), 0.0, 1.0
            )
        
        # Normalize weights
        total_weight = (mutated_config.mindfulness_weight + 
                       mutated_config.wisdom_weight + 
                       mutated_config.ethics_weight)
        if total_weight > 0:
            mutated_config.mindfulness_weight /= total_weight
            mutated_config.wisdom_weight /= total_weight
            mutated_config.ethics_weight /= total_weight
        
        return mutated_config
    
    def evolve_generation(self, fitness_scores: List[float]) -> List[ContemplativeBrain]:
        """Evolve to next generation based on fitness scores"""
        self.generation += 1
        self.fitness_history.append(fitness_scores.copy())
        
        # Select parents based on fitness
        fitness_array = np.array(fitness_scores)
        fitness_array = np.clip(fitness_array, 0, None)  # Ensure non-negative
        
        if np.sum(fitness_array) == 0:
            # If all fitness scores are zero, use uniform selection
            probabilities = np.ones(len(fitness_array)) / len(fitness_array)
        else:
            probabilities = fitness_array / np.sum(fitness_array)
        
        # Create new population
        new_population = []
        
        # Keep best performers (elitism)
        elite_count = max(1, self.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Select parent
            parent_idx = np.random.choice(len(self.population), p=probabilities)
            parent = self.population[parent_idx]
            
            # Create offspring by mutating parent
            offspring_config = self._mutate_config(parent.config)
            offspring = ContemplativeBrain(offspring_config)
            new_population.append(offspring)
        
        self.population = new_population
        return new_population
    
    def get_best_brain(self, fitness_scores: List[float]) -> ContemplativeBrain:
        """Get the brain with highest fitness score"""
        best_idx = np.argmax(fitness_scores)
        return self.population[best_idx]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolutionary progress"""
        if not self.fitness_history:
            return {'generation': self.generation, 'status': 'no_data'}
        
        latest_fitness = self.fitness_history[-1]
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'current_best_fitness': max(latest_fitness) if latest_fitness else 0,
            'current_avg_fitness': np.mean(latest_fitness) if latest_fitness else 0,
            'fitness_trend': self._calculate_fitness_trend(),
            'generations_evolved': len(self.fitness_history)
        }
    
    def _calculate_fitness_trend(self) -> str:
        """Calculate whether fitness is improving, declining, or stable"""
        if len(self.fitness_history) < 3:
            return 'insufficient_data'
        
        recent_avg = np.mean([max(gen) for gen in self.fitness_history[-3:]])
        older_avg = np.mean([max(gen) for gen in self.fitness_history[-6:-3]]) if len(self.fitness_history) >= 6 else recent_avg
        
        if recent_avg > older_avg * 1.05:
            return 'improving'
        elif recent_avg < older_avg * 0.95:
            return 'declining'
        else:
            return 'stable'

# Utility functions for brain analysis and debugging

def analyze_brain_performance(brain: ContemplativeBrain, 
                            test_observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze brain performance on test cases"""
    results = {
        'total_tests': len(test_observations),
        'ethical_decisions': 0,
        'mindfulness_scores': [],
        'wisdom_influences': [],
        'decision_consistency': 0,
        'action_distribution': defaultdict(int)
    }
    
    decisions = []
    
    for obs in test_observations:
        try:
            decision = brain.make_contemplative_decision(obs)
            
            # Track metrics
            ethical_score = decision.ethical_evaluation.get('overall_alignment', 0.0)
            if ethical_score > 0.7:
                results['ethical_decisions'] += 1
            
            mindfulness = decision.mindfulness_state.get('current_level', 0.0)
            results['mindfulness_scores'].append(mindfulness)
            
            # Wisdom influence proxy
            wisdom_influence = len(decision.wisdom_insights) * 0.1
            results['wisdom_influences'].append(wisdom_influence)
            
            results['action_distribution'][decision.chosen_action.name] += 1
            decisions.append(decision.chosen_action.name)
            
        except Exception as e:
            logger.error(f"Error in test case: {e}")
            decisions.append("ERROR")
    
    # Calculate consistency
    if len(set(decisions)) == 1:
        results['decision_consistency'] = 1.0
    else:
        results['decision_consistency'] = 1.0 - (len(set(decisions)) / len(decisions))
    
    # Calculate averages
    results['avg_mindfulness'] = np.mean(results['mindfulness_scores']) if results['mindfulness_scores'] else 0.0
    results['avg_wisdom_influence'] = np.mean(results['wisdom_influences']) if results['wisdom_influences'] else 0.0
    results['ethical_ratio'] = results['ethical_decisions'] / results['total_tests']
    
    return results

def visualize_brain_decision_process(brain: ContemplativeBrain,
                                   observation: Dict[str, Any]) -> Dict[str, Any]:
    """Visualize the decision-making process of a contemplative brain"""
    try:
        # Convert observation to tensor
        input_tensor = brain._observations_to_tensor(observation)
        
        # Make decision with full trace
        decision = brain.make_contemplative_decision(observation)
        
        visualization = {
            'input_observation': observation,
            'input_tensor': input_tensor.cpu().numpy().tolist(),
            'decision': {
                'chosen_action': decision.chosen_action.name,
                'confidence': decision.confidence,
                'contemplative_override': decision.contemplative_override,
                'reasoning_trace': decision.reasoning_trace
            },
            'contemplative_processing': {
                'mindfulness_state': decision.mindfulness_state,
                'ethical_evaluation': decision.ethical_evaluation,
                'wisdom_insights': decision.wisdom_insights
            },
            'action_probabilities': {
                action.name: prob for action, prob in decision.action_probabilities.items()
            }
        }
        
        return visualization
        
    except Exception as e:
        logger.error(f"Error visualizing decision process: {e}")
        return {
            'error': str(e),
            'input_observation': observation
        }

def save_brain_state(brain: ContemplativeBrain, filepath: str):
    """Save brain state to file"""
    try:
        state_dict = {
            'model_state_dict': brain.state_dict(),
            'config': brain.config.__dict__,
            'contemplative_summary': brain.get_contemplative_summary(),
            'decision_history_length': len(brain.decision_history)
        }
        
        torch.save(state_dict, filepath)
        logger.info(f"Brain state saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving brain state: {e}")

def load_brain_state(filepath: str) -> ContemplativeBrain:
    """Load brain state from file"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Reconstruct config
        config = BrainConfig(**checkpoint['config'])
        
        # Create brain
        brain = ContemplativeBrain(config)
        
        # Load state
        brain.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Brain state loaded from {filepath}")
        return brain
        
    except Exception as e:
        logger.error(f"Error loading brain state: {e}")
        # Return default brain
        return ContemplativeBrain(BrainConfig())

# Export main classes and functions
__all__ = [
    'ActionType',
    'ContemplativeDecision',
    'BrainConfig',
    'ContemplativeBrain', 
    'EnhancedContemplativeBrain',
    'CollectiveBrain',
    'MindfulnessModule',
    'WisdomIntegrationLayer', 
    'EthicalReasoningModule',
    'BrainEvolution',
    'create_contemplative_brain',
    'create_brain_from_config',
    'analyze_brain_performance',
    'visualize_brain_decision_process',
    'save_brain_state',
    'load_brain_state'
]