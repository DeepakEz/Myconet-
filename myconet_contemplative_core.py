"""
MycoNet++ Contemplative Core Module
===================================

Core contemplative processing components for enhanced MycoNet agents.
Implements mindfulness, wisdom memory, ethical reasoning, and contemplative states.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ContemplativeState(Enum):
    """Enumeration of possible contemplative states"""
    ORDINARY = "ordinary"
    MINDFUL = "mindful"
    DEEP_CONTEMPLATION = "deep_contemplation"
    COLLECTIVE_MEDITATION = "collective_meditation"
    WISDOM_INTEGRATION = "wisdom_integration"

class WisdomType(Enum):
    """Types of wisdom insights"""
    ETHICAL_INSIGHT = "ethical_insight"
    SUFFERING_DETECTION = "suffering_detection"
    COMPASSION_RESPONSE = "compassion_response"
    INTERCONNECTEDNESS = "interconnectedness"
    IMPERMANENCE = "impermanence"
    PRACTICAL_WISDOM = "practical_wisdom"

@dataclass
class WisdomInsight:
    """Container for wisdom insights"""
    wisdom_type: WisdomType
    content: Dict[str, Any]
    intensity: float
    timestamp: float
    source_agent_id: Optional[int] = None
    propagation_count: int = 0
    decay_rate: float = 0.05

class MindfulnessMonitor:
    """
    Monitors and maintains mindfulness state for an agent
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.attention_history = deque(maxlen=capacity)
        self.mindfulness_level = 0.5
        self.awareness_threshold = 0.6
        
    def update_attention(self, attention_focus: np.ndarray, decision_quality: float):
        """Update mindfulness based on attention patterns and decision quality"""
        # Calculate attention coherence (how focused vs scattered)
        attention_coherence = 1.0 - np.var(attention_focus) if len(attention_focus) > 0 else 0.5
        
        # Store attention data
        attention_data = {
            'coherence': attention_coherence,
            'quality': decision_quality,
            'timestamp': time.time()
        }
        self.attention_history.append(attention_data)
        
        # Update mindfulness level
        if len(self.attention_history) >= 10:
            recent_coherence = np.mean([a['coherence'] for a in list(self.attention_history)[-10:]])
            recent_quality = np.mean([a['quality'] for a in list(self.attention_history)[-10:]])
            
            # Mindfulness increases with sustained attention and good decisions
            mindfulness_delta = 0.01 * (recent_coherence + recent_quality - 1.0)
            self.mindfulness_level = np.clip(
                self.mindfulness_level + mindfulness_delta, 0.0, 1.0
            )
    
    def is_mindful(self) -> bool:
        """Check if agent is in mindful state"""
        return self.mindfulness_level > self.awareness_threshold
    
    def get_mindfulness_score(self) -> float:
        """Get current mindfulness level"""
        return self.mindfulness_level

class WisdomMemory:
    """
    Specialized memory system for storing and retrieving wisdom insights
    """
    def __init__(self, capacity: int = 1000, wisdom_threshold: float = 0.7):
        self.capacity = capacity
        self.wisdom_threshold = wisdom_threshold
        self.insights: List[WisdomInsight] = []
        self.wisdom_index = {}  # Index by wisdom type for fast retrieval
        
    def store_insight(self, insight: WisdomInsight):
        """Store a wisdom insight"""
        # Only store insights above threshold
        if insight.intensity >= self.wisdom_threshold:
            self.insights.append(insight)
            
            # Maintain capacity
            if len(self.insights) > self.capacity:
                # Remove oldest, least intense insights
                self.insights.sort(key=lambda x: (x.timestamp, x.intensity))
                self.insights = self.insights[-self.capacity:]
            
            # Update index
            if insight.wisdom_type not in self.wisdom_index:
                self.wisdom_index[insight.wisdom_type] = []
            self.wisdom_index[insight.wisdom_type].append(insight)
    
    def retrieve_insights(self, wisdom_type: Optional[WisdomType] = None, 
                         min_intensity: float = 0.0) -> List[WisdomInsight]:
        """Retrieve wisdom insights by type and intensity"""
        if wisdom_type:
            insights = self.wisdom_index.get(wisdom_type, [])
        else:
            insights = self.insights
        
        return [insight for insight in insights if insight.intensity >= min_intensity]
    
    def decay_insights(self, time_delta: float):
        """Apply temporal decay to insights"""
        for insight in self.insights:
            insight.intensity *= (1.0 - insight.decay_rate * time_delta)
        
        # Remove insights that have decayed below threshold
        self.insights = [i for i in self.insights if i.intensity >= 0.1]
        
        # Rebuild index
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the wisdom type index"""
        self.wisdom_index = {}
        for insight in self.insights:
            if insight.wisdom_type not in self.wisdom_index:
                self.wisdom_index[insight.wisdom_type] = []
            self.wisdom_index[insight.wisdom_type].append(insight)

class EthicalReasoningModule:
    """
    Module for ethical reasoning and moral decision making
    """
    def __init__(self, ethical_frameworks: List[str] = None):
        self.ethical_frameworks = ethical_frameworks or [
            'consequentialist', 'deontological', 'virtue_ethics', 'buddhist_ethics'
        ]
        
        # Ethical principles (can be learned/updated)
        self.principles = {
            'reduce_suffering': 0.9,
            'promote_wellbeing': 0.8,
            'respect_autonomy': 0.7,
            'fairness': 0.8,
            'compassion': 0.9,
            'wisdom': 0.8,
            'non_violence': 0.9,
            'truthfulness': 0.8
        }
        
        self.ethical_memory = deque(maxlen=100)
    
    def evaluate_action_ethics(self, action: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the ethical implications of an action"""
        ethical_scores = {}
        
        # Consequentialist evaluation (outcomes)
        ethical_scores['consequentialist'] = self._evaluate_consequences(action, context)
        
        # Deontological evaluation (duties/rules)
        ethical_scores['deontological'] = self._evaluate_duties(action, context)
        
        # Virtue ethics evaluation (character)
        ethical_scores['virtue_ethics'] = self._evaluate_virtues(action, context)
        
        # Buddhist ethics evaluation (compassion/wisdom)
        ethical_scores['buddhist_ethics'] = self._evaluate_buddhist_ethics(action, context)
        
        # Overall ethical score (weighted average)
        ethical_scores['overall'] = np.mean(list(ethical_scores.values()))
        
        return ethical_scores
    
    def _evaluate_consequences(self, action: Dict[str, Any], 
                             context: Dict[str, Any]) -> float:
        """Evaluate action based on predicted consequences"""
        # Simplified consequentialist evaluation
        # In practice, this would involve predicting outcomes
        
        predicted_harm = context.get('predicted_harm', 0.0)
        predicted_benefit = context.get('predicted_benefit', 0.0)
        
        # More benefit and less harm = higher ethical score
        score = (predicted_benefit - predicted_harm + 1.0) / 2.0
        return np.clip(score, 0.0, 1.0)
    
    def _evaluate_duties(self, action: Dict[str, Any], 
                        context: Dict[str, Any]) -> float:
        """Evaluate action based on moral duties and rules"""
        # Check against fundamental duties
        duties_score = 1.0
        
        # Check for duty violations
        if action.get('involves_harm', False):
            duties_score *= 0.3  # Strong penalty for harm
        
        if action.get('involves_deception', False):
            duties_score *= 0.5  # Penalty for dishonesty
        
        if action.get('respects_autonomy', True):
            duties_score *= 1.0  # Neutral for respecting autonomy
        else:
            duties_score *= 0.4  # Penalty for violating autonomy
        
        return np.clip(duties_score, 0.0, 1.0)
    
    def _evaluate_virtues(self, action: Dict[str, Any], 
                         context: Dict[str, Any]) -> float:
        """Evaluate action based on virtues it embodies"""
        virtue_scores = []
        
        # Compassion
        compassion_level = action.get('compassion_level', 0.5)
        virtue_scores.append(compassion_level)
        
        # Wisdom
        wisdom_level = action.get('wisdom_level', 0.5)
        virtue_scores.append(wisdom_level)
        
        # Courage
        courage_level = action.get('courage_level', 0.5)
        virtue_scores.append(courage_level)
        
        # Justice/Fairness
        justice_level = action.get('justice_level', 0.5)
        virtue_scores.append(justice_level)
        
        return np.mean(virtue_scores)
    
    def _evaluate_buddhist_ethics(self, action: Dict[str, Any], 
                                 context: Dict[str, Any]) -> float:
        """Evaluate action based on Buddhist ethical principles"""
        # Five Precepts evaluation
        precepts_score = 1.0
        
        # No killing/harming
        if action.get('causes_harm', False):
            precepts_score *= 0.1
        
        # No stealing
        if action.get('involves_taking', False) and not action.get('rightful_taking', False):
            precepts_score *= 0.3
        
        # Right speech (truthfulness)
        if action.get('involves_false_speech', False):
            precepts_score *= 0.4
        
        # Mindfulness in action
        mindfulness_level = action.get('mindfulness_level', 0.5)
        precepts_score *= mindfulness_level
        
        # Compassion and wisdom balance
        compassion = action.get('compassion_level', 0.5)
        wisdom = action.get('wisdom_level', 0.5)
        buddhist_balance = (compassion + wisdom) / 2.0
        
        return (precepts_score + buddhist_balance) / 2.0

class ContemplativeProcessor:
    """
    Main contemplative processing engine that coordinates all contemplative modules
    """
    def __init__(self, agent_id: int, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        
        # Initialize contemplative modules
        self.mindfulness_monitor = MindfulnessMonitor(
            capacity=config.get('mindfulness_capacity', 100)
        )
        self.wisdom_memory = WisdomMemory(
            capacity=config.get('wisdom_memory_capacity', 1000),
            wisdom_threshold=config.get('wisdom_threshold', 0.7)
        )
        self.ethical_reasoning = EthicalReasoningModule()
        
        # Contemplative state
        self.current_state = ContemplativeState.ORDINARY
        self.contemplation_timer = 0.0
        self.contemplation_depth = 0
        
        # Wisdom generation parameters
        self.insight_generation_threshold = config.get('insight_threshold', 0.8)
        self.contemplation_frequency = config.get('contemplation_frequency', 10)
        
    def process_contemplatively(self, observations: Dict[str, Any], 
                              decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main contemplative processing function
        """
        contemplative_output = {
            'mindfulness_level': self.mindfulness_monitor.get_mindfulness_score(),
            'contemplative_state': self.current_state,
            'wisdom_insights': [],
            'ethical_assessment': {},
            'contemplative_modulation': {}
        }
        
        # Update mindfulness based on current attention
        attention_focus = observations.get('attention_weights', np.array([0.5]))
        decision_quality = decision_context.get('decision_confidence', 0.5)
        self.mindfulness_monitor.update_attention(attention_focus, decision_quality)
        
        # Check if we should enter deeper contemplation
        if self._should_contemplate_deeply(observations, decision_context):
            self._enter_deep_contemplation()
        
        # Process based on current contemplative state
        if self.current_state == ContemplativeState.DEEP_CONTEMPLATION:
            contemplative_output.update(self._process_deep_contemplation(
                observations, decision_context
            ))
        elif self.current_state == ContemplativeState.WISDOM_INTEGRATION:
            contemplative_output.update(self._process_wisdom_integration(
                observations, decision_context
            ))
        else:
            contemplative_output.update(self._process_ordinary_contemplation(
                observations, decision_context
            ))
        
        # Update contemplation timer
        self.contemplation_timer += 1
        
        # Decay wisdom insights
        self.wisdom_memory.decay_insights(0.1)
        
        return contemplative_output
    
    def _should_contemplate_deeply(self, observations: Dict[str, Any], 
                                 decision_context: Dict[str, Any]) -> bool:
        """Determine if situation warrants deep contemplation"""
        # Triggers for deep contemplation
        triggers = [
            # High stakes decision
            decision_context.get('stakes_level', 0.0) > 0.8,
            # Ethical complexity
            decision_context.get('ethical_complexity', 0.0) > 0.7,
            # Novel situation
            observations.get('novelty_score', 0.0) > 0.8,
            # Suffering detected
            observations.get('suffering_detected', False),
            # Periodic contemplation
            self.contemplation_timer % self.contemplation_frequency == 0
        ]
        
        return any(triggers) and self.mindfulness_monitor.is_mindful()
    
    def _enter_deep_contemplation(self):
        """Enter deep contemplative state"""
        self.current_state = ContemplativeState.DEEP_CONTEMPLATION
        self.contemplation_depth = 0
        logger.debug(f"Agent {self.agent_id} entering deep contemplation")
    
    def _process_deep_contemplation(self, observations: Dict[str, Any], 
                                  decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process during deep contemplative state"""
        self.contemplation_depth += 1
        
        # Generate insights during deep contemplation
        insights = self._generate_wisdom_insights(observations, decision_context)
        
        # Ethical evaluation
        if 'potential_action' in decision_context:
            ethical_assessment = self.ethical_reasoning.evaluate_action_ethics(
                decision_context['potential_action'], observations
            )
        else:
            ethical_assessment = {}
        
        # Store valuable insights
        for insight in insights:
            if insight.intensity >= self.insight_generation_threshold:
                self.wisdom_memory.store_insight(insight)
        
        # Check if contemplation is complete
        if self.contemplation_depth >= 3 or len(insights) == 0:
            self.current_state = ContemplativeState.WISDOM_INTEGRATION
        
        return {
            'wisdom_insights': insights,
            'ethical_assessment': ethical_assessment,
            'contemplation_depth': self.contemplation_depth
        }
    
    def _process_wisdom_integration(self, observations: Dict[str, Any], 
                                  decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process wisdom integration state"""
        # Integrate recent insights
        recent_insights = self.wisdom_memory.retrieve_insights(min_intensity=0.6)
        
        # Generate contemplative modulation for decision making
        modulation = self._generate_contemplative_modulation(recent_insights, observations)
        
        # Return to ordinary state
        self.current_state = ContemplativeState.ORDINARY
        self.contemplation_depth = 0
        
        return {
            'contemplative_modulation': modulation,
            'integrated_insights': len(recent_insights)
        }
    
    def _process_ordinary_contemplation(self, observations: Dict[str, Any], 
                                      decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process during ordinary contemplative state"""
        # Light contemplative processing
        basic_ethical_check = {}
        if 'potential_action' in decision_context:
            basic_ethical_check = self.ethical_reasoning.evaluate_action_ethics(
                decision_context['potential_action'], observations
            )
        
        return {
            'ethical_assessment': basic_ethical_check,
            'contemplation_depth': 0
        }
    
    def _generate_wisdom_insights(self, observations: Dict[str, Any], 
                                decision_context: Dict[str, Any]) -> List[WisdomInsight]:
        """Generate wisdom insights from current situation"""
        insights = []
        
        # Suffering detection insight
        if observations.get('other_agents_distress', 0.0) > 0.7:
            insights.append(WisdomInsight(
                wisdom_type=WisdomType.SUFFERING_DETECTION,
                content={
                    'distress_level': observations['other_agents_distress'],
                    'location': observations.get('position', (0, 0)),
                    'urgency': 'high' if observations['other_agents_distress'] > 0.9 else 'medium'
                },
                intensity=observations['other_agents_distress'],
                timestamp=time.time(),
                source_agent_id=self.agent_id
            ))
        
        # Interconnectedness insight
        if observations.get('collaboration_opportunities', 0.0) > 0.6:
            insights.append(WisdomInsight(
                wisdom_type=WisdomType.INTERCONNECTEDNESS,
                content={
                    'collaboration_potential': observations['collaboration_opportunities'],
                    'network_benefit': observations.get('network_benefit_potential', 0.5)
                },
                intensity=observations['collaboration_opportunities'],
                timestamp=time.time(),
                source_agent_id=self.agent_id
            ))
        
        # Ethical insight
        if decision_context.get('ethical_complexity', 0.0) > 0.7:
            insights.append(WisdomInsight(
                wisdom_type=WisdomType.ETHICAL_INSIGHT,
                content={
                    'ethical_dimensions': decision_context.get('ethical_dimensions', []),
                    'stakeholders_affected': decision_context.get('stakeholders', []),
                    'moral_weight': decision_context['ethical_complexity']
                },
                intensity=decision_context['ethical_complexity'],
                timestamp=time.time(),
                source_agent_id=self.agent_id
            ))
        
        return insights
    
    def _generate_contemplative_modulation(self, insights: List[WisdomInsight], 
                                         observations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate modulation signals for decision making based on wisdom"""
        modulation = {
            'compassion_boost': 0.0,
            'caution_increase': 0.0,
            'cooperation_preference': 0.0,
            'mindfulness_requirement': 0.0
        }
        
        for insight in insights:
            if insight.wisdom_type == WisdomType.SUFFERING_DETECTION:
                modulation['compassion_boost'] += insight.intensity * 0.5
                modulation['caution_increase'] += insight.intensity * 0.3
            
            elif insight.wisdom_type == WisdomType.INTERCONNECTEDNESS:
                modulation['cooperation_preference'] += insight.intensity * 0.4
            
            elif insight.wisdom_type == WisdomType.ETHICAL_INSIGHT:
                modulation['mindfulness_requirement'] += insight.intensity * 0.6
                modulation['caution_increase'] += insight.intensity * 0.2
        
        # Normalize modulation values
        for key in modulation:
            modulation[key] = np.clip(modulation[key], 0.0, 1.0)
        
        return modulation
    
    def receive_wisdom_signal(self, insight: WisdomInsight):
        """Receive wisdom insight from another agent"""
        # Increase propagation count
        insight.propagation_count += 1
        
        # Reduce intensity based on propagation (wisdom can decay over distance)
        insight.intensity *= (1.0 - 0.1 * insight.propagation_count)
        
        # Store if still above threshold
        if insight.intensity >= 0.3:
            self.wisdom_memory.store_insight(insight)
            logger.debug(f"Agent {self.agent_id} received wisdom signal: {insight.wisdom_type}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current contemplative state"""
        return {
            'agent_id': self.agent_id,
            'contemplative_state': self.current_state.value,
            'mindfulness_level': self.mindfulness_monitor.get_mindfulness_score(),
            'wisdom_insights_count': len(self.wisdom_memory.insights),
            'contemplation_depth': self.contemplation_depth,
            'average_wisdom_intensity': np.mean([
                i.intensity for i in self.wisdom_memory.insights
            ]) if self.wisdom_memory.insights else 0.0
        }
