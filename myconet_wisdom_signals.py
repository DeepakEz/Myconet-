"""
MycoNet++ Wisdom Signal System
=============================

Enhanced chemical communication system for propagating wisdom insights
through the mycelial network. Extends the existing pheromone system
with contemplative signal types.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from myconet_contemplative_core import WisdomInsight, WisdomType

logger = logging.getLogger(__name__)

class WisdomSignalType(Enum):
    """Types of wisdom signals that can propagate through the network"""
    ETHICAL_INSIGHT = "ethical_insight"
    SUFFERING_ALERT = "suffering_alert"
    COMPASSION_GRADIENT = "compassion_gradient"
    WISDOM_BEACON = "wisdom_beacon"
    MEDITATION_SYNC = "meditation_sync"
    COOPERATION_CALL = "cooperation_call"
    CAUTION_WARNING = "caution_warning"
    MINDFULNESS_WAVE = "mindfulness_wave"

@dataclass
class WisdomSignalConfig:
    """Configuration for wisdom signal propagation"""
    signal_types: List[WisdomSignalType] = field(default_factory=lambda: list(WisdomSignalType))
    base_diffusion_rate: float = 0.1
    base_decay_rate: float = 0.05
    propagation_distance: int = 5
    intensity_threshold: float = 0.1
    cross_signal_interference: bool = True
    signal_amplification: Dict[WisdomSignalType, float] = field(default_factory=dict)

class WisdomSignalLayer:
    """
    Individual layer for a specific type of wisdom signal
    Similar to pheromone layers but with wisdom-specific properties
    """
    def __init__(self, width: int, height: int, signal_type: WisdomSignalType, config: WisdomSignalConfig):
        self.width = width
        self.height = height
        self.signal_type = signal_type
        self.config = config
        
        # Signal intensity grid
        self.intensity_grid = np.zeros((height, width), dtype=np.float32)
        
        # Signal content grid (stores actual wisdom content)
        self.content_grid = [[None for _ in range(width)] for _ in range(height)]
        
        # Signal metadata
        self.source_agents = np.zeros((height, width), dtype=np.int32)  # Which agent created signal
        self.creation_time = np.zeros((height, width), dtype=np.float32)
        
        # Signal-specific properties
        self.diffusion_rate = config.base_diffusion_rate
        self.decay_rate = config.base_decay_rate
        
        # Apply signal-specific modifiers
        if signal_type in config.signal_amplification:
            amplification = config.signal_amplification[signal_type]
            self.diffusion_rate *= amplification
        
        # Different signal types have different propagation characteristics
        self._configure_signal_properties()
    
    def _configure_signal_properties(self):
        """Configure signal properties based on wisdom signal type"""
        if self.signal_type == WisdomSignalType.SUFFERING_ALERT:
            self.diffusion_rate *= 1.5  # Suffering alerts spread quickly
            self.decay_rate *= 0.5      # But persist longer
        
        elif self.signal_type == WisdomSignalType.MEDITATION_SYNC:
            self.diffusion_rate *= 2.0  # Meditation sync spreads very quickly
            self.decay_rate *= 2.0      # But decays quickly too
        
        elif self.signal_type == WisdomSignalType.WISDOM_BEACON:
            self.diffusion_rate *= 0.7  # Wisdom spreads more slowly
            self.decay_rate *= 0.3      # But persists much longer
        
        elif self.signal_type == WisdomSignalType.COMPASSION_GRADIENT:
            self.diffusion_rate *= 1.2  # Moderate spread
            self.decay_rate *= 0.6      # Good persistence
    
    def add_signal(self, x: int, y: int, intensity: float, 
                  insight: Optional[WisdomInsight] = None, agent_id: int = 0):
        """Add a wisdom signal at the specified location"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Add to existing intensity
            self.intensity_grid[y, x] = min(1.0, self.intensity_grid[y, x] + intensity)
            
            # Store insight content if provided
            if insight:
                self.content_grid[y][x] = insight
            
            # Track source and timing
            if self.intensity_grid[y, x] > 0:
                self.source_agents[y, x] = agent_id
                self.creation_time[y, x] = 0.0  # Reset creation time
    
    def get_signal_strength(self, x: int, y: int) -> float:
        """Get signal intensity at location"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.intensity_grid[y, x]
        return 0.0
    
    def get_signal_content(self, x: int, y: int) -> Optional[WisdomInsight]:
        """Get wisdom insight content at location"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.content_grid[y][x]
        return None
    
    def get_signal_gradient(self, x: int, y: int, radius: int = 1) -> Tuple[float, float]:
        """Get signal gradient (direction of strongest signal) at location"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return (0.0, 0.0)
        
        # Calculate gradient in 8 directions
        dx, dy = 0.0, 0.0
        center_intensity = self.intensity_grid[y, x]
        
        for dy_offset in [-radius, 0, radius]:
            for dx_offset in [-radius, 0, radius]:
                if dx_offset == 0 and dy_offset == 0:
                    continue
                
                nx, ny = x + dx_offset, y + dy_offset
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_intensity = self.intensity_grid[ny, nx]
                    gradient_strength = neighbor_intensity - center_intensity
                    
                    # Normalize direction
                    length = np.sqrt(dx_offset**2 + dy_offset**2)
                    if length > 0:
                        dx += (dx_offset / length) * gradient_strength
                        dy += (dy_offset / length) * gradient_strength
        
        return (dx, dy)
    
    def update_diffusion(self, time_step: float = 1.0):
        """Update signal diffusion and decay"""
        # Create temporary grid for diffusion calculation
        new_intensity = np.copy(self.intensity_grid)
        
        # Apply diffusion (signals spread to neighboring cells)
        for y in range(self.height):
            for x in range(self.width):
                if self.intensity_grid[y, x] > self.config.intensity_threshold:
                    current_intensity = self.intensity_grid[y, x]
                    diffusion_amount = current_intensity * self.diffusion_rate * time_step
                    
                    # Spread to 8 neighboring cells
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                # Distance-based diffusion
                                distance = np.sqrt(dx**2 + dy**2)
                                diffusion_coefficient = diffusion_amount / (8.0 * distance)
                                new_intensity[ny, nx] += diffusion_coefficient
                                
                                # Also propagate content if strong enough
                                if (diffusion_coefficient > 0.3 and 
                                    self.content_grid[y][x] and 
                                    not self.content_grid[ny][nx]):
                                    self.content_grid[ny][nx] = self.content_grid[y][x]
        
        # Apply decay
        new_intensity *= (1.0 - self.decay_rate * time_step)
        
        # Update creation time and apply time-based decay
        self.creation_time += time_step
        time_decay = np.exp(-self.creation_time * 0.1)  # Exponential time decay
        new_intensity *= time_decay
        
        # Clean up very weak signals
        weak_mask = new_intensity < self.config.intensity_threshold
        new_intensity[weak_mask] = 0.0
        
        # Clear content where intensity is too low
        for y in range(self.height):
            for x in range(self.width):
                if new_intensity[y, x] < self.config.intensity_threshold:
                    self.content_grid[y][x] = None
                    self.source_agents[y, x] = 0
                    self.creation_time[y, x] = 0.0
        
        self.intensity_grid = new_intensity
    
    def get_local_signals(self, x: int, y: int, radius: int = 2) -> List[Tuple[WisdomInsight, float, Tuple[int, int]]]:
        """Get all wisdom signals within radius of location"""
        signals = []
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    intensity = self.intensity_grid[ny, nx]
                    content = self.content_grid[ny][nx]
                    
                    if intensity > self.config.intensity_threshold and content:
                        distance = np.sqrt(dx**2 + dy**2)
                        # Adjust intensity by distance
                        adjusted_intensity = intensity * (1.0 - distance / (radius + 1))
                        signals.append((content, adjusted_intensity, (nx, ny)))
        
        return signals

class WisdomSignalGrid:
    """
    Enhanced grid system for propagating wisdom signals through the mycelial network
    Extends and integrates with existing pheromone systems
    """
    def __init__(self, width: int, height: int, config):
        self.width = width
        self.height = height

        # Handle both dictionary and object configuration
        if isinstance(config, dict):
            self.signal_types = config.get('signal_types', [])
            self.base_diffusion_rate = config.get('base_diffusion_rate', 0.1)
            self.base_decay_rate = config.get('base_decay_rate', 0.05)
            self.propagation_distance = config.get('propagation_distance', 5)
            self.intensity_threshold = config.get('intensity_threshold', 0.1)
            self.cross_signal_interference = config.get('cross_signal_interference', True)
            self.signal_amplification = config.get('signal_amplification', {})
        else:
            # Object-style access (original code)
            self.signal_types = config.signal_types
            self.base_diffusion_rate = config.base_diffusion_rate
            self.base_decay_rate = config.base_decay_rate
            self.propagation_distance = config.propagation_distance
            self.intensity_threshold = config.intensity_threshold
            self.cross_signal_interference = config.cross_signal_interference
            self.signal_amplification = config.signal_amplification

        # Create proper WisdomSignalConfig object for the layers
        signal_config = WisdomSignalConfig(
            signal_types=self.signal_types,
            base_diffusion_rate=self.base_diffusion_rate,
            base_decay_rate=self.base_decay_rate,
            propagation_distance=self.propagation_distance,
            intensity_threshold=self.intensity_threshold,
            cross_signal_interference=self.cross_signal_interference,
            signal_amplification=self.signal_amplification
        )

        # Initialize signal layers for each type - CREATE PROPER OBJECTS!
        self.signal_layers = {}
        for signal_type in self.signal_types:
            # Handle both enum objects and string representations
            if hasattr(signal_type, 'value'):
                signal_enum = signal_type
                signal_name = signal_type.value
            elif isinstance(signal_type, str):
                # Handle string representations like "WisdomSignalType.ETHICAL_INSIGHT"
                if signal_type.startswith("WisdomSignalType."):
                    signal_name = signal_type.split(".")[-1]
                else:
                    signal_name = signal_type
                    
                # Convert string to enum safely
                try:
                    signal_enum = WisdomSignalType(signal_name.lower())
                except ValueError:
                    # Handle case where signal_name doesn't match enum values
                    signal_mapping = {
                        'ETHICAL_INSIGHT': WisdomSignalType.ETHICAL_INSIGHT,
                        'SUFFERING_ALERT': WisdomSignalType.SUFFERING_ALERT,
                        'COMPASSION_GRADIENT': WisdomSignalType.COMPASSION_GRADIENT,
                        'WISDOM_BEACON': WisdomSignalType.WISDOM_BEACON,
                        'MEDITATION_SYNC': WisdomSignalType.MEDITATION_SYNC,
                        'COOPERATION_CALL': WisdomSignalType.COOPERATION_CALL,
                        'CAUTION_WARNING': WisdomSignalType.CAUTION_WARNING,
                        'MINDFULNESS_WAVE': WisdomSignalType.MINDFULNESS_WAVE
                    }
                    signal_enum = signal_mapping.get(signal_name, WisdomSignalType.WISDOM_BEACON)
            else:
                signal_name = str(signal_type)
                signal_enum = WisdomSignalType.WISDOM_BEACON  # Default fallback
             
            # CREATE PROPER WisdomSignalLayer OBJECTS (not numpy arrays!)
            self.signal_layers[signal_enum] = WisdomSignalLayer(
                width=width, 
                height=height, 
                signal_type=signal_enum, 
                config=signal_config
            )

        # Initialize other attributes that your original code expects
        self.total_signals_created = 0
        self.total_signals_propagated = 0
        self.signal_statistics = defaultdict(int)
        self.interference_matrix = self._create_interference_matrix()
    
        logger.info(f"Initialized WisdomSignalGrid ({width}x{height}) with {len(self.signal_layers)} signal types")
    
    def _create_interference_matrix(self) -> Dict[Tuple[WisdomSignalType, WisdomSignalType], float]:
        """Create matrix defining how different signals interfere with each other"""
        interference = {}
        
        # Define signal interactions
        signal_interactions = {
            # Meditation sync amplifies wisdom signals
            (WisdomSignalType.MEDITATION_SYNC, WisdomSignalType.WISDOM_BEACON): 1.3,
            (WisdomSignalType.MEDITATION_SYNC, WisdomSignalType.MINDFULNESS_WAVE): 1.5,
            
            # Suffering alerts amplify compassion
            (WisdomSignalType.SUFFERING_ALERT, WisdomSignalType.COMPASSION_GRADIENT): 1.4,
            
            # Caution warnings can interfere with cooperation calls
            (WisdomSignalType.CAUTION_WARNING, WisdomSignalType.COOPERATION_CALL): 0.7,
            
            # Wisdom beacons amplify ethical insights
            (WisdomSignalType.WISDOM_BEACON, WisdomSignalType.ETHICAL_INSIGHT): 1.2,
        }
        
        # Initialize all interactions to neutral (1.0)
        for signal1 in WisdomSignalType:
            for signal2 in WisdomSignalType:
                interference[(signal1, signal2)] = 1.0
        
        # Apply specific interactions (and their reverses)
        for (signal1, signal2), factor in signal_interactions.items():
            interference[(signal1, signal2)] = factor
            interference[(signal2, signal1)] = factor
        
        return interference
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics for analysis - REQUIRED BY MAIN FILE - FIXED DIVISION BY ZERO"""
        # Count active signals by type
        signal_counts = defaultdict(int)
        total_intensity = 0.0
        active_signals = 0
    
        for signal_type, layer in self.signal_layers.items():
            layer_intensity = np.sum(layer.intensity_grid)
            if layer_intensity > self.intensity_threshold:
                signal_counts[signal_type.value] = int(np.sum(layer.intensity_grid > self.intensity_threshold))
                total_intensity += layer_intensity
                active_signals += signal_counts[signal_type.value]
    
        # FIXED: Calculate diversity safely (avoid division by zero)
        total_signal_types = len(WisdomSignalType)
        if total_signal_types > 0:
            signal_diversity = len(signal_counts) / total_signal_types
        else:
            signal_diversity = 0.0
    
        # FIXED: Calculate network coherence safely (avoid division by zero)
        if active_signals > 0:
            network_coherence = total_intensity / active_signals
        else:
            network_coherence = 0.0
    
        # Calculate wisdom flow efficiency
        max_possible_signals = self.width * self.height * 0.1
        if max_possible_signals > 0:
            wisdom_flow_efficiency = min(1.0, active_signals / max_possible_signals)
        else:
            wisdom_flow_efficiency = 0.0
    
        return {
            'signal_diversity': signal_diversity,
            'network_coherence': network_coherence,
            'wisdom_flow_efficiency': wisdom_flow_efficiency,
            'total_signals': active_signals,
            'active_signals': active_signals,
            'signal_counts_by_type': dict(signal_counts),
            'average_signal_intensity': network_coherence,
            'total_signal_intensity': total_intensity
        }

    def add_signal(self, signal_type: WisdomSignalType, x: int, y: int, intensity: float, 
                   source_agent_id: int = -1, insight: Optional[WisdomInsight] = None):
        """Add a signal to the grid - REQUIRED BY MAIN FILE"""
        if signal_type in self.signal_layers:
            self.signal_layers[signal_type].add_signal(x, y, intensity, insight, source_agent_id)
            self.total_signals_created += 1
            self.signal_statistics[signal_type] += 1

    def amplify_signal_type(self, signal_type: WisdomSignalType, amplification_factor: float):
        """Amplify all signals of a specific type - REQUIRED BY MAIN FILE"""
        if signal_type in self.signal_layers:
            layer = self.signal_layers[signal_type]
            layer.intensity_grid *= amplification_factor
            # Clip to prevent oversaturation
            layer.intensity_grid = np.clip(layer.intensity_grid, 0.0, 1.0)
    
    def propagate_wisdom_signal(self, x: int, y: int, signal_type: WisdomSignalType, 
                               intensity: float, insight: Optional[WisdomInsight] = None, 
                               agent_id: int = 0):
        """Propagate a wisdom signal from a specific location"""
        if signal_type in self.signal_layers:
            self.signal_layers[signal_type].add_signal(x, y, intensity, insight, agent_id)
            self.total_signals_created += 1
            self.signal_statistics[signal_type] += 1
            
            logger.debug(f"Wisdom signal {signal_type.value} propagated at ({x}, {y}) "
                        f"with intensity {intensity:.2f}")
    
    def get_signals_at_location(self, x: int, y: int, radius: int = 1) -> Dict[WisdomSignalType, List[Tuple[WisdomInsight, float]]]:
        """Get all wisdom signals at a location"""
        location_signals = {}
        
        for signal_type, layer in self.signal_layers.items():
            signals = layer.get_local_signals(x, y, radius)
            if signals:
                location_signals[signal_type] = [(insight, intensity) for insight, intensity, pos in signals]
        
        return location_signals
    
    def get_signal_gradients(self, x: int, y: int) -> Dict[WisdomSignalType, Tuple[float, float]]:
        """Get signal gradients for all signal types at location"""
        gradients = {}
        for signal_type, layer in self.signal_layers.items():
            gradients[signal_type] = layer.get_signal_gradient(x, y)
        return gradients
    
    def update_all_signals(self, time_step: float = 1.0):
        """Update diffusion and decay for all signal types"""
        # Update individual layers
        for layer in self.signal_layers.values():
            layer.update_diffusion(time_step)
    
        # Apply cross-signal interference if enabled
        if self.cross_signal_interference:  # Use self.cross_signal_interference instead of self.config.cross_signal_interference
            self._apply_signal_interference()
    
    def _apply_signal_interference(self):
        """Apply interference between different signal types"""
        # Create temporary intensity modifications
        intensity_modifications = {}
        
        for signal_type in self.signal_layers:
            intensity_modifications[signal_type] = np.ones_like(
                self.signal_layers[signal_type].intensity_grid
            )
        
        # Calculate interference effects
        for signal1_type, layer1 in self.signal_layers.items():
            for signal2_type, layer2 in self.signal_layers.items():
                if signal1_type != signal2_type:
                    interference_factor = self.interference_matrix.get(
                        (signal1_type, signal2_type), 1.0
                    )
                    
                    if interference_factor != 1.0:
                        # Apply interference based on overlapping signal strengths
                        overlap_mask = (layer1.intensity_grid > 0.1) & (layer2.intensity_grid > 0.1)
                        interference_strength = layer1.intensity_grid * (interference_factor - 1.0)
                        intensity_modifications[signal2_type][overlap_mask] += interference_strength[overlap_mask]
        
        # Apply modifications
        for signal_type, layer in self.signal_layers.items():
            modification = intensity_modifications[signal_type]
            layer.intensity_grid *= np.clip(modification, 0.1, 2.0)  # Limit interference effects
    
    def trigger_network_meditation(self, center_x: int, center_y: int, radius: int = 10, intensity: float = 0.8):
        """Trigger a network-wide meditation synchronization signal"""
        meditation_signal = WisdomInsight(
            wisdom_type=WisdomType.PRACTICAL_WISDOM,
            content={
                'meditation_type': 'network_synchronization',
                'center': (center_x, center_y),
                'radius': radius,
                'synchronization_frequency': 'collective'
            },
            intensity=intensity,
            timestamp=0.0,
            source_agent_id=-1  # Special ID for network-wide signals
        )
        
        # Create expanding circle of meditation sync signals
        for y in range(max(0, center_y - radius), min(self.height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(self.width, center_x + radius + 1)):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= radius:
                    # Intensity decreases with distance from center
                    signal_intensity = intensity * (1.0 - distance / radius)
                    self.propagate_wisdom_signal(
                        x, y, WisdomSignalType.MEDITATION_SYNC, 
                        signal_intensity, meditation_signal, -1
                    )
    
    def detect_suffering_areas(self, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """Detect areas with high suffering alert signals"""
        if WisdomSignalType.SUFFERING_ALERT not in self.signal_layers:
            return []
        
        suffering_layer = self.signal_layers[WisdomSignalType.SUFFERING_ALERT]
        suffering_areas = []
        
        for y in range(self.height):
            for x in range(self.width):
                intensity = suffering_layer.intensity_grid[y, x]
                if intensity >= threshold:
                    suffering_areas.append((x, y, intensity))
        
        return suffering_areas
    
    def get_wisdom_hotspots(self, min_intensity: float = 0.6) -> Dict[WisdomSignalType, List[Tuple[int, int, float]]]:
        """Identify locations with high wisdom signal concentrations"""
        hotspots = {}
        
        for signal_type, layer in self.signal_layers.items():
            type_hotspots = []
            for y in range(self.height):
                for x in range(self.width):
                    intensity = layer.intensity_grid[y, x]
                    if intensity >= min_intensity:
                        type_hotspots.append((x, y, intensity))
            
            if type_hotspots:
                hotspots[signal_type] = type_hotspots
        
        return hotspots
    
    def calculate_network_wisdom_metrics(self) -> Dict[str, float]:
        """Calculate network-wide wisdom signal metrics"""
        metrics = {
            'total_signal_intensity': 0.0,
            'signal_diversity': 0.0,
            'network_contemplative_coherence': 0.0,
            'wisdom_flow_efficiency': 0.0
        }
        
        # Calculate total intensity across all signals
        total_intensity = 0.0
        active_signal_types = 0
        
        for signal_type, layer in self.signal_layers.items():
            layer_intensity = np.sum(layer.intensity_grid)
            total_intensity += layer_intensity
            
            if layer_intensity > 0.1:
                active_signal_types += 1
        
        metrics['total_signal_intensity'] = total_intensity
        metrics['signal_diversity'] = active_signal_types / max(1, len(self.signal_layers)) if hasattr(self, 'signal_layers') and self.signal_layers else 0.0
        
        # Calculate contemplative coherence (how synchronized the network is)
        if WisdomSignalType.MEDITATION_SYNC in self.signal_layers:
            meditation_layer = self.signal_layers[WisdomSignalType.MEDITATION_SYNC]
            meditation_coverage = np.sum(meditation_layer.intensity_grid > 0.3) / (self.width * self.height)
            metrics['network_contemplative_coherence'] = meditation_coverage
        
        # Calculate wisdom flow efficiency (how well wisdom propagates)
        if self.total_signals_created > 0:
            metrics['wisdom_flow_efficiency'] = self.total_signals_propagated / self.total_signals_created
        
        return metrics
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about signal propagation"""
        return {
            'total_signals_created': self.total_signals_created,
            'total_signals_propagated': self.total_signals_propagated,
            'signals_by_type': dict(self.signal_statistics),
            'network_metrics': self.calculate_network_wisdom_metrics(),
            'active_signal_layers': len([
                layer for layer in self.signal_layers.values() 
                if np.sum(layer.intensity_grid) > 0.1
            ])
        }
    
    def visualize_signals(self, signal_type: Optional[WisdomSignalType] = None) -> np.ndarray:
        """Create visualization array for signal intensities"""
        if signal_type and signal_type in self.signal_layers:
            return self.signal_layers[signal_type].intensity_grid.copy()
        else:
            # Combine all signal types
            combined = np.zeros((self.height, self.width))
            for layer in self.signal_layers.values():
                combined += layer.intensity_grid
            return np.clip(combined, 0.0, 1.0)

class WisdomSignalProcessor:
    """
    Processes wisdom signals for individual agents
    Integrates with agent's contemplative processing
    """
    def __init__(self, agent_id: int, signal_grid: WisdomSignalGrid):
        self.agent_id = agent_id
        self.signal_grid = signal_grid
        self.signal_sensitivity = {
            WisdomSignalType.SUFFERING_ALERT: 0.9,
            WisdomSignalType.MEDITATION_SYNC: 0.8,
            WisdomSignalType.WISDOM_BEACON: 0.7,
            WisdomSignalType.COMPASSION_GRADIENT: 0.8,
            WisdomSignalType.ETHICAL_INSIGHT: 0.7,
            WisdomSignalType.COOPERATION_CALL: 0.6,
            WisdomSignalType.CAUTION_WARNING: 0.8,
            WisdomSignalType.MINDFULNESS_WAVE: 0.7
        }
        
        self.last_signals_received = {}
        self.signal_response_history = []
    
    def process_local_signals(self, x: int, y: int, radius: int = 2) -> Dict[str, Any]:
        """Process wisdom signals in the local area"""
        local_signals = self.signal_grid.get_signals_at_location(x, y, radius)
        signal_gradients = self.signal_grid.get_signal_gradients(x, y)
        
        processed_signals = {
            'received_insights': [],
            'signal_influences': {},
            'recommended_actions': [],
            'emotional_state_modifiers': {}
        }
        
        # Process each signal type
        for signal_type, signals in local_signals.items():
            sensitivity = self.signal_sensitivity.get(signal_type, 0.5)
            
            for insight, intensity in signals:
                adjusted_intensity = intensity * sensitivity
                
                if adjusted_intensity > 0.3:  # Threshold for signal response
                    processed_signals['received_insights'].append({
                        'insight': insight,
                        'intensity': adjusted_intensity,
                        'signal_type': signal_type
                    })
                    
                    # Generate specific responses based on signal type
                    self._generate_signal_response(signal_type, adjusted_intensity, processed_signals)
        
        # Process gradients for movement guidance
        for signal_type, (dx, dy) in signal_gradients.items():
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                processed_signals['signal_influences'][signal_type] = {
                    'gradient_x': dx,
                    'gradient_y': dy,
                    'strength': np.sqrt(dx**2 + dy**2)
                }
        
        self.last_signals_received = processed_signals
        return processed_signals
    
    def _generate_signal_response(self, signal_type: WisdomSignalType, intensity: float, 
                                processed_signals: Dict[str, Any]):
        """Generate appropriate response to specific signal type"""
        if signal_type == WisdomSignalType.SUFFERING_ALERT:
            processed_signals['recommended_actions'].append({
                'action': 'investigate_suffering',
                'urgency': intensity,
                'type': 'compassionate_response'
            })
            processed_signals['emotional_state_modifiers']['compassion'] = intensity * 0.5
            processed_signals['emotional_state_modifiers']['urgency'] = intensity * 0.3
        
        elif signal_type == WisdomSignalType.MEDITATION_SYNC:
            processed_signals['recommended_actions'].append({
                'action': 'join_meditation',
                'synchronization_strength': intensity,
                'type': 'contemplative_action'
            })
            processed_signals['emotional_state_modifiers']['mindfulness'] = intensity * 0.6
            processed_signals['emotional_state_modifiers']['inner_peace'] = intensity * 0.4
        
        elif signal_type == WisdomSignalType.COOPERATION_CALL:
            processed_signals['recommended_actions'].append({
                'action': 'seek_collaboration',
                'cooperation_urgency': intensity,
                'type': 'social_action'
            })
            processed_signals['emotional_state_modifiers']['social_orientation'] = intensity * 0.5
        
        elif signal_type == WisdomSignalType.WISDOM_BEACON:
            processed_signals['recommended_actions'].append({
                'action': 'approach_wisdom_source',
                'learning_opportunity': intensity,
                'type': 'learning_action'
            })
            processed_signals['emotional_state_modifiers']['curiosity'] = intensity * 0.4
            processed_signals['emotional_state_modifiers']['receptiveness'] = intensity * 0.5
        
        elif signal_type == WisdomSignalType.CAUTION_WARNING:
            processed_signals['recommended_actions'].append({
                'action': 'increase_caution',
                'caution_level': intensity,
                'type': 'protective_action'
            })
            processed_signals['emotional_state_modifiers']['caution'] = intensity * 0.6
            processed_signals['emotional_state_modifiers']['alertness'] = intensity * 0.4
    
    def emit_wisdom_signal(self, x: int, y: int, signal_type: WisdomSignalType, 
                          intensity: float, insight: Optional[WisdomInsight] = None):
        """Emit a wisdom signal from agent's current location"""
        self.signal_grid.propagate_wisdom_signal(x, y, signal_type, intensity, insight, self.agent_id)
        
        # Track emission in response history
        self.signal_response_history.append({
            'action': 'emit_signal',
            'signal_type': signal_type,
            'intensity': intensity,
            'position': (x, y),
            'timestamp': 0.0  # Would be actual timestamp in full implementation
        })
    
    def get_signal_processing_summary(self) -> Dict[str, Any]:
        """Get summary of recent signal processing activity"""
        return {
            'agent_id': self.agent_id,
            'last_signals_count': len(self.last_signals_received.get('received_insights', [])),
            'signal_sensitivity': self.signal_sensitivity,
            'recent_responses': len(self.signal_response_history),
            'most_sensitive_signal': max(self.signal_sensitivity.items(), key=lambda x: x[1])
        }