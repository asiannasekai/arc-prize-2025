"""
Multi-Expert Aggregation System
Implements consensus algorithms and confidence weighting for combining outputs 
from multiple expert models with uncertainty quantification for ARC-AGI tasks.

Key insight: Since current AI systems struggle with novel problems (4% vs 100% human performance),
we need sophisticated aggregation that can handle uncertainty and conflicting expert opinions.
"""

import numpy as np
import json
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpertPrediction:
    """Represents a prediction from a single expert model"""
    expert_type: str
    prediction: np.ndarray  # The predicted output grid
    confidence: float  # Model's confidence in prediction (0-1)
    reasoning: str  # Textual explanation of the reasoning
    processing_time: float  # Time taken to generate prediction
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class AggregatedPrediction:
    """Final aggregated prediction from multiple experts"""
    final_prediction: np.ndarray
    confidence: float
    expert_contributions: Dict[str, float]  # Weight each expert contributed
    consensus_score: float  # How much experts agreed (0-1)
    uncertainty_score: float  # Measure of prediction uncertainty (0-1)
    reasoning: str  # Combined reasoning explanation
    metadata: Dict[str, Any] = field(default_factory=dict)

class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies"""
    
    @abstractmethod
    def aggregate(self, predictions: List[ExpertPrediction]) -> AggregatedPrediction:
        """Aggregate multiple expert predictions into a single result"""
        pass

class WeightedVotingAggregator(AggregationStrategy):
    """Aggregates predictions using weighted voting based on confidence scores"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.3,
                 similarity_weight: float = 0.4,
                 confidence_weight: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.similarity_weight = similarity_weight
        self.confidence_weight = confidence_weight
        
    def _compute_prediction_similarity(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """Compute similarity between two predictions"""
        if pred1.shape != pred2.shape:
            return 0.0
        
        # Exact match score
        exact_match = np.mean(pred1 == pred2)
        
        # Pattern similarity (considering spatial relationships)
        pattern_sim = self._compute_pattern_similarity(pred1, pred2)
        
        return 0.7 * exact_match + 0.3 * pattern_sim
    
    def _compute_pattern_similarity(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """Compute pattern-based similarity considering ARC-specific features"""
        if pred1.shape != pred2.shape:
            return 0.0
            
        # Color distribution similarity
        colors1 = np.unique(pred1)
        colors2 = np.unique(pred2)
        color_overlap = len(set(colors1) & set(colors2)) / len(set(colors1) | set(colors2))
        
        # Structural similarity (connected components)
        struct_sim = self._structural_similarity(pred1, pred2)
        
        return 0.6 * color_overlap + 0.4 * struct_sim
    
    def _structural_similarity(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """Compute structural similarity based on connected components"""
        try:
            from scipy.ndimage import label
            
            # Count connected components for each unique color
            sim_scores = []
            all_colors = set(pred1.flatten()) | set(pred2.flatten())
            
            for color in all_colors:
                mask1 = (pred1 == color).astype(int)
                mask2 = (pred2 == color).astype(int)
                
                if mask1.sum() == 0 and mask2.sum() == 0:
                    continue
                    
                components1 = label(mask1)[1]
                components2 = label(mask2)[1]
                
                # Similarity based on number of components
                if components1 == 0 and components2 == 0:
                    sim_scores.append(1.0)
                else:
                    max_components = max(components1, components2, 1)
                    sim_scores.append(1.0 - abs(components1 - components2) / max_components)
            
            return np.mean(sim_scores) if sim_scores else 0.0
            
        except ImportError:
            # Fallback to simple pixel comparison
            return np.mean(pred1 == pred2)
    
    def aggregate(self, predictions: List[ExpertPrediction]) -> AggregatedPrediction:
        """Aggregate predictions using weighted voting"""
        if not predictions:
            raise ValueError("No predictions to aggregate")
        
        if len(predictions) == 1:
            pred = predictions[0]
            return AggregatedPrediction(
                final_prediction=pred.prediction,
                confidence=pred.confidence,
                expert_contributions={pred.expert_type: 1.0},
                consensus_score=1.0,
                uncertainty_score=1.0 - pred.confidence,
                reasoning=pred.reasoning
            )
        
        # Filter predictions by confidence threshold
        valid_predictions = [p for p in predictions if p.confidence >= self.confidence_threshold]
        if not valid_predictions:
            valid_predictions = predictions  # Use all if none meet threshold
        
        # Compute pairwise similarities
        n = len(valid_predictions)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                sim = self._compute_prediction_similarity(
                    valid_predictions[i].prediction, 
                    valid_predictions[j].prediction
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            similarity_matrix[i, i] = 1.0
        
        # Compute weights based on confidence and agreement with others
        weights = []
        for i, pred in enumerate(valid_predictions):
            # Base weight from confidence
            conf_weight = pred.confidence
            
            # Agreement weight (how similar to other predictions)
            agreement_weight = np.mean(similarity_matrix[i])
            
            # Combined weight
            final_weight = (self.confidence_weight * conf_weight + 
                          self.similarity_weight * agreement_weight)
            weights.append(final_weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Find the prediction with highest weight as base
        best_idx = np.argmax(weights)
        final_prediction = valid_predictions[best_idx].prediction.copy()
        
        # Create expert contributions mapping
        expert_contributions = {}
        for i, pred in enumerate(valid_predictions):
            expert_contributions[pred.expert_type] = float(weights[i])
        
        # Compute consensus score
        consensus_score = float(np.mean(similarity_matrix))
        
        # Compute uncertainty score
        weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(weights))
        uncertainty_score = weight_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Aggregate reasoning
        reasoning_parts = []
        for i, pred in enumerate(valid_predictions):
            if weights[i] > 0.1:  # Include reasoning from significant contributors
                reasoning_parts.append(f"{pred.expert_type} (weight: {weights[i]:.2f}): {pred.reasoning}")
        
        final_reasoning = "\\n".join(reasoning_parts)
        
        return AggregatedPrediction(
            final_prediction=final_prediction,
            confidence=float(np.sum(weights * [p.confidence for p in valid_predictions])),
            expert_contributions=expert_contributions,
            consensus_score=consensus_score,
            uncertainty_score=uncertainty_score,
            reasoning=final_reasoning,
            metadata={
                'similarity_matrix': similarity_matrix.tolist(),
                'weights': weights.tolist(),
                'num_experts': len(valid_predictions)
            }
        )

class BayesianModelAveraging(AggregationStrategy):
    """Bayesian Model Averaging for expert predictions"""
    
    def __init__(self, prior_weights: Optional[Dict[str, float]] = None):
        self.prior_weights = prior_weights or {}
        
    def _compute_likelihood(self, prediction: ExpertPrediction, observations: List[ExpertPrediction]) -> float:
        """Compute likelihood of prediction given observations from other experts"""
        if len(observations) <= 1:
            return prediction.confidence
        
        # Compute how well this prediction agrees with others
        agreements = []
        for obs in observations:
            if obs.expert_type != prediction.expert_type:
                sim = np.mean(prediction.prediction == obs.prediction)
                agreements.append(sim)
        
        if not agreements:
            return prediction.confidence
        
        # Likelihood is combination of confidence and agreement
        agreement_score = np.mean(agreements)
        return 0.7 * prediction.confidence + 0.3 * agreement_score
    
    def aggregate(self, predictions: List[ExpertPrediction]) -> AggregatedPrediction:
        """Aggregate using Bayesian Model Averaging"""
        if not predictions:
            raise ValueError("No predictions to aggregate")
        
        if len(predictions) == 1:
            pred = predictions[0]
            return AggregatedPrediction(
                final_prediction=pred.prediction,
                confidence=pred.confidence,
                expert_contributions={pred.expert_type: 1.0},
                consensus_score=1.0,
                uncertainty_score=1.0 - pred.confidence,
                reasoning=pred.reasoning
            )
        
        # Compute posterior weights using Bayesian updating
        posterior_weights = []
        
        for pred in predictions:
            # Prior weight
            prior = self.prior_weights.get(pred.expert_type, 1.0)
            
            # Likelihood
            likelihood = self._compute_likelihood(pred, predictions)
            
            # Posterior (unnormalized)
            posterior = prior * likelihood
            posterior_weights.append(posterior)
        
        # Normalize weights
        total_weight = sum(posterior_weights)
        if total_weight > 0:
            posterior_weights = [w / total_weight for w in posterior_weights]
        else:
            posterior_weights = [1.0 / len(predictions)] * len(predictions)
        
        # Select prediction with highest posterior weight
        best_idx = np.argmax(posterior_weights)
        final_prediction = predictions[best_idx].prediction
        
        # Create expert contributions
        expert_contributions = {}
        for i, pred in enumerate(predictions):
            expert_contributions[pred.expert_type] = posterior_weights[i]
        
        # Compute consensus and uncertainty
        consensus_scores = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                sim = np.mean(predictions[i].prediction == predictions[j].prediction)
                consensus_scores.append(sim)
        
        consensus_score = np.mean(consensus_scores) if consensus_scores else 1.0
        
        # Uncertainty from weight distribution
        weights_array = np.array(posterior_weights)
        uncertainty_score = 1.0 - np.max(weights_array)  # High uncertainty if weights are spread
        
        # Aggregate confidence
        final_confidence = sum(pred.confidence * weight 
                             for pred, weight in zip(predictions, posterior_weights))
        
        return AggregatedPrediction(
            final_prediction=final_prediction,
            confidence=final_confidence,
            expert_contributions=expert_contributions,
            consensus_score=consensus_score,
            uncertainty_score=uncertainty_score,
            reasoning=f"Bayesian averaging of {len(predictions)} experts",
            metadata={
                'posterior_weights': posterior_weights,
                'prior_weights': self.prior_weights
            }
        )

class ConsensusAggregator(AggregationStrategy):
    """Requires consensus among experts, with fallback strategies"""
    
    def __init__(self, 
                 consensus_threshold: float = 0.7,
                 min_agreeing_experts: int = 2):
        self.consensus_threshold = consensus_threshold
        self.min_agreeing_experts = min_agreeing_experts
    
    def aggregate(self, predictions: List[ExpertPrediction]) -> AggregatedPrediction:
        """Aggregate requiring consensus, with fallback to confidence-based selection"""
        if not predictions:
            raise ValueError("No predictions to aggregate")
        
        if len(predictions) == 1:
            pred = predictions[0]
            return AggregatedPrediction(
                final_prediction=pred.prediction,
                confidence=pred.confidence,
                expert_contributions={pred.expert_type: 1.0},
                consensus_score=1.0,
                uncertainty_score=1.0 - pred.confidence,
                reasoning=pred.reasoning
            )
        
        # Find groups of agreeing predictions
        agreement_groups = []
        used_indices = set()
        
        for i, pred1 in enumerate(predictions):
            if i in used_indices:
                continue
                
            group = [i]
            for j, pred2 in enumerate(predictions[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                similarity = np.mean(pred1.prediction == pred2.prediction)
                if similarity >= self.consensus_threshold:
                    group.append(j)
                    used_indices.add(j)
            
            if len(group) >= self.min_agreeing_experts:
                agreement_groups.append(group)
                used_indices.update(group)
        
        if agreement_groups:
            # Find the largest consensus group
            largest_group = max(agreement_groups, key=len)
            group_predictions = [predictions[i] for i in largest_group]
            
            # Use the highest confidence prediction from the consensus group
            best_pred = max(group_predictions, key=lambda p: p.confidence)
            
            # Compute group statistics
            group_confidences = [p.confidence for p in group_predictions]
            avg_confidence = np.mean(group_confidences)
            
            expert_contributions = {}
            for pred in group_predictions:
                expert_contributions[pred.expert_type] = pred.confidence / sum(group_confidences)
            
            return AggregatedPrediction(
                final_prediction=best_pred.prediction,
                confidence=avg_confidence,
                expert_contributions=expert_contributions,
                consensus_score=1.0,  # Perfect consensus within group
                uncertainty_score=1.0 - avg_confidence,
                reasoning=f"Consensus among {len(group_predictions)} experts: {[p.expert_type for p in group_predictions]}",
                metadata={
                    'consensus_group_size': len(group_predictions),
                    'total_experts': len(predictions)
                }
            )
        else:
            # No consensus found, fall back to highest confidence
            best_pred = max(predictions, key=lambda p: p.confidence)
            
            return AggregatedPrediction(
                final_prediction=best_pred.prediction,
                confidence=best_pred.confidence,
                expert_contributions={best_pred.expert_type: 1.0},
                consensus_score=0.0,  # No consensus
                uncertainty_score=1.0,  # High uncertainty due to disagreement
                reasoning=f"No consensus found, using highest confidence expert: {best_pred.expert_type}",
                metadata={
                    'fallback_used': True,
                    'disagreement_level': 'high'
                }
            )

class MultiExpertAggregationSystem:
    """Main system for aggregating multiple expert model predictions"""
    
    def __init__(self, 
                 aggregation_strategy: AggregationStrategy = None,
                 expert_performance_history: Optional[Dict[str, List[float]]] = None):
        self.strategy = aggregation_strategy or WeightedVotingAggregator()
        self.performance_history = expert_performance_history or {}
        self.prediction_cache = {}
        
    def set_strategy(self, strategy: AggregationStrategy):
        """Change aggregation strategy"""
        self.strategy = strategy
        
    def update_expert_performance(self, expert_type: str, accuracy: float):
        """Update performance history for an expert"""
        if expert_type not in self.performance_history:
            self.performance_history[expert_type] = []
        self.performance_history[expert_type].append(accuracy)
        
        # Keep only recent history (last 100 predictions)
        if len(self.performance_history[expert_type]) > 100:
            self.performance_history[expert_type] = self.performance_history[expert_type][-100:]
    
    def get_expert_reliability(self, expert_type: str) -> float:
        """Get reliability score for an expert based on historical performance"""
        if expert_type not in self.performance_history:
            return 0.5  # Default neutral reliability
        
        history = self.performance_history[expert_type]
        if not history:
            return 0.5
        
        # Weight recent performance more heavily
        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights = weights / weights.sum()
        
        return np.average(history, weights=weights)
    
    def aggregate_predictions(self, 
                            predictions: List[ExpertPrediction],
                            task_context: Optional[Dict[str, Any]] = None) -> AggregatedPrediction:
        """Aggregate expert predictions with optional task context"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Adjust confidence scores based on expert reliability
        adjusted_predictions = []
        for pred in predictions:
            reliability = self.get_expert_reliability(pred.expert_type)
            adjusted_confidence = pred.confidence * reliability
            
            adjusted_pred = ExpertPrediction(
                expert_type=pred.expert_type,
                prediction=pred.prediction,
                confidence=adjusted_confidence,
                reasoning=pred.reasoning,
                processing_time=pred.processing_time,
                metadata={**pred.metadata, 'original_confidence': pred.confidence, 'reliability': reliability}
            )
            adjusted_predictions.append(adjusted_pred)
        
        # Apply aggregation strategy
        result = self.strategy.aggregate(adjusted_predictions)
        
        # Add task context to metadata
        if task_context:
            result.metadata['task_context'] = task_context
        
        return result
    
    def evaluate_prediction_quality(self, 
                                  prediction: AggregatedPrediction,
                                  ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate quality of aggregated prediction against ground truth"""
        metrics = {}
        
        # Exact match accuracy
        metrics['exact_match'] = float(np.array_equal(prediction.final_prediction, ground_truth))
        
        # Pixel-wise accuracy
        if prediction.final_prediction.shape == ground_truth.shape:
            metrics['pixel_accuracy'] = float(np.mean(prediction.final_prediction == ground_truth))
        else:
            metrics['pixel_accuracy'] = 0.0
        
        # Confidence calibration (how well confidence matches accuracy)
        metrics['confidence_error'] = abs(prediction.confidence - metrics['pixel_accuracy'])
        
        # Pattern match (considering ARC-specific patterns)
        metrics['pattern_match'] = self._compute_pattern_match(prediction.final_prediction, ground_truth)
        
        return metrics
    
    def _compute_pattern_match(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute pattern-based match score for ARC tasks"""
        if prediction.shape != ground_truth.shape:
            return 0.0
        
        # Color distribution match
        pred_colors = set(prediction.flatten())
        true_colors = set(ground_truth.flatten())
        color_match = len(pred_colors & true_colors) / len(pred_colors | true_colors)
        
        # Structural pattern match
        structural_match = np.mean(prediction == ground_truth)
        
        return 0.6 * structural_match + 0.4 * color_match
    
    def analyze_disagreement(self, predictions: List[ExpertPrediction]) -> Dict[str, Any]:
        """Analyze disagreement patterns among experts"""
        if len(predictions) < 2:
            return {'disagreement_level': 'none', 'analysis': 'Single prediction'}
        
        # Compute pairwise similarities
        similarities = []
        expert_pairs = []
        
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                pred1, pred2 = predictions[i], predictions[j]
                
                if pred1.prediction.shape == pred2.prediction.shape:
                    sim = np.mean(pred1.prediction == pred2.prediction)
                else:
                    sim = 0.0
                
                similarities.append(sim)
                expert_pairs.append((pred1.expert_type, pred2.expert_type))
        
        avg_similarity = np.mean(similarities)
        
        # Classify disagreement level
        if avg_similarity > 0.8:
            disagreement_level = 'low'
        elif avg_similarity > 0.5:
            disagreement_level = 'medium'
        else:
            disagreement_level = 'high'
        
        # Find most disagreeing experts
        min_sim_idx = np.argmin(similarities)
        most_disagreeing = expert_pairs[min_sim_idx]
        
        return {
            'disagreement_level': disagreement_level,
            'avg_similarity': float(avg_similarity),
            'similarities': similarities,
            'expert_pairs': expert_pairs,
            'most_disagreeing_pair': most_disagreeing,
            'min_similarity': float(similarities[min_sim_idx])
        }
    
    def save_state(self, filepath: str):
        """Save system state including performance history"""
        state = {
            'performance_history': self.performance_history,
            'strategy_type': type(self.strategy).__name__
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load system state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.performance_history = state.get('performance_history', {})

def create_demo_predictions() -> List[ExpertPrediction]:
    """Create demo predictions for testing aggregation"""
    # Example ARC-like predictions from different experts
    
    # Tiling expert prediction
    tiling_pred = ExpertPrediction(
        expert_type="tiling",
        prediction=np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
        confidence=0.85,
        reasoning="Pattern shows 2x2 tiling repeated across 4x4 grid",
        processing_time=0.12
    )
    
    # Symmetry expert prediction (slightly different)
    symmetry_pred = ExpertPrediction(
        expert_type="symmetry",
        prediction=np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
        confidence=0.78,
        reasoning="Detected horizontal and vertical symmetry in pattern",
        processing_time=0.09
    )
    
    # Mapping expert prediction (different interpretation)
    mapping_pred = ExpertPrediction(
        expert_type="mapping",
        prediction=np.array([[2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3]]),
        confidence=0.65,
        reasoning="Applied color mapping transformation: 1->2, 2->1, 3->4, 4->3",
        processing_time=0.15
    )
    
    return [tiling_pred, symmetry_pred, mapping_pred]

def main():
    """Demo of the multi-expert aggregation system"""
    print("🤖 Multi-Expert Aggregation System Demo")
    print("=" * 45)
    
    # Create demo predictions
    predictions = create_demo_predictions()
    
    print(f"\\n📊 Input: {len(predictions)} expert predictions")
    for pred in predictions:
        print(f"  - {pred.expert_type}: confidence={pred.confidence:.2f}")
    
    # Test different aggregation strategies
    strategies = [
        ("Weighted Voting", WeightedVotingAggregator()),
        ("Bayesian Averaging", BayesianModelAveraging()),
        ("Consensus", ConsensusAggregator())
    ]
    
    system = MultiExpertAggregationSystem()
    
    for strategy_name, strategy in strategies:
        print(f"\\n🔧 Testing {strategy_name} Strategy:")
        print("-" * 30)
        
        system.set_strategy(strategy)
        result = system.aggregate_predictions(predictions)
        
        print(f"Final prediction shape: {result.final_prediction.shape}")
        print(f"Aggregated confidence: {result.confidence:.3f}")
        print(f"Consensus score: {result.consensus_score:.3f}")
        print(f"Uncertainty score: {result.uncertainty_score:.3f}")
        print("Expert contributions:")
        for expert, contrib in result.expert_contributions.items():
            print(f"  - {expert}: {contrib:.3f}")
    
    # Test disagreement analysis
    print("\\n🔍 Disagreement Analysis:")
    print("-" * 25)
    disagreement = system.analyze_disagreement(predictions)
    print(f"Disagreement level: {disagreement['disagreement_level']}")
    print(f"Average similarity: {disagreement['avg_similarity']:.3f}")
    print(f"Most disagreeing pair: {disagreement['most_disagreeing_pair']}")
    
    print("\\n✅ Multi-Expert Aggregation System Ready!")
    print("   - Weighted voting with confidence and similarity")
    print("   - Bayesian model averaging")
    print("   - Consensus-based aggregation with fallbacks")
    print("   - Uncertainty quantification")
    print("   - Expert performance tracking")

if __name__ == "__main__":
    main()