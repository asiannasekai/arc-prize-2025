"""
Integration Example: Multi-Expert Aggregation with Federated ARC System
Demonstrates how the aggregation system integrates with router, experts, and knowledge graph.

This addresses the ARC-AGI challenge of generalizing to novel problems by:
1. Routing tasks to appropriate expert specialists
2. Getting multiple expert predictions with confidence scores
3. Aggregating predictions using sophisticated consensus algorithms
4. Leveraging uncertainty quantification for better decision making
"""

import numpy as np
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

# Import our components
from multi_expert_aggregation import (
    ExpertPrediction, AggregatedPrediction, 
    MultiExpertAggregationSystem, WeightedVotingAggregator,
    BayesianModelAveraging, ConsensusAggregator
)

logger = logging.getLogger(__name__)

@dataclass
class ARCTask:
    """Represents an ARC task with input/output examples"""
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_input: np.ndarray
    test_output: np.ndarray = None  # For evaluation
    task_id: str = ""
    metadata: Dict[str, Any] = None

class MockExpertModel:
    """Mock expert model for demonstration purposes"""
    
    def __init__(self, expert_type: str, base_confidence: float = 0.7):
        self.expert_type = expert_type
        self.base_confidence = base_confidence
        
    def predict(self, task: ARCTask) -> ExpertPrediction:
        """Generate a mock prediction for demonstration"""
        import time
        start_time = time.time()
        
        # Mock prediction based on expert type
        input_shape = task.test_input.shape
        
        if self.expert_type == "tiling":
            # Tiling expert: repeat patterns
            prediction = self._generate_tiling_prediction(task)
            reasoning = "Applied tiling pattern based on training examples"
            confidence = self.base_confidence + 0.1
            
        elif self.expert_type == "symmetry":
            # Symmetry expert: apply symmetry transformations
            prediction = self._generate_symmetry_prediction(task)
            reasoning = "Applied symmetry transformation (reflection/rotation)"
            confidence = self.base_confidence
            
        elif self.expert_type == "extraction":
            # Extraction expert: extract and transform objects
            prediction = self._generate_extraction_prediction(task)
            reasoning = "Extracted objects and applied transformation rules"
            confidence = self.base_confidence - 0.05
            
        elif self.expert_type == "mapping":
            # Mapping expert: color/shape mappings
            prediction = self._generate_mapping_prediction(task)
            reasoning = "Applied color/shape mapping transformation"
            confidence = self.base_confidence - 0.1
            
        else:  # rule
            # Rule expert: complex logical rules
            prediction = self._generate_rule_prediction(task)
            reasoning = "Applied complex logical rule based on pattern analysis"
            confidence = self.base_confidence - 0.15
        
        processing_time = time.time() - start_time
        
        return ExpertPrediction(
            expert_type=self.expert_type,
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={'input_shape': input_shape, 'output_shape': prediction.shape}
        )
    
    def _generate_tiling_prediction(self, task: ARCTask) -> np.ndarray:
        """Generate tiling-based prediction"""
        # Simple tiling: repeat 2x2 pattern
        base_pattern = np.array([[1, 2], [3, 4]])
        output_shape = task.test_input.shape
        
        prediction = np.zeros(output_shape, dtype=int)
        for i in range(0, output_shape[0], 2):
            for j in range(0, output_shape[1], 2):
                end_i = min(i + 2, output_shape[0])
                end_j = min(j + 2, output_shape[1])
                pattern_slice = base_pattern[:end_i-i, :end_j-j]
                prediction[i:end_i, j:end_j] = pattern_slice
        
        return prediction
    
    def _generate_symmetry_prediction(self, task: ARCTask) -> np.ndarray:
        """Generate symmetry-based prediction"""
        # Apply horizontal flip
        return np.fliplr(task.test_input)
    
    def _generate_extraction_prediction(self, task: ARCTask) -> np.ndarray:
        """Generate extraction-based prediction"""
        # Extract unique elements and rearrange
        unique_elements = np.unique(task.test_input)
        output = task.test_input.copy()
        
        # Simple transformation: increment all values
        for i, elem in enumerate(unique_elements):
            output[output == elem] = (elem + 1) % 10
        
        return output
    
    def _generate_mapping_prediction(self, task: ARCTask) -> np.ndarray:
        """Generate mapping-based prediction"""
        # Color mapping transformation
        color_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
        output = task.test_input.copy()
        
        for old_color, new_color in color_map.items():
            output[task.test_input == old_color] = new_color
        
        return output
    
    def _generate_rule_prediction(self, task: ARCTask) -> np.ndarray:
        """Generate rule-based prediction"""
        # Complex rule: rotate and modify
        rotated = np.rot90(task.test_input)
        return (rotated + 1) % 5

class MockRouter:
    """Mock router for selecting relevant experts"""
    
    def __init__(self):
        self.expert_types = ["tiling", "symmetry", "extraction", "mapping", "rule"]
    
    def select_experts(self, task: ARCTask, max_experts: int = 3) -> List[str]:
        """Select most relevant experts for the task"""
        # Simple mock selection based on task characteristics
        input_shape = task.test_input.shape
        unique_colors = len(np.unique(task.test_input))
        
        selected = []
        
        # Always include tiling for structured patterns
        if input_shape[0] % 2 == 0 and input_shape[1] % 2 == 0:
            selected.append("tiling")
        
        # Include symmetry for square grids
        if input_shape[0] == input_shape[1]:
            selected.append("symmetry")
        
        # Include mapping if many colors
        if unique_colors > 3:
            selected.append("mapping")
        else:
            selected.append("extraction")
        
        # Limit to max_experts
        return selected[:max_experts]

class FederatedARCSystem:
    """Main federated ARC system integrating all components"""
    
    def __init__(self):
        # Initialize components
        self.router = MockRouter()
        self.experts = {
            expert_type: MockExpertModel(expert_type) 
            for expert_type in ["tiling", "symmetry", "extraction", "mapping", "rule"]
        }
        self.aggregation_system = MultiExpertAggregationSystem(
            aggregation_strategy=WeightedVotingAggregator()
        )
        
        # Performance tracking
        self.task_history = []
        self.performance_metrics = {
            'total_tasks': 0,
            'successful_predictions': 0,
            'avg_confidence': 0.0,
            'expert_usage': {expert: 0 for expert in self.experts.keys()}
        }
    
    def solve_task(self, task: ARCTask, strategy: str = "weighted_voting") -> AggregatedPrediction:
        """Solve an ARC task using the federated system"""
        logger.info(f"Solving task {task.task_id}")
        
        # 1. Router selects relevant experts
        selected_experts = self.router.select_experts(task)
        logger.info(f"Router selected experts: {selected_experts}")
        
        # 2. Get predictions from selected experts
        expert_predictions = []
        for expert_type in selected_experts:
            if expert_type in self.experts:
                prediction = self.experts[expert_type].predict(task)
                expert_predictions.append(prediction)
                self.performance_metrics['expert_usage'][expert_type] += 1
        
        # 3. Set aggregation strategy
        if strategy == "bayesian":
            self.aggregation_system.set_strategy(BayesianModelAveraging())
        elif strategy == "consensus":
            self.aggregation_system.set_strategy(ConsensusAggregator())
        else:  # weighted_voting
            self.aggregation_system.set_strategy(WeightedVotingAggregator())
        
        # 4. Aggregate predictions
        aggregated = self.aggregation_system.aggregate_predictions(
            expert_predictions,
            task_context={'task_id': task.task_id, 'input_shape': task.test_input.shape}
        )
        
        # 5. Update metrics
        self.performance_metrics['total_tasks'] += 1
        self.performance_metrics['avg_confidence'] = (
            (self.performance_metrics['avg_confidence'] * (self.performance_metrics['total_tasks'] - 1) +
             aggregated.confidence) / self.performance_metrics['total_tasks']
        )
        
        # Store task history
        self.task_history.append({
            'task_id': task.task_id,
            'selected_experts': selected_experts,
            'final_confidence': aggregated.confidence,
            'consensus_score': aggregated.consensus_score,
            'uncertainty_score': aggregated.uncertainty_score,
            'strategy_used': strategy
        })
        
        return aggregated
    
    def evaluate_prediction(self, prediction: AggregatedPrediction, ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate prediction against ground truth"""
        metrics = self.aggregation_system.evaluate_prediction_quality(prediction, ground_truth)
        
        # Update expert performance based on this result
        for expert_type, contribution in prediction.expert_contributions.items():
            if contribution > 0.1:  # Only update for significant contributors
                self.aggregation_system.update_expert_performance(
                    expert_type, metrics['exact_match']
                )
        
        if metrics['exact_match'] > 0.5:
            self.performance_metrics['successful_predictions'] += 1
        
        return metrics
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics"""
        success_rate = (self.performance_metrics['successful_predictions'] / 
                       max(self.performance_metrics['total_tasks'], 1))
        
        return {
            'total_tasks_processed': self.performance_metrics['total_tasks'],
            'success_rate': success_rate,
            'average_confidence': self.performance_metrics['avg_confidence'],
            'expert_usage_frequency': self.performance_metrics['expert_usage'],
            'recent_tasks': self.task_history[-5:] if self.task_history else []
        }

def create_demo_task() -> ARCTask:
    """Create a demo ARC task for testing"""
    # Create simple pattern recognition task
    train_input_1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    train_output_1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    train_input_2 = np.array([[2, 0, 2], [0, 2, 0], [2, 0, 2]])
    train_output_2 = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])
    
    test_input = np.array([[3, 0, 3], [0, 3, 0], [3, 0, 3]])
    test_output = np.array([[0, 3, 0], [3, 0, 3], [0, 3, 0]])  # Ground truth
    
    return ARCTask(
        train_examples=[(train_input_1, train_output_1), (train_input_2, train_output_2)],
        test_input=test_input,
        test_output=test_output,
        task_id="demo_task_001"
    )

def run_comprehensive_demo():
    """Run comprehensive demo of the federated ARC system"""
    print("🚀 Federated ARC System - Comprehensive Demo")
    print("=" * 50)
    
    # Initialize system
    system = FederatedARCSystem()
    task = create_demo_task()
    
    print("\\n📋 Demo Task Details:")
    print(f"Task ID: {task.task_id}")
    print(f"Test input shape: {task.test_input.shape}")
    print("Test input:")
    print(task.test_input)
    print("Expected output:")
    print(task.test_output)
    
    # Test different aggregation strategies
    strategies = ["weighted_voting", "bayesian", "consensus"]
    results = {}
    
    for strategy in strategies:
        print(f"\\n🔧 Testing {strategy.replace('_', ' ').title()} Strategy:")
        print("-" * 35)
        
        # Solve task
        prediction = system.solve_task(task, strategy=strategy)
        
        print(f"Final prediction:")
        print(prediction.final_prediction)
        print(f"Confidence: {prediction.confidence:.3f}")
        print(f"Consensus score: {prediction.consensus_score:.3f}")
        print(f"Uncertainty: {prediction.uncertainty_score:.3f}")
        
        # Evaluate against ground truth
        evaluation = system.evaluate_prediction(prediction, task.test_output)
        results[strategy] = evaluation
        
        print(f"Exact match: {evaluation['exact_match']}")
        print(f"Pixel accuracy: {evaluation['pixel_accuracy']:.3f}")
        print(f"Pattern match: {evaluation['pattern_match']:.3f}")
        
        print("Expert contributions:")
        for expert, contrib in prediction.expert_contributions.items():
            print(f"  - {expert}: {contrib:.3f}")
    
    # System statistics
    print("\\n📊 System Performance Summary:")
    print("-" * 30)
    stats = system.get_system_stats()
    print(f"Tasks processed: {stats['total_tasks_processed']}")
    print(f"Success rate: {stats['success_rate']:.3f}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    
    print("\\nExpert usage frequency:")
    for expert, count in stats['expert_usage_frequency'].items():
        print(f"  - {expert}: {count} times")
    
    # Best strategy analysis
    print("\\n🏆 Strategy Comparison:")
    print("-" * 25)
    for strategy, eval_metrics in results.items():
        print(f"{strategy}: accuracy={eval_metrics['pixel_accuracy']:.3f}, "
              f"pattern_match={eval_metrics['pattern_match']:.3f}")
    
    print("\\n✅ Federated ARC System Demo Complete!")
    print("   Key capabilities demonstrated:")
    print("   - Intelligent expert routing")
    print("   - Multi-expert prediction generation")
    print("   - Sophisticated aggregation with uncertainty")
    print("   - Performance tracking and adaptation")
    print("   - Comprehensive evaluation metrics")

if __name__ == "__main__":
    run_comprehensive_demo()