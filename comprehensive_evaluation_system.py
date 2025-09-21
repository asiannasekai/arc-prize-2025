"""
Comprehensive Evaluation System for ARC-AGI Challenge
Provides detailed metrics, benchmarking, and analysis to measure system performance
against the target of beating 4% AI baseline toward 85% accuracy for $600K prize.
"""

import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import logging
from collections import defaultdict
import pickle

from arc_challenge_solver import ARCTaskSolver, load_arc_tasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskEvaluation:
    """Evaluation results for a single task"""
    task_id: str
    exact_match: bool
    pixel_accuracy: float
    pattern_similarity: float
    confidence: float
    processing_time: float
    expert_contributions: Dict[str, float]
    pattern_type: str = "unknown"
    difficulty_score: float = 0.0
    grid_size: Tuple[int, int] = (0, 0)
    num_colors: int = 0
    error_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    overall_accuracy: float
    pattern_type_accuracy: Dict[str, float]
    difficulty_analysis: Dict[str, Any]
    expert_performance: Dict[str, Any]
    confidence_calibration: Dict[str, float]
    processing_efficiency: Dict[str, float]
    error_patterns: List[Dict[str, Any]]
    benchmark_comparison: Dict[str, float]
    task_evaluations: List[TaskEvaluation] = field(default_factory=list)

class ARCEvaluationSystem:
    """Comprehensive evaluation system for ARC tasks"""
    
    def __init__(self, solver: ARCTaskSolver = None):
        self.solver = solver or ARCTaskSolver()
        self.evaluation_cache = {}
        
        # Initialize evaluation metrics
        self.metrics = {
            'exact_match_count': 0,
            'total_tasks': 0,
            'pixel_accuracy_sum': 0.0,
            'pattern_similarity_sum': 0.0,
            'confidence_sum': 0.0,
            'processing_time_sum': 0.0
        }
        
        # Pattern type classification
        self.pattern_classifiers = {
            'tiling': self._is_tiling_pattern,
            'symmetry': self._is_symmetry_pattern,
            'extraction': self._is_extraction_pattern,
            'mapping': self._is_mapping_pattern,
            'rule': self._is_rule_pattern
        }
        
    def evaluate_task(self, 
                     task_data: Dict[str, Any], 
                     ground_truth: np.ndarray = None) -> TaskEvaluation:
        """Evaluate a single ARC task"""
        task_id = task_data.get('task_id', 'unknown')
        
        # Check cache first
        cache_key = f"{task_id}_{hash(str(task_data))}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        start_time = time.time()
        
        try:
            # Solve the task
            solution = self.solver.solve_task(task_data)
            prediction = np.array(solution['output'])
            
            # Get ground truth if not provided
            if ground_truth is None:
                ground_truth = self._extract_ground_truth(task_data)
            
            # Compute metrics
            exact_match = np.array_equal(prediction, ground_truth) if ground_truth is not None else False
            pixel_accuracy = self._compute_pixel_accuracy(prediction, ground_truth)
            pattern_similarity = self._compute_pattern_similarity(prediction, ground_truth)
            
            # Analyze task characteristics
            pattern_type = self._classify_pattern_type(task_data)
            difficulty_score = self._compute_difficulty_score(task_data)
            grid_size = prediction.shape
            num_colors = len(np.unique(prediction))
            
            # Error analysis
            error_analysis = self._analyze_errors(prediction, ground_truth, task_data)
            
            processing_time = time.time() - start_time
            
            # Create evaluation result
            evaluation = TaskEvaluation(
                task_id=task_id,
                exact_match=exact_match,
                pixel_accuracy=pixel_accuracy,
                pattern_similarity=pattern_similarity,
                confidence=solution.get('confidence', 0.0),
                processing_time=processing_time,
                expert_contributions=solution.get('expert_contributions', {}),
                pattern_type=pattern_type,
                difficulty_score=difficulty_score,
                grid_size=grid_size,
                num_colors=num_colors,
                error_analysis=error_analysis
            )
            
            # Cache result
            self.evaluation_cache[cache_key] = evaluation
            
            # Update metrics
            self._update_metrics(evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating task {task_id}: {e}")
            return TaskEvaluation(
                task_id=task_id,
                exact_match=False,
                pixel_accuracy=0.0,
                pattern_similarity=0.0,
                confidence=0.0,
                processing_time=time.time() - start_time,
                expert_contributions={},
                error_analysis={'evaluation_error': str(e)}
            )
    
    def evaluate_dataset(self, 
                        tasks: List[Dict[str, Any]], 
                        max_tasks: int = None) -> EvaluationReport:
        """Evaluate multiple tasks and generate comprehensive report"""
        logger.info(f"Starting evaluation of {len(tasks)} tasks")
        
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        task_evaluations = []
        
        for i, task_data in enumerate(tasks):
            logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task_data.get('task_id', 'unknown')}")
            evaluation = self.evaluate_task(task_data)
            task_evaluations.append(evaluation)
        
        # Generate comprehensive report
        report = self._generate_evaluation_report(task_evaluations)
        report.task_evaluations = task_evaluations
        
        logger.info(f"Evaluation completed. Overall accuracy: {report.overall_accuracy:.3f}")
        return report
    
    def _extract_ground_truth(self, task_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract ground truth from task data"""
        try:
            if 'test' in task_data and task_data['test']:
                test_case = task_data['test'][0]
                if 'output' in test_case:
                    return np.array(test_case['output'])
            return None
        except Exception:
            return None
    
    def _compute_pixel_accuracy(self, prediction: np.ndarray, ground_truth: Optional[np.ndarray]) -> float:
        """Compute pixel-wise accuracy"""
        if ground_truth is None:
            return 0.0
        
        if prediction.shape != ground_truth.shape:
            return 0.0
        
        return float(np.mean(prediction == ground_truth))
    
    def _compute_pattern_similarity(self, prediction: np.ndarray, ground_truth: Optional[np.ndarray]) -> float:
        """Compute pattern-based similarity score"""
        if ground_truth is None:
            return 0.0
        
        if prediction.shape != ground_truth.shape:
            return 0.0
        
        # Color distribution similarity
        pred_colors = set(prediction.flatten())
        true_colors = set(ground_truth.flatten())
        color_similarity = len(pred_colors & true_colors) / len(pred_colors | true_colors) if pred_colors | true_colors else 0.0
        
        # Structural similarity
        structural_similarity = np.mean(prediction == ground_truth)
        
        return 0.6 * structural_similarity + 0.4 * color_similarity
    
    def _classify_pattern_type(self, task_data: Dict[str, Any]) -> str:
        """Classify the pattern type of a task"""
        # Use first training example for classification
        if 'train' in task_data and task_data['train']:
            example = task_data['train'][0]
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check each pattern type
            for pattern_type, classifier in self.pattern_classifiers.items():
                if classifier(input_grid, output_grid):
                    return pattern_type
        
        return 'unknown'
    
    def _is_tiling_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if pattern involves tiling"""
        # Look for repeating 2x2 patterns
        h, w = output_grid.shape
        if h >= 4 and w >= 4:
            top_left = output_grid[:2, :2]
            return (np.array_equal(top_left, output_grid[2:4, :2]) and
                    np.array_equal(top_left, output_grid[:2, 2:4]))
        return False
    
    def _is_symmetry_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if pattern involves symmetry"""
        return (np.array_equal(output_grid, np.fliplr(input_grid)) or
                np.array_equal(output_grid, np.flipud(input_grid)) or
                np.array_equal(output_grid, input_grid.T))
    
    def _is_extraction_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if pattern involves object extraction"""
        # Check if output focuses on specific objects/regions
        input_objects = len(np.unique(input_grid))
        output_objects = len(np.unique(output_grid))
        return output_objects < input_objects
    
    def _is_mapping_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if pattern involves color/shape mapping"""
        if input_grid.shape == output_grid.shape:
            # Check for consistent color mapping
            input_colors = set(input_grid.flatten())
            output_colors = set(output_grid.flatten())
            return len(input_colors) == len(output_colors) and input_colors != output_colors
        return False
    
    def _is_rule_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if pattern involves complex rules"""
        # Default to rule-based if no other pattern matches
        return True
    
    def _compute_difficulty_score(self, task_data: Dict[str, Any]) -> float:
        """Compute difficulty score for a task"""
        try:
            if 'train' in task_data and task_data['train']:
                example = task_data['train'][0]
                input_grid = np.array(example['input'])
                
                # Factors that increase difficulty
                grid_size = input_grid.size
                num_colors = len(np.unique(input_grid))
                color_entropy = -np.sum([(np.sum(input_grid == c) / input_grid.size) * 
                                       np.log2((np.sum(input_grid == c) / input_grid.size) + 1e-10) 
                                       for c in np.unique(input_grid)])
                
                # Normalize and combine factors
                size_factor = min(grid_size / 100, 1.0)  # Normalized to [0,1]
                color_factor = min(num_colors / 10, 1.0)
                entropy_factor = min(color_entropy / 5, 1.0)
                
                return (size_factor + color_factor + entropy_factor) / 3
            
            return 0.5  # Default medium difficulty
            
        except Exception:
            return 0.5
    
    def _analyze_errors(self, 
                       prediction: np.ndarray, 
                       ground_truth: Optional[np.ndarray],
                       task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction errors"""
        if ground_truth is None:
            return {'no_ground_truth': True}
        
        analysis = {}
        
        # Shape mismatch
        if prediction.shape != ground_truth.shape:
            analysis['shape_mismatch'] = {
                'predicted_shape': prediction.shape,
                'expected_shape': ground_truth.shape
            }
        
        # Color errors
        pred_colors = set(prediction.flatten())
        true_colors = set(ground_truth.flatten())
        analysis['color_analysis'] = {
            'predicted_colors': sorted(list(pred_colors)),
            'expected_colors': sorted(list(true_colors)),
            'missing_colors': sorted(list(true_colors - pred_colors)),
            'extra_colors': sorted(list(pred_colors - true_colors))
        }
        
        # Spatial errors (if same shape)
        if prediction.shape == ground_truth.shape:
            diff_mask = prediction != ground_truth
            analysis['spatial_errors'] = {
                'error_rate': float(np.mean(diff_mask)),
                'error_locations': np.where(diff_mask),
                'clustered_errors': self._find_error_clusters(diff_mask)
            }
        
        return analysis
    
    def _find_error_clusters(self, diff_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Find clusters of errors in the prediction"""
        # Simple clustering based on connected components
        try:
            from scipy.ndimage import label
            labeled_errors, num_clusters = label(diff_mask)
            
            clusters = []
            for i in range(1, num_clusters + 1):
                cluster_mask = labeled_errors == i
                cluster_size = np.sum(cluster_mask)
                cluster_coords = np.where(cluster_mask)
                
                clusters.append({
                    'size': int(cluster_size),
                    'center': (float(np.mean(cluster_coords[0])), float(np.mean(cluster_coords[1]))),
                    'bounds': {
                        'min_row': int(np.min(cluster_coords[0])),
                        'max_row': int(np.max(cluster_coords[0])),
                        'min_col': int(np.min(cluster_coords[1])),
                        'max_col': int(np.max(cluster_coords[1]))
                    }
                })
            
            return clusters
            
        except ImportError:
            return []
    
    def _update_metrics(self, evaluation: TaskEvaluation):
        """Update running metrics"""
        self.metrics['total_tasks'] += 1
        
        if evaluation.exact_match:
            self.metrics['exact_match_count'] += 1
        
        self.metrics['pixel_accuracy_sum'] += evaluation.pixel_accuracy
        self.metrics['pattern_similarity_sum'] += evaluation.pattern_similarity
        self.metrics['confidence_sum'] += evaluation.confidence
        self.metrics['processing_time_sum'] += evaluation.processing_time
    
    def _generate_evaluation_report(self, evaluations: List[TaskEvaluation]) -> EvaluationReport:
        """Generate comprehensive evaluation report"""
        if not evaluations:
            return EvaluationReport(
                overall_accuracy=0.0,
                pattern_type_accuracy={},
                difficulty_analysis={},
                expert_performance={},
                confidence_calibration={},
                processing_efficiency={},
                error_patterns=[],
                benchmark_comparison={}
            )
        
        # Overall accuracy
        exact_matches = sum(1 for e in evaluations if e.exact_match)
        overall_accuracy = exact_matches / len(evaluations)
        
        # Pattern type analysis
        pattern_type_accuracy = {}
        pattern_groups = defaultdict(list)
        
        for eval_result in evaluations:
            pattern_groups[eval_result.pattern_type].append(eval_result)
        
        for pattern_type, group_evals in pattern_groups.items():
            group_exact_matches = sum(1 for e in group_evals if e.exact_match)
            pattern_type_accuracy[pattern_type] = group_exact_matches / len(group_evals)
        
        # Difficulty analysis
        difficulty_bins = {'easy': [], 'medium': [], 'hard': []}
        for eval_result in evaluations:
            if eval_result.difficulty_score < 0.33:
                difficulty_bins['easy'].append(eval_result)
            elif eval_result.difficulty_score < 0.67:
                difficulty_bins['medium'].append(eval_result)
            else:
                difficulty_bins['hard'].append(eval_result)
        
        difficulty_analysis = {}
        for difficulty, group_evals in difficulty_bins.items():
            if group_evals:
                group_exact_matches = sum(1 for e in group_evals if e.exact_match)
                difficulty_analysis[difficulty] = {
                    'accuracy': group_exact_matches / len(group_evals),
                    'count': len(group_evals),
                    'avg_confidence': np.mean([e.confidence for e in group_evals]),
                    'avg_processing_time': np.mean([e.processing_time for e in group_evals])
                }
        
        # Expert performance analysis
        expert_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'contribution_sum': 0.0})
        
        for eval_result in evaluations:
            for expert, contribution in eval_result.expert_contributions.items():
                expert_stats[expert]['total'] += 1
                expert_stats[expert]['contribution_sum'] += contribution
                if eval_result.exact_match:
                    expert_stats[expert]['correct'] += 1
        
        expert_performance = {}
        for expert, stats in expert_stats.items():
            if stats['total'] > 0:
                expert_performance[expert] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'usage_count': stats['total'],
                    'avg_contribution': stats['contribution_sum'] / stats['total']
                }
        
        # Confidence calibration
        confidence_bins = defaultdict(list)
        for eval_result in evaluations:
            conf_bin = int(eval_result.confidence * 10) / 10  # Round to nearest 0.1
            confidence_bins[conf_bin].append(eval_result.exact_match)
        
        confidence_calibration = {}
        for conf_level, matches in confidence_bins.items():
            if matches:
                confidence_calibration[f"{conf_level:.1f}"] = {
                    'predicted_accuracy': conf_level,
                    'actual_accuracy': np.mean(matches),
                    'count': len(matches),
                    'calibration_error': abs(conf_level - np.mean(matches))
                }
        
        # Processing efficiency
        processing_efficiency = {
            'avg_time_per_task': np.mean([e.processing_time for e in evaluations]),
            'median_time_per_task': np.median([e.processing_time for e in evaluations]),
            'std_time_per_task': np.std([e.processing_time for e in evaluations]),
            'tasks_per_second': 1.0 / np.mean([e.processing_time for e in evaluations])
        }
        
        # Error pattern analysis
        error_patterns = self._analyze_error_patterns(evaluations)
        
        # Benchmark comparison
        benchmark_comparison = {
            'arc_agi_baseline': 0.04,  # Current AI baseline
            'human_performance': 1.0,  # Human benchmark
            'our_performance': overall_accuracy,
            'improvement_over_baseline': (overall_accuracy - 0.04) / 0.04 if overall_accuracy > 0.04 else 0.0,
            'distance_to_prize_target': 0.85 - overall_accuracy,  # Distance to $600K prize threshold
            'distance_to_human': 1.0 - overall_accuracy
        }
        
        return EvaluationReport(
            overall_accuracy=overall_accuracy,
            pattern_type_accuracy=pattern_type_accuracy,
            difficulty_analysis=difficulty_analysis,
            expert_performance=expert_performance,
            confidence_calibration=confidence_calibration,
            processing_efficiency=processing_efficiency,
            error_patterns=error_patterns,
            benchmark_comparison=benchmark_comparison
        )
    
    def _analyze_error_patterns(self, evaluations: List[TaskEvaluation]) -> List[Dict[str, Any]]:
        """Analyze common error patterns across evaluations"""
        error_patterns = []
        
        # Common failure modes
        shape_mismatches = [e for e in evaluations if 'shape_mismatch' in e.error_analysis]
        if shape_mismatches:
            error_patterns.append({
                'type': 'shape_mismatch',
                'frequency': len(shape_mismatches) / len(evaluations),
                'description': 'Predictions have incorrect output dimensions',
                'examples': [e.task_id for e in shape_mismatches[:5]]
            })
        
        # Color prediction errors
        color_errors = [e for e in evaluations 
                       if 'color_analysis' in e.error_analysis and 
                       (e.error_analysis['color_analysis'].get('missing_colors') or
                        e.error_analysis['color_analysis'].get('extra_colors'))]
        if color_errors:
            error_patterns.append({
                'type': 'color_prediction_error',
                'frequency': len(color_errors) / len(evaluations),
                'description': 'Incorrect color palette in predictions',
                'examples': [e.task_id for e in color_errors[:5]]
            })
        
        # Low confidence failures
        low_conf_failures = [e for e in evaluations if not e.exact_match and e.confidence < 0.3]
        if low_conf_failures:
            error_patterns.append({
                'type': 'low_confidence_failure',
                'frequency': len(low_conf_failures) / len(evaluations),
                'description': 'Failed predictions with low system confidence',
                'examples': [e.task_id for e in low_conf_failures[:5]]
            })
        
        return error_patterns
    
    def save_evaluation_report(self, report: EvaluationReport, filepath: str):
        """Save evaluation report to file"""
        report_data = {
            'overall_accuracy': report.overall_accuracy,
            'pattern_type_accuracy': report.pattern_type_accuracy,
            'difficulty_analysis': report.difficulty_analysis,
            'expert_performance': report.expert_performance,
            'confidence_calibration': report.confidence_calibration,
            'processing_efficiency': report.processing_efficiency,
            'error_patterns': report.error_patterns,
            'benchmark_comparison': report.benchmark_comparison,
            'evaluation_metadata': {
                'total_tasks_evaluated': len(report.task_evaluations),
                'evaluation_timestamp': time.time()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Evaluation report saved to {filepath}")
    
    def generate_visualizations(self, report: EvaluationReport, output_dir: str = "./evaluation_plots"):
        """Generate visualization plots for evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Overall Performance vs Benchmarks
        fig, ax = plt.subplots(figsize=(10, 6))
        
        benchmarks = ['AI Baseline\\n(4%)', 'Our System', 'Prize Target\\n(85%)', 'Human\\n(100%)']
        accuracies = [0.04, report.overall_accuracy, 0.85, 1.0]
        colors = ['red', 'blue', 'orange', 'green']
        
        bars = ax.bar(benchmarks, accuracies, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('ARC-AGI Performance Comparison')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Pattern Type Accuracy
        if report.pattern_type_accuracy:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            pattern_types = list(report.pattern_type_accuracy.keys())
            accuracies = list(report.pattern_type_accuracy.values())
            
            bars = ax.bar(pattern_types, accuracies, color='skyblue', alpha=0.7)
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Pattern Type')
            ax.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, accuracy in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{accuracy:.2f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'pattern_type_accuracy.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Confidence Calibration
        if report.confidence_calibration:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            conf_levels = [float(k) for k in report.confidence_calibration.keys()]
            predicted_acc = [report.confidence_calibration[k]['predicted_accuracy'] for k in report.confidence_calibration.keys()]
            actual_acc = [report.confidence_calibration[k]['actual_accuracy'] for k in report.confidence_calibration.keys()]
            
            ax.plot(conf_levels, predicted_acc, 'b-', label='Predicted Accuracy', marker='o')
            ax.plot(conf_levels, actual_acc, 'r-', label='Actual Accuracy', marker='s')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Accuracy')
            ax.set_title('Confidence Calibration Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'confidence_calibration.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualization plots saved to {output_path}")

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on ARC tasks"""
    print("🧪 ARC-AGI Comprehensive Evaluation System")
    print("=" * 45)
    
    # Initialize evaluation system
    solver = ARCTaskSolver()
    evaluator = ARCEvaluationSystem(solver)
    
    try:
        # Load evaluation tasks
        eval_tasks = load_arc_tasks('arc-agi_evaluation_challenges.json')
        print(f"\\n📊 Loaded {len(eval_tasks)} evaluation tasks")
        
        # Run evaluation on subset for demo (first 10 tasks)
        demo_tasks = eval_tasks[:10]
        
        print(f"\\n🔄 Running comprehensive evaluation on {len(demo_tasks)} tasks...")
        
        # Perform evaluation
        report = evaluator.evaluate_dataset(demo_tasks)
        
        # Display results
        print("\\n✅ EVALUATION RESULTS:")
        print("=" * 25)
        
        print(f"\\n📈 Overall Performance:")
        print(f"  Exact Match Accuracy: {report.overall_accuracy:.1%}")
        print(f"  Tasks Evaluated: {len(report.task_evaluations)}")
        
        print(f"\\n🎯 Benchmark Comparison:")
        bc = report.benchmark_comparison
        print(f"  AI Baseline (4%): {bc['arc_agi_baseline']:.1%}")
        print(f"  Our System: {bc['our_performance']:.1%}")
        print(f"  Prize Target (85%): 85.0%")
        print(f"  Human Performance: {bc['human_performance']:.1%}")
        
        if bc['our_performance'] > bc['arc_agi_baseline']:
            improvement = bc['improvement_over_baseline']
            print(f"  🚀 Improvement over baseline: {improvement:.1%}")
        
        distance_to_prize = bc['distance_to_prize_target']
        print(f"  📏 Distance to prize target: {distance_to_prize:.1%}")
        
        print(f"\\n🔍 Pattern Type Analysis:")
        for pattern_type, accuracy in report.pattern_type_accuracy.items():
            print(f"  {pattern_type}: {accuracy:.1%}")
        
        print(f"\\n🤖 Expert Performance:")
        for expert, stats in report.expert_performance.items():
            print(f"  {expert}: {stats['accuracy']:.1%} accuracy, {stats['usage_count']} uses")
        
        print(f"\\n⚡ Processing Efficiency:")
        pe = report.processing_efficiency
        print(f"  Average time per task: {pe['avg_time_per_task']:.3f}s")
        print(f"  Tasks per second: {pe['tasks_per_second']:.1f}")
        
        print(f"\\n🎯 Confidence Analysis:")
        for conf_level, stats in report.confidence_calibration.items():
            print(f"  Confidence {conf_level}: {stats['actual_accuracy']:.1%} accuracy "
                  f"(predicted {stats['predicted_accuracy']:.1%})")
        
        print(f"\\n⚠️  Error Pattern Analysis:")
        for error_pattern in report.error_patterns:
            print(f"  {error_pattern['type']}: {error_pattern['frequency']:.1%} of tasks")
            print(f"    {error_pattern['description']}")
        
        # Save detailed report
        evaluator.save_evaluation_report(report, 'comprehensive_evaluation_report.json')
        
        # Generate visualizations
        evaluator.generate_visualizations(report)
        
        print(f"\\n💾 Detailed Results:")
        print(f"  📄 Report saved: comprehensive_evaluation_report.json")
        print(f"  📊 Plots saved: ./evaluation_plots/")
        
        # Summary for ARC-AGI Challenge
        print(f"\\n🏆 ARC-AGI Challenge Summary:")
        print(f"  Current Status: {report.overall_accuracy:.1%} accuracy")
        
        if report.overall_accuracy >= 0.85:
            print(f"  🎉 PRIZE THRESHOLD ACHIEVED! Eligible for $600K prize!")
        elif report.overall_accuracy > 0.04:
            print(f"  ✅ Beating AI baseline by {(report.overall_accuracy - 0.04) / 0.04:.1%}")
            print(f"  🎯 Need {distance_to_prize:.1%} more to reach prize threshold")
        else:
            print(f"  📈 Working to beat 4% AI baseline")
        
    except FileNotFoundError:
        print("\\n⚠️  Evaluation dataset not found. Running synthetic evaluation...")
        
        # Create synthetic tasks for demonstration
        synthetic_tasks = []
        for i in range(5):
            task = {
                'task_id': f'synthetic_{i:03d}',
                'train': [
                    {
                        'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                        'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
                    }
                ],
                'test': [
                    {
                        'input': [[2, 0, 2], [0, 2, 0], [2, 0, 2]],
                        'output': [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
                    }
                ]
            }
            synthetic_tasks.append(task)
        
        report = evaluator.evaluate_dataset(synthetic_tasks)
        print(f"\\n✅ Synthetic Evaluation Results:")
        print(f"  Accuracy: {report.overall_accuracy:.1%}")
        print(f"  Tasks: {len(synthetic_tasks)}")
    
    finally:
        solver.close()
    
    print("\\n🧪 Evaluation System Capabilities Demonstrated:")
    print("   ✓ Comprehensive accuracy metrics")
    print("   ✓ Pattern type analysis")
    print("   ✓ Difficulty-based performance breakdown")
    print("   ✓ Expert contribution analysis")
    print("   ✓ Confidence calibration assessment")
    print("   ✓ Error pattern identification")
    print("   ✓ Benchmark comparison with ARC-AGI challenge")
    print("   ✓ Performance visualization")
    print("   ✓ Detailed reporting and caching")

if __name__ == "__main__":
    run_comprehensive_evaluation()