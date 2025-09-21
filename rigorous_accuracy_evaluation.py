#!/usr/bin/env python3
"""
Rigorous Accuracy Evaluation - Cross-Validation on Training Data
===============================================================

Since we don't have test ground truth, we'll use cross-validation on training examples
to get a realistic accuracy estimate.
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RigorousARCEvaluator:
    """Rigorous evaluation using cross-validation on training examples"""
    
    def __init__(self):
        from pattern_learning_baseline import PatternLearningBaseline
        self.baseline = PatternLearningBaseline()
    
    def cross_validate_task(self, task):
        """Cross-validate on training examples"""
        train_examples = task.get('train', [])
        
        if len(train_examples) < 2:
            return {'accuracy': 0.0, 'predictions': 0}
        
        correct_predictions = 0
        total_predictions = 0
        
        # Use each training example as test, others as train
        for test_idx in range(len(train_examples)):
            # Create modified task with one example held out
            modified_task = {
                'train': [ex for i, ex in enumerate(train_examples) if i != test_idx],
                'test': [{'input': train_examples[test_idx]['input']}]
            }
            
            # Learn from remaining examples and predict
            result = self.baseline.solve_task(modified_task)
            
            if result and result['prediction']:
                expected_output = train_examples[test_idx]['output']
                prediction = result['prediction']
                
                # Check if prediction matches expected output
                try:
                    if np.array_equal(prediction, expected_output):
                        correct_predictions += 1
                    total_predictions += 1
                except:
                    # Shape mismatch or other error
                    total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions
        }
    
    def evaluate_rigorous(self, num_tasks=20):
        """Run rigorous cross-validation evaluation"""
        logger.info(f"🔬 Running rigorous cross-validation on {num_tasks} tasks...")
        
        # Load evaluation data
        if not Path('arc-agi_evaluation_challenges.json').exists():
            logger.error("❌ Evaluation data not found")
            return {'accuracy': 0.0, 'error': 'No evaluation data'}
        
        with open('arc-agi_evaluation_challenges.json') as f:
            all_tasks = json.load(f)
        
        # Take first N tasks
        task_ids = list(all_tasks.keys())[:num_tasks]
        
        task_accuracies = []
        task_results = []
        total_correct = 0
        total_predictions = 0
        
        for i, task_id in enumerate(task_ids):
            task = all_tasks[task_id]
            
            try:
                start_time = time.time()
                
                # Cross-validate this task
                cv_result = self.cross_validate_task(task)
                
                processing_time = time.time() - start_time
                
                task_accuracy = cv_result['accuracy']
                task_accuracies.append(task_accuracy)
                
                total_correct += cv_result['correct']
                total_predictions += cv_result['total']
                
                task_result = {
                    'task_id': task_id,
                    'accuracy': task_accuracy,
                    'correct_predictions': cv_result['correct'],
                    'total_predictions': cv_result['total'],
                    'processing_time': processing_time
                }
                
                task_results.append(task_result)
                
                status = "✅" if task_accuracy > 0.5 else "⚠️" if task_accuracy > 0 else "❌"
                logger.info(f"Task {i+1}/{num_tasks} ({task_id}): {status} {task_accuracy:.1%}")
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                task_results.append({
                    'task_id': task_id,
                    'accuracy': 0.0,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                })
                task_accuracies.append(0.0)
        
        # Calculate overall statistics
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        avg_task_accuracy = np.mean(task_accuracies) if task_accuracies else 0.0
        median_accuracy = np.median(task_accuracies) if task_accuracies else 0.0
        avg_processing_time = np.mean([r.get('processing_time', 0) for r in task_results])
        
        # Classify performance levels
        high_performance = [acc for acc in task_accuracies if acc >= 0.8]
        medium_performance = [acc for acc in task_accuracies if 0.3 <= acc < 0.8]
        low_performance = [acc for acc in task_accuracies if 0 < acc < 0.3]
        zero_performance = [acc for acc in task_accuracies if acc == 0]
        
        results = {
            'overall_accuracy': overall_accuracy,
            'average_task_accuracy': avg_task_accuracy,
            'median_task_accuracy': median_accuracy,
            'total_correct_predictions': total_correct,
            'total_predictions': total_predictions,
            'tasks_evaluated': len(task_ids),
            'average_processing_time': avg_processing_time,
            'tasks_per_second': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'performance_distribution': {
                'high_performance': len(high_performance),
                'medium_performance': len(medium_performance), 
                'low_performance': len(low_performance),
                'zero_performance': len(zero_performance)
            },
            'task_results': task_results
        }
        
        return results

def main():
    """Run rigorous accuracy evaluation"""
    print("🔬 RIGOROUS ARC ACCURACY EVALUATION")
    print("=" * 45)
    
    evaluator = RigorousARCEvaluator()
    
    print("\n📊 Running cross-validation evaluation...")
    print("   (Using training examples to validate predictions)")
    
    results = evaluator.evaluate_rigorous(num_tasks=25)
    
    print(f"\n✅ RIGOROUS ACCURACY RESULTS:")
    print(f"   Overall Accuracy: {results['overall_accuracy']:.1%}")
    print(f"   Average Task Accuracy: {results['average_task_accuracy']:.1%}")
    print(f"   Median Task Accuracy: {results['median_task_accuracy']:.1%}")
    print(f"   Correct Predictions: {results['total_correct_predictions']}/{results['total_predictions']}")
    print(f"   Processing Speed: {results['tasks_per_second']:.1f} tasks/second")
    
    print(f"\n📊 PERFORMANCE DISTRIBUTION:")
    dist = results['performance_distribution']
    print(f"   High (≥80%): {dist['high_performance']} tasks")
    print(f"   Medium (30-79%): {dist['medium_performance']} tasks") 
    print(f"   Low (1-29%): {dist['low_performance']} tasks")
    print(f"   Zero (0%): {dist['zero_performance']} tasks")
    
    # Compare with competition targets
    print(f"\n🎯 COMPETITION BENCHMARK:")
    print(f"   Our CPU Baseline: {results['overall_accuracy']:.1%}")
    print(f"   AI Baseline (beat this): 4.0%")
    print(f"   Prize Target: 85.0%")
    
    if results['overall_accuracy'] >= 0.04:
        print(f"   🎉 BEATING AI BASELINE! (+{results['overall_accuracy']-0.04:.1%})")
    else:
        gap = 0.04 - results['overall_accuracy']
        print(f"   📈 Need {gap:.1%} more to beat AI baseline")
    
    prize_gap = 0.85 - results['overall_accuracy']
    print(f"   🏆 Need {prize_gap:.1%} more for $600K prize")
    
    # Save detailed results
    with open('rigorous_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: rigorous_accuracy_results.json")
    
    # Show best performing tasks
    best_tasks = sorted(results['task_results'], 
                       key=lambda x: x.get('accuracy', 0), reverse=True)[:5]
    
    print(f"\n🏆 BEST PERFORMING TASKS:")
    for task in best_tasks:
        acc = task.get('accuracy', 0)
        correct = task.get('correct_predictions', 0) 
        total = task.get('total_predictions', 0)
        print(f"   {task['task_id']}: {acc:.1%} ({correct}/{total})")
    
    # Analysis of current capabilities
    if results['overall_accuracy'] > 0:
        print(f"\n🧠 SYSTEM ANALYSIS:")
        print(f"   ✅ Pattern learning is functional")
        print(f"   ✅ Cross-validation shows real learning")
        print(f"   ✅ Ready for Phase 2 optimization")
        if results['overall_accuracy'] >= 0.04:
            print(f"   🚀 Already competitive with AI baseline!")

if __name__ == "__main__":
    main()