#!/usr/bin/env python3
"""
Quick Phase 1 Implementation - CPU Baseline Test
===============================================

Implements critical fixes and runs sample evaluation to get initial accuracy.
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickARCBaseline:
    """Quick implementation to get initial accuracy baseline"""
    
    def __init__(self):
        self.neo4j_driver = None
        self.setup_neo4j()
        
    def setup_neo4j(self):
        """Setup Neo4j with proper schema"""
        try:
            uri = "bolt://localhost:7687"
            user = "neo4j"
            password = "neo4jpassword"
            
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Test connection and setup basic patterns
            with self.neo4j_driver.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create basic patterns with correct schema
                patterns = [
                    {
                        'pattern_id': 'horizontal_flip',
                        'transformation_type': 'geometric',
                        'description': 'Horizontal flip of grid',
                        'grid_size': 3,
                        'num_colors': 2,
                        'complexity_score': 0.3,
                        'confidence_score': 0.8,
                        'frequency': 10
                    },
                    {
                        'pattern_id': 'vertical_flip',
                        'transformation_type': 'geometric', 
                        'description': 'Vertical flip of grid',
                        'grid_size': 3,
                        'num_colors': 2,
                        'complexity_score': 0.3,
                        'confidence_score': 0.8,
                        'frequency': 8
                    },
                    {
                        'pattern_id': 'color_swap',
                        'transformation_type': 'mapping',
                        'description': 'Swap two colors',
                        'grid_size': 3,
                        'num_colors': 2,
                        'complexity_score': 0.2,
                        'confidence_score': 0.9,
                        'frequency': 15
                    }
                ]
                
                for pattern in patterns:
                    session.run("""
                        CREATE (p:Pattern {
                            pattern_id: $pattern_id,
                            transformation_type: $transformation_type,
                            description: $description,
                            grid_size: $grid_size,
                            num_colors: $num_colors,
                            complexity_score: $complexity_score,
                            confidence_score: $confidence_score,
                            frequency: $frequency
                        })
                    """, **pattern)
                
                logger.info("✅ Neo4j schema fixed and patterns loaded")
                
        except Exception as e:
            logger.warning(f"Neo4j setup failed: {e}, will run without KG")
            self.neo4j_driver = None
    
    def simple_pattern_solver(self, task):
        """Simple pattern-based solver for baseline"""
        try:
            train_examples = task.get('train', [])
            test_input = task.get('test', [{}])[0].get('input', [])
            
            if not train_examples or not test_input:
                return []
            
            # Get first training example
            train_input = train_examples[0]['input']
            train_output = train_examples[0]['output']
            
            # Simple transformations to try
            predictions = []
            
            # Try horizontal flip
            h_flip = [row[::-1] for row in test_input]
            predictions.append(h_flip)
            
            # Try vertical flip  
            v_flip = test_input[::-1]
            predictions.append(v_flip)
            
            # Try identity (no change)
            predictions.append(test_input)
            
            # Try color mapping (0->1, 1->0 for binary)
            color_map = []
            for row in test_input:
                new_row = []
                for cell in row:
                    if cell == 0:
                        new_row.append(1)
                    elif cell == 1:
                        new_row.append(0)
                    else:
                        new_row.append(cell)
                color_map.append(new_row)
            predictions.append(color_map)
            
            # Return first prediction for simplicity
            return predictions[0] if predictions else test_input
            
        except Exception as e:
            logger.warning(f"Solver failed: {e}")
            return test_input  # Return input as fallback
    
    def evaluate_sample(self, num_tasks=5):
        """Run evaluation on small sample"""
        logger.info(f"🧪 Running baseline evaluation on {num_tasks} tasks...")
        
        # Load evaluation data
        if not Path('arc-agi_evaluation_challenges.json').exists():
            logger.error("❌ Evaluation data not found")
            return {'accuracy': 0.0, 'error': 'No evaluation data'}
        
        with open('arc-agi_evaluation_challenges.json') as f:
            all_tasks = json.load(f)
        
        # Take first N tasks for sample
        task_ids = list(all_tasks.keys())[:num_tasks]
        
        results = []
        correct = 0
        total = 0
        
        for task_id in task_ids:
            task = all_tasks[task_id]
            
            try:
                start_time = time.time()
                
                # Get prediction
                prediction = self.simple_pattern_solver(task)
                
                # Get expected output (if available)
                test_examples = task.get('test', [])
                if test_examples and 'output' in test_examples[0]:
                    expected = test_examples[0]['output']
                    
                    # Check if prediction matches
                    matches = (prediction == expected)
                    if isinstance(matches, bool):
                        is_correct = matches
                    else:
                        # For numpy arrays or lists
                        is_correct = np.array_equal(prediction, expected)
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    result = {
                        'task_id': task_id,
                        'correct': is_correct,
                        'prediction_shape': np.array(prediction).shape if prediction else None,
                        'expected_shape': np.array(expected).shape,
                        'processing_time': time.time() - start_time
                    }
                else:
                    # No ground truth available
                    result = {
                        'task_id': task_id,
                        'correct': None,
                        'prediction_shape': np.array(prediction).shape if prediction else None,
                        'processing_time': time.time() - start_time
                    }
                
                results.append(result)
                logger.info(f"Task {task_id}: {'✅' if result.get('correct') else '❌' if result.get('correct') is not None else '❓'}")
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                results.append({
                    'task_id': task_id,
                    'correct': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                })
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0
        avg_time = np.mean([r.get('processing_time', 0) for r in results])
        
        evaluation_summary = {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_evaluated': total,
            'average_processing_time': avg_time,
            'tasks_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'results': results
        }
        
        return evaluation_summary

def main():
    """Run CPU baseline evaluation"""
    print("🚀 ARC BASELINE EVALUATION ON CPU")
    print("=" * 40)
    
    # Initialize baseline solver
    baseline = QuickARCBaseline()
    
    # Run sample evaluation
    print("\n📊 Running sample evaluation...")
    results = baseline.evaluate_sample(num_tasks=10)
    
    print(f"\n✅ BASELINE RESULTS:")
    print(f"   Accuracy: {results['accuracy']:.1%}")
    print(f"   Correct: {results['correct_predictions']}/{results['total_evaluated']}")
    print(f"   Avg Time: {results['average_processing_time']:.3f}s per task")
    print(f"   Speed: {results['tasks_per_second']:.1f} tasks/second")
    
    # Compare with targets
    print(f"\n🎯 BENCHMARK COMPARISON:")
    print(f"   Our CPU Baseline: {results['accuracy']:.1%}")
    print(f"   AI Baseline (target): 4.0%") 
    print(f"   Prize Target: 85.0%")
    print(f"   Gap to Prize: {0.85 - results['accuracy']:.1%}")
    
    if results['accuracy'] > 0:
        print(f"\n🎉 SUCCESS: We have a working baseline!")
        print(f"   Ready for Phase 2 optimization")
    else:
        print(f"\n⚠️ BASELINE: Starting from 0% - common for initial implementation")
        print(f"   Phase 1 infrastructure improvements needed")
    
    # Save results
    with open('cpu_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: cpu_baseline_results.json")
    
    # Show detailed task results
    print(f"\n📋 Task Details:")
    for result in results['results'][:5]:  # Show first 5
        status = "✅" if result.get('correct') else "❌" if result.get('correct') is not None else "❓"
        print(f"   {result['task_id']}: {status} ({result.get('processing_time', 0):.3f}s)")
    
    if baseline.neo4j_driver:
        baseline.neo4j_driver.close()

if __name__ == "__main__":
    main()