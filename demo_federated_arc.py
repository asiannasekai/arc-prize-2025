"""
Demonstration of Federated ARC System
Run this to see how the federated approach would work on real ARC data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from federated_arc_system import FederatedARCSystem, ARCTask, PatternType
from training_config import create_expert_configs, SystemConfig

def visualize_arc_task(task: ARCTask, title: str = "ARC Task"):
    """Visualize an ARC task with training examples and test case"""
    
    num_examples = len(task.train_examples)
    fig, axes = plt.subplots(2, num_examples + 1, figsize=(15, 6))
    
    # Plot training examples
    for i, example in enumerate(task.train_examples):
        # Input
        axes[0, i].imshow(np.array(example['input']), cmap='tab10', vmin=0, vmax=9)
        axes[0, i].set_title(f'Train {i+1} Input')
        axes[0, i].axis('off')
        
        # Output
        axes[1, i].imshow(np.array(example['output']), cmap='tab10', vmin=0, vmax=9)
        axes[1, i].set_title(f'Train {i+1} Output')
        axes[1, i].axis('off')
    
    # Plot test case
    if task.test_examples:
        test_input = task.test_examples[0]['input']
        axes[0, num_examples].imshow(np.array(test_input), cmap='tab10', vmin=0, vmax=9)
        axes[0, num_examples].set_title('Test Input')
        axes[0, num_examples].axis('off')
        
        # Empty space for solution
        axes[1, num_examples].text(0.5, 0.5, 'Solution\n?', 
                                  ha='center', va='center', fontsize=16)
        axes[1, num_examples].set_title('Test Output')
        axes[1, num_examples].axis('off')
    
    plt.suptitle(f'{title} - {task.task_id}')
    plt.tight_layout()
    plt.savefig(f'arc_task_{task.task_id}.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_dataset_distribution():
    """Analyze the distribution of pattern types in the training set"""
    
    print("=== ANALYZING ARC DATASET DISTRIBUTION ===")
    
    # Load training data
    with open('arc-agi_training_challenges.json', 'r') as f:
        training_data = json.load(f)
    
    # Initialize system for classification
    system = FederatedARCSystem()
    
    # Classify all tasks
    pattern_counts = {pattern_type: 0 for pattern_type in PatternType}
    size_distributions = {'micro': 0, 'standard': 0, 'macro': 0}
    
    task_classifications = {}
    
    for task_id, task_data in list(training_data.items())[:100]:  # Analyze first 100 for speed
        task = ARCTask(
            task_id=task_id,
            train_examples=task_data['train'],
            test_examples=task_data['test']
        )
        
        # Classify pattern type
        pattern_type, confidence = system.router.classify_task(task)
        pattern_counts[pattern_type] += 1
        
        # Classify size
        features = task.get_pattern_features()
        grid_area = features.get('grid_area', 1)
        
        if grid_area <= 25:
            size_distributions['micro'] += 1
        elif grid_area <= 225:
            size_distributions['standard'] += 1
        else:
            size_distributions['macro'] += 1
            
        task_classifications[task_id] = {
            'pattern_type': pattern_type.value,
            'confidence': confidence,
            'features': features
        }
    
    # Print results
    print("\\nPattern Type Distribution:")
    for pattern_type, count in pattern_counts.items():
        percentage = (count / 100) * 100
        print(f"  {pattern_type.value}: {count} tasks ({percentage:.1f}%)")
    
    print("\\nSize Distribution:")
    for size_type, count in size_distributions.items():
        percentage = (count / 100) * 100
        print(f"  {size_type}: {count} tasks ({percentage:.1f}%)")
    
    return task_classifications

def demonstrate_expert_routing():
    """Demonstrate how tasks are routed to different experts"""
    
    print("\\n=== DEMONSTRATING EXPERT ROUTING ===")
    
    # Load training data
    with open('arc-agi_training_challenges.json', 'r') as f:
        training_data = json.load(f)
    
    # Initialize system
    system = FederatedARCSystem()
    
    # Select diverse examples for demonstration
    sample_task_ids = list(training_data.keys())[:5]
    
    for task_id in sample_task_ids:
        task_data = training_data[task_id]
        task = ARCTask(
            task_id=task_id,
            train_examples=task_data['train'],
            test_examples=task_data['test']
        )
        
        # Get task features
        features = task.get_pattern_features()
        
        # Route to experts
        pattern_type, confidence = system.router.classify_task(task)
        selected_experts = system.router.select_experts(task)
        
        print(f"\\nTask {task_id}:")
        print(f"  Input size: {features['input_size']}")
        print(f"  Output size: {features['output_size']}")
        print(f"  Size ratio: {features['size_ratio']:.2f}")
        print(f"  Color change: {features['color_change']}")
        print(f"  Pattern type: {pattern_type.value} (confidence: {confidence:.2f})")
        print(f"  Selected experts: {selected_experts}")

def demonstrate_knowledge_graph():
    """Demonstrate knowledge graph pattern storage and retrieval"""
    
    print("\\n=== DEMONSTRATING KNOWLEDGE GRAPH ===")
    
    # Load training data
    with open('arc-agi_training_challenges.json', 'r') as f:
        training_data = json.load(f)
    
    # Initialize system and populate KG
    system = FederatedARCSystem()
    
    # Add some patterns to the knowledge graph
    sample_tasks = list(training_data.items())[:10]
    
    for task_id, task_data in sample_tasks:
        task = ARCTask(
            task_id=task_id,
            train_examples=task_data['train'],
            test_examples=task_data['test']
        )
        
        pattern_type, _ = system.router.classify_task(task)
        features = task.get_pattern_features()
        transformation_rule = system._extract_transformation_rule(task)
        
        system.pattern_kg.add_pattern(
            pattern_id=task_id,
            pattern_type=pattern_type,
            features=features,
            transformation_rule=transformation_rule
        )
    
    print(f"Added {len(sample_tasks)} patterns to knowledge graph")
    
    # Test pattern retrieval
    test_task_id, test_task_data = list(training_data.items())[15]  # Use a different task
    test_task = ARCTask(
        task_id=test_task_id,
        train_examples=test_task_data['train'],
        test_examples=test_task_data['test']
    )
    
    query_features = test_task.get_pattern_features()
    similar_patterns = system.pattern_kg.find_similar_patterns(query_features)
    
    print(f"\\nQuerying for task {test_task_id}:")
    print(f"Query features: {query_features}")
    print(f"Similar patterns found: {similar_patterns}")

def run_end_to_end_example():
    """Run a complete end-to-end example"""
    
    print("\\n=== END-TO-END EXAMPLE ===")
    
    # Load training data
    with open('arc-agi_training_challenges.json', 'r') as f:
        training_data = json.load(f)
    
    # Initialize and train system
    system = FederatedARCSystem()
    print("Training system on full dataset...")
    system.train_on_dataset(training_data)
    
    # Select an interesting test case
    test_task_id = list(training_data.keys())[0]  # Take the first task
    test_task_data = training_data[test_task_id]
    
    test_task = ARCTask(
        task_id=test_task_id,
        train_examples=test_task_data['train'],
        test_examples=test_task_data['test']
    )
    
    print(f"\\nSolving task {test_task_id}...")
    
    # Show task details
    features = test_task.get_pattern_features()
    print(f"Task features: {features}")
    
    # Solve the task
    solution = system.solve_task(test_task)
    
    print(f"Generated solution: {solution}")
    print(f"Actual test input: {test_task.test_examples[0]['input']}")
    
    # Load the actual solution for comparison
    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    if test_task_id in solutions:
        actual_solution = solutions[test_task_id][0]  # First solution
        print(f"Actual solution: {actual_solution}")
        
        # Simple accuracy check
        if np.array_equal(solution, actual_solution):
            print("✅ CORRECT SOLUTION!")
        else:
            print("❌ Solution doesn't match (expected for demo)")

def main():
    """Run all demonstrations"""
    
    print("🚀 FEDERATED ARC SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    try:
        # 1. Analyze dataset
        classifications = analyze_dataset_distribution()
        
        # 2. Demonstrate routing
        demonstrate_expert_routing()
        
        # 3. Demonstrate knowledge graph
        demonstrate_knowledge_graph()
        
        # 4. Run end-to-end example
        run_end_to_end_example()
        
        print("\\n" + "=" * 50)
        print("🎉 DEMONSTRATION COMPLETED!")
        print("\\nNext steps:")
        print("1. Fine-tune expert models using QLoRA")
        print("2. Implement proper Neo4j knowledge graph")
        print("3. Add more sophisticated aggregation mechanisms")
        print("4. Evaluate on the full test set")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the ARC JSON files are in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()