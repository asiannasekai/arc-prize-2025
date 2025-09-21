"""
Task Classification Dataset Creator for ARC Router Training
Processes all 1000 training tasks to create labeled dataset for expert routing
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle
from tqdm import tqdm
import logging
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    TILING = "tiling"
    EXTRACTION = "extraction" 
    MAPPING = "mapping"
    SYMMETRY = "symmetry"
    COMPLETION = "completion"
    RULE_BASED = "rule_based"

@dataclass
class TaskFeatures:
    """Features extracted from an ARC task"""
    task_id: str
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    size_ratio: float
    input_colors: int
    output_colors: int
    color_change: bool
    shape_change: bool
    grid_area: int
    complexity_score: float
    training_examples: int
    pattern_type: PatternType
    confidence: float

class ARCPatternClassifier:
    """Classifies ARC tasks into pattern types for router training"""
    
    def __init__(self):
        self.features_cache = {}
        self.classification_rules = self._create_classification_rules()
    
    def _create_classification_rules(self) -> Dict[str, Any]:
        """Create rule-based classification system"""
        return {
            'tiling': {
                'size_ratio_min': 4.0,
                'size_ratio_max': 36.0,
                'shape_change': True,
                'keywords': ['repeat', 'tile', 'pattern', 'replicate']
            },
            'extraction': {
                'size_ratio_min': 0.1,
                'size_ratio_max': 0.8,
                'shape_change': True,
                'keywords': ['extract', 'select', 'filter', 'isolate']
            },
            'mapping': {
                'size_ratio_min': 0.9,
                'size_ratio_max': 1.1,
                'color_change': True,
                'shape_change': False,
                'keywords': ['color', 'map', 'transform', 'replace']
            },
            'symmetry': {
                'size_ratio_min': 0.9,
                'size_ratio_max': 1.1,
                'shape_change': False,
                'keywords': ['rotate', 'flip', 'mirror', 'reflect']
            },
            'completion': {
                'size_ratio_min': 0.9,
                'size_ratio_max': 1.1,
                'keywords': ['complete', 'fill', 'continue', 'extend']
            },
            'rule_based': {
                'default': True,
                'keywords': ['rule', 'logic', 'condition', 'if']
            }
        }
    
    def extract_features(self, task_id: str, task_data: Dict) -> TaskFeatures:
        """Extract comprehensive features from a task"""
        if task_id in self.features_cache:
            return self.features_cache[task_id]
        
        train_examples = task_data['train']
        
        if not train_examples:
            raise ValueError(f"Task {task_id} has no training examples")
        
        # Analyze first training example
        input_grid = np.array(train_examples[0]['input'])
        output_grid = np.array(train_examples[0]['output'])
        
        input_shape = input_grid.shape
        output_shape = output_grid.shape
        size_ratio = (output_shape[0] * output_shape[1]) / (input_shape[0] * input_shape[1])
        
        # Color analysis
        input_colors = len(np.unique(input_grid))
        output_colors = len(np.unique(output_grid))
        color_change = input_colors != output_colors or not np.array_equal(
            np.unique(input_grid), np.unique(output_grid)
        )
        
        # Shape analysis
        shape_change = input_shape != output_shape
        
        # Grid area and complexity
        grid_area = input_shape[0] * input_shape[1]
        complexity_score = (grid_area + input_colors + output_colors + len(train_examples)) / 50.0
        
        # Classify pattern type
        pattern_type, confidence = self._classify_pattern(
            size_ratio, color_change, shape_change, input_colors, output_colors, grid_area
        )
        
        features = TaskFeatures(
            task_id=task_id,
            input_shape=input_shape,
            output_shape=output_shape,
            size_ratio=size_ratio,
            input_colors=input_colors,
            output_colors=output_colors,
            color_change=color_change,
            shape_change=shape_change,
            grid_area=grid_area,
            complexity_score=complexity_score,
            training_examples=len(train_examples),
            pattern_type=pattern_type,
            confidence=confidence
        )
        
        self.features_cache[task_id] = features
        return features
    
    def _classify_pattern(self, size_ratio: float, color_change: bool, shape_change: bool,
                         input_colors: int, output_colors: int, grid_area: int) -> Tuple[PatternType, float]:
        """Classify pattern type using rule-based approach"""
        
        scores = {}
        
        # Tiling patterns
        if size_ratio >= 4.0 and shape_change:
            scores[PatternType.TILING] = 0.9
        elif size_ratio >= 2.0 and shape_change:
            scores[PatternType.TILING] = 0.7
        
        # Extraction patterns
        if size_ratio <= 0.8 and shape_change:
            scores[PatternType.EXTRACTION] = 0.85
        elif size_ratio <= 0.95 and shape_change:
            scores[PatternType.EXTRACTION] = 0.6
        
        # Mapping patterns
        if 0.9 <= size_ratio <= 1.1 and color_change and not shape_change:
            scores[PatternType.MAPPING] = 0.9
        elif 0.8 <= size_ratio <= 1.2 and color_change:
            scores[PatternType.MAPPING] = 0.7
        
        # Symmetry patterns (detect if output could be rotation/reflection)
        if 0.9 <= size_ratio <= 1.1 and not color_change and not shape_change:
            scores[PatternType.SYMMETRY] = 0.6
        
        # Completion patterns (similar size, slight color increase)
        if 0.9 <= size_ratio <= 1.1 and output_colors > input_colors:
            scores[PatternType.COMPLETION] = 0.7
        
        # Rule-based (default for complex patterns)
        scores[PatternType.RULE_BASED] = 0.5
        
        # Select highest scoring pattern
        if scores:
            best_pattern = max(scores.items(), key=lambda x: x[1])
            return best_pattern[0], best_pattern[1]
        else:
            return PatternType.RULE_BASED, 0.5
    
    def process_all_tasks(self, training_file: str) -> List[TaskFeatures]:
        """Process all training tasks and extract features"""
        logger.info(f"Loading training data from {training_file}")
        
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        all_features = []
        
        logger.info(f"Processing {len(training_data)} training tasks...")
        
        for task_id, task_data in tqdm(training_data.items(), desc="Extracting features"):
            try:
                features = self.extract_features(task_id, task_data)
                all_features.append(features)
            except Exception as e:
                logger.error(f"Failed to process task {task_id}: {e}")
        
        logger.info(f"Successfully processed {len(all_features)} tasks")
        return all_features
    
    def create_router_dataset(self, features_list: List[TaskFeatures]) -> pd.DataFrame:
        """Create training dataset for router model"""
        
        data_rows = []
        
        for features in features_list:
            row = {
                'task_id': features.task_id,
                'input_height': features.input_shape[0],
                'input_width': features.input_shape[1],
                'output_height': features.output_shape[0],
                'output_width': features.output_shape[1],
                'size_ratio': features.size_ratio,
                'input_colors': features.input_colors,
                'output_colors': features.output_colors,
                'color_change': features.color_change,
                'shape_change': features.shape_change,
                'grid_area': features.grid_area,
                'complexity_score': features.complexity_score,
                'training_examples': features.training_examples,
                'pattern_type': features.pattern_type.value,
                'confidence': features.confidence
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        return df
    
    def analyze_distribution(self, features_list: List[TaskFeatures]) -> Dict[str, Any]:
        """Analyze pattern type distribution"""
        
        pattern_counts = {}
        confidence_scores = {}
        
        for features in features_list:
            pattern_type = features.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            if pattern_type not in confidence_scores:
                confidence_scores[pattern_type] = []
            confidence_scores[pattern_type].append(features.confidence)
        
        # Calculate statistics
        total_tasks = len(features_list)
        distribution = {}
        
        for pattern_type, count in pattern_counts.items():
            avg_confidence = np.mean(confidence_scores[pattern_type])
            distribution[pattern_type] = {
                'count': count,
                'percentage': (count / total_tasks) * 100,
                'avg_confidence': avg_confidence
            }
        
        return {
            'total_tasks': total_tasks,
            'distribution': distribution,
            'sorted_by_count': sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        }

def create_expert_datasets(features_list: List[TaskFeatures], training_data: Dict) -> Dict[str, List[str]]:
    """Create datasets for each expert model"""
    
    expert_datasets = {
        'tiling_expert': [],
        'extraction_expert': [],
        'mapping_expert': [],
        'symmetry_expert': [],
        'completion_expert': [],
        'rule_expert': []
    }
    
    for features in features_list:
        pattern_type = features.pattern_type.value
        task_id = features.task_id
        
        if pattern_type == 'tiling':
            expert_datasets['tiling_expert'].append(task_id)
        elif pattern_type == 'extraction':
            expert_datasets['extraction_expert'].append(task_id)
        elif pattern_type == 'mapping':
            expert_datasets['mapping_expert'].append(task_id)
        elif pattern_type == 'symmetry':
            expert_datasets['symmetry_expert'].append(task_id)
        elif pattern_type == 'completion':
            expert_datasets['completion_expert'].append(task_id)
        else:
            expert_datasets['rule_expert'].append(task_id)
    
    return expert_datasets

def main():
    """Main function to create classification dataset"""
    
    print("🔬 Creating ARC Task Classification Dataset")
    print("=" * 50)
    
    # Initialize classifier
    classifier = ARCPatternClassifier()
    
    # Process all training tasks
    training_file = "/workspaces/arc-prize-2025/arc-agi_training_challenges.json"
    features_list = classifier.process_all_tasks(training_file)
    
    # Create router training dataset
    router_df = classifier.create_router_dataset(features_list)
    
    # Save router dataset
    router_df.to_csv('arc_router_training_data.csv', index=False)
    logger.info("✅ Saved router training data to arc_router_training_data.csv")
    
    # Save features as pickle for fast loading
    with open('arc_task_features.pkl', 'wb') as f:
        pickle.dump(features_list, f)
    logger.info("✅ Saved task features to arc_task_features.pkl")
    
    # Analyze distribution
    distribution = classifier.analyze_distribution(features_list)
    
    print("\\n📊 Pattern Type Distribution:")
    print("-" * 30)
    for pattern_type, stats in distribution['distribution'].items():
        count = stats['count']
        percentage = stats['percentage']
        confidence = stats['avg_confidence']
        print(f"{pattern_type:12}: {count:4d} tasks ({percentage:5.1f}%) - confidence: {confidence:.2f}")
    
    # Create expert datasets
    with open(training_file, 'r') as f:
        training_data = json.load(f)
    
    expert_datasets = create_expert_datasets(features_list, training_data)
    
    print("\\n🎯 Expert Dataset Sizes:")
    print("-" * 25)
    for expert, task_ids in expert_datasets.items():
        print(f"{expert:15}: {len(task_ids):4d} tasks")
    
    # Save expert datasets
    with open('arc_expert_datasets.json', 'w') as f:
        json.dump(expert_datasets, f, indent=2)
    logger.info("✅ Saved expert datasets to arc_expert_datasets.json")
    
    # Save distribution analysis
    with open('arc_pattern_distribution.json', 'w') as f:
        json.dump(distribution, f, indent=2)
    logger.info("✅ Saved distribution analysis to arc_pattern_distribution.json")
    
    print("\\n🎉 Task Classification Dataset Creation Complete!")
    print(f"📈 Total tasks processed: {len(features_list)}")
    print(f"📁 Files created:")
    print(f"   - arc_router_training_data.csv ({len(router_df)} rows)")
    print(f"   - arc_task_features.pkl")
    print(f"   - arc_expert_datasets.json")
    print(f"   - arc_pattern_distribution.json")

if __name__ == "__main__":
    main()