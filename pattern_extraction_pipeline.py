"""
Pattern Extraction Pipeline for ARC Knowledge Graph
Extracts patterns from training data and populates Neo4j KG
"""

import json
import numpy as np
import pickle
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('/workspaces/arc-prize-2025')

from neo4j_pattern_kg import Neo4jPatternKG, TransformationType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedPattern:
    """Represents an extracted pattern from ARC data"""
    pattern_id: str
    pattern_type: TransformationType
    features: Dict[str, Any]
    transformation_rule: Dict[str, Any]
    examples: List[Dict[str, Any]]
    complexity_score: float
    
class ARCPatternExtractor:
    """Extracts and analyzes patterns from ARC tasks"""
    
    def __init__(self):
        self.extracted_patterns = []
        self.pattern_cache = {}
    
    def extract_spatial_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract spatial features from a grid"""
        height, width = grid.shape
        unique_colors = np.unique(grid)
        
        features = {
            'height': int(height),
            'width': int(width),
            'area': int(height * width),
            'unique_colors': len(unique_colors),
            'color_values': [int(c) for c in unique_colors],
            'density': float(np.count_nonzero(grid) / (height * width)),
            'symmetry_h': self._check_horizontal_symmetry(grid),
            'symmetry_v': self._check_vertical_symmetry(grid),
            'connectivity': self._analyze_connectivity(grid),
            'pattern_complexity': self._calculate_pattern_complexity(grid)
        }
        
        return features
    
    def _check_horizontal_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has horizontal symmetry"""
        return np.array_equal(grid, np.flipud(grid))
    
    def _check_vertical_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has vertical symmetry"""
        return np.array_equal(grid, np.fliplr(grid))
    
    def _analyze_connectivity(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze connectivity patterns in the grid"""
        # Simple connectivity analysis
        connected_components = 0
        visited = np.zeros_like(grid, dtype=bool)
        
        def dfs(i, j, color):
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or 
                visited[i, j] or grid[i, j] != color):
                return 0
            
            visited[i, j] = True
            size = 1
            
            # Check 4-connected neighbors
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                size += dfs(i + di, j + dj, color)
            
            return size
        
        component_sizes = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:  # Assuming 0 is background
                    size = dfs(i, j, grid[i, j])
                    if size > 0:
                        component_sizes.append(size)
                        connected_components += 1
        
        return {
            'num_components': connected_components,
            'component_sizes': component_sizes,
            'largest_component': max(component_sizes) if component_sizes else 0,
            'avg_component_size': np.mean(component_sizes) if component_sizes else 0
        }
    
    def _calculate_pattern_complexity(self, grid: np.ndarray) -> float:
        """Calculate a complexity score for the pattern"""
        # Factors contributing to complexity:
        # 1. Number of unique colors
        # 2. Size of the grid
        # 3. Spatial distribution of colors
        
        unique_colors = len(np.unique(grid))
        grid_size = grid.shape[0] * grid.shape[1]
        
        # Calculate entropy-like measure
        color_counts = {}
        for color in grid.flatten():
            color_counts[int(color)] = color_counts.get(int(color), 0) + 1
        
        entropy = 0
        total_cells = grid_size
        for count in color_counts.values():
            if count > 0:
                p = count / total_cells
                entropy -= p * np.log2(p)
        
        # Normalize complexity score
        complexity = (unique_colors * 0.3 + 
                     np.log(grid_size + 1) * 0.3 + 
                     entropy * 0.4)
        
        return float(complexity)
    
    def detect_transformation_type(self, input_grid: np.ndarray, 
                                 output_grid: np.ndarray) -> Tuple[TransformationType, float]:
        """Detect the type of transformation between input and output"""
        
        input_shape = input_grid.shape
        output_shape = output_grid.shape
        size_ratio = (output_shape[0] * output_shape[1]) / (input_shape[0] * input_shape[1])
        
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        
        # Rule-based classification
        confidence = 0.7  # Default confidence
        
        # Tiling: Output is larger and contains repeated patterns
        if size_ratio >= 2.0 and output_shape[0] >= input_shape[0] and output_shape[1] >= input_shape[1]:
            # Check if input pattern is repeated in output
            if (output_shape[0] % input_shape[0] == 0 and 
                output_shape[1] % input_shape[1] == 0):
                confidence = 0.9
            return TransformationType.TILING, confidence
        
        # Extraction: Output is smaller
        if size_ratio <= 0.8:
            confidence = 0.85
            return TransformationType.EXTRACTION, confidence
        
        # Mapping: Same size, different colors
        if (input_shape == output_shape and 
            input_colors != output_colors):
            confidence = 0.9
            return TransformationType.MAPPING, confidence
        
        # Symmetry: Same size, same colors, but different arrangement
        if (input_shape == output_shape and 
            input_colors == output_colors and 
            not np.array_equal(input_grid, output_grid)):
            # Check for simple transformations
            if (np.array_equal(output_grid, np.rot90(input_grid)) or
                np.array_equal(output_grid, np.flipud(input_grid)) or
                np.array_equal(output_grid, np.fliplr(input_grid))):
                confidence = 0.95
            else:
                confidence = 0.6
            return TransformationType.SYMMETRY, confidence
        
        # Default to rule-based
        return TransformationType.RULE_BASED, 0.5
    
    def extract_transformation_rule(self, input_grid: np.ndarray, 
                                  output_grid: np.ndarray,
                                  transformation_type: TransformationType) -> Dict[str, Any]:
        """Extract specific transformation rule"""
        
        rule = {
            'type': transformation_type.value,
            'parameters': {},
            'confidence': 0.5
        }
        
        if transformation_type == TransformationType.TILING:
            if (output_grid.shape[0] % input_grid.shape[0] == 0 and 
                output_grid.shape[1] % input_grid.shape[1] == 0):
                scale_y = output_grid.shape[0] // input_grid.shape[0]
                scale_x = output_grid.shape[1] // input_grid.shape[1]
                rule['parameters'] = {
                    'scale_x': int(scale_x),
                    'scale_y': int(scale_y),
                    'total_scale': int(scale_x * scale_y)
                }
                rule['confidence'] = 0.9
        
        elif transformation_type == TransformationType.EXTRACTION:
            extraction_ratio = (output_grid.shape[0] * output_grid.shape[1]) / (input_grid.shape[0] * input_grid.shape[1])
            rule['parameters'] = {
                'extraction_ratio': float(extraction_ratio),
                'output_shape': [int(output_grid.shape[0]), int(output_grid.shape[1])],
                'input_shape': [int(input_grid.shape[0]), int(input_grid.shape[1])]
            }
            rule['confidence'] = 0.8
        
        elif transformation_type == TransformationType.MAPPING:
            # Detect color mapping
            if input_grid.shape == output_grid.shape:
                color_mapping = {}
                for inp_val, out_val in zip(input_grid.flatten(), output_grid.flatten()):
                    inp_val = int(inp_val)
                    out_val = int(out_val)
                    if inp_val not in color_mapping:
                        color_mapping[inp_val] = out_val
                    elif color_mapping[inp_val] != out_val:
                        # Inconsistent mapping
                        color_mapping = {}
                        break
                
                rule['parameters'] = {
                    'color_mapping': color_mapping,
                    'is_consistent': bool(color_mapping)
                }
                rule['confidence'] = 0.9 if color_mapping else 0.6
        
        elif transformation_type == TransformationType.SYMMETRY:
            # Check specific symmetry operations
            symmetry_ops = []
            if np.array_equal(output_grid, np.rot90(input_grid)):
                symmetry_ops.append('rotate_90_cw')
            elif np.array_equal(output_grid, np.rot90(input_grid, k=3)):
                symmetry_ops.append('rotate_90_ccw')
            elif np.array_equal(output_grid, np.rot90(input_grid, k=2)):
                symmetry_ops.append('rotate_180')
            elif np.array_equal(output_grid, np.flipud(input_grid)):
                symmetry_ops.append('flip_horizontal')
            elif np.array_equal(output_grid, np.fliplr(input_grid)):
                symmetry_ops.append('flip_vertical')
            
            rule['parameters'] = {
                'operations': symmetry_ops,
                'detected_operations': len(symmetry_ops)
            }
            rule['confidence'] = 0.95 if symmetry_ops else 0.5
        
        return rule
    
    def extract_pattern_from_task(self, task_id: str, task_data: Dict) -> ExtractedPattern:
        """Extract pattern from a single ARC task"""
        
        if task_id in self.pattern_cache:
            return self.pattern_cache[task_id]
        
        train_examples = task_data['train']
        
        if not train_examples:
            raise ValueError(f"Task {task_id} has no training examples")
        
        # Analyze first example for primary pattern
        first_example = train_examples[0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        # Extract features
        input_features = self.extract_spatial_features(input_grid)
        output_features = self.extract_spatial_features(output_grid)
        
        # Detect transformation
        transformation_type, confidence = self.detect_transformation_type(input_grid, output_grid)
        transformation_rule = self.extract_transformation_rule(input_grid, output_grid, transformation_type)
        
        # Combined features
        combined_features = {
            'input_features': input_features,
            'output_features': output_features,
            'size_ratio': float((output_grid.shape[0] * output_grid.shape[1]) / 
                               (input_grid.shape[0] * input_grid.shape[1])),
            'color_change': input_features['color_values'] != output_features['color_values'],
            'shape_change': input_features['height'] != output_features['height'] or 
                           input_features['width'] != output_features['width'],
            'num_examples': len(train_examples)
        }
        
        # Calculate overall complexity
        complexity_score = (input_features['pattern_complexity'] + 
                          output_features['pattern_complexity'] + 
                          len(train_examples)) / 3.0
        
        # Prepare examples for storage
        examples = []
        for i, example in enumerate(train_examples):
            examples.append({
                'example_id': i,
                'input_grid': example['input'],
                'output_grid': example['output'],
                'input_shape': [len(example['input']), len(example['input'][0])],
                'output_shape': [len(example['output']), len(example['output'][0])]
            })
        
        pattern = ExtractedPattern(
            pattern_id=task_id,
            pattern_type=transformation_type,
            features=combined_features,
            transformation_rule=transformation_rule,
            examples=examples,
            complexity_score=complexity_score
        )
        
        self.pattern_cache[task_id] = pattern
        return pattern
    
    def process_all_tasks(self, training_file: str) -> List[ExtractedPattern]:
        """Process all training tasks and extract patterns"""
        
        logger.info(f"Loading training data from {training_file}")
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        patterns = []
        
        logger.info(f"Extracting patterns from {len(training_data)} tasks...")
        
        for task_id, task_data in tqdm(training_data.items(), desc="Extracting patterns"):
            try:
                pattern = self.extract_pattern_from_task(task_id, task_data)
                patterns.append(pattern)
            except Exception as e:
                logger.error(f"Failed to extract pattern from task {task_id}: {e}")
        
        logger.info(f"Successfully extracted {len(patterns)} patterns")
        self.extracted_patterns = patterns
        return patterns

class KnowledgeGraphPopulator:
    """Populates Neo4j knowledge graph with extracted patterns"""
    
    def __init__(self):
        self.kg = Neo4jPatternKG()
    
    def populate_from_patterns(self, patterns: List[ExtractedPattern]):
        """Populate knowledge graph with extracted patterns"""
        
        logger.info(f"Populating knowledge graph with {len(patterns)} patterns...")
        
        success_count = 0
        for pattern in tqdm(patterns, desc="Adding to KG"):
            try:
                # Convert pattern examples to expected format
                formatted_examples = []
                for example in pattern.examples:
                    formatted_examples.append({
                        'input': example['input_grid'],
                        'output': example['output_grid']
                    })
                
                # Convert pattern to task data format for KG
                task_data = {
                    'train': formatted_examples
                }
                
                # Convert pattern type string back to enum
                pattern_type_str = pattern.pattern_type.value if hasattr(pattern.pattern_type, 'value') else str(pattern.pattern_type)
                transformation_type = TransformationType(pattern_type_str)
                
                self.kg.add_pattern(
                    task_id=pattern.pattern_id,
                    task_data=task_data,
                    transformation_type=transformation_type
                )
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to add pattern {pattern.pattern_id} to KG: {e}")
        
        logger.info(f"Successfully added {success_count}/{len(patterns)} patterns to knowledge graph")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return self.kg.get_pattern_statistics()
    
    def close(self):
        """Close knowledge graph connection"""
        self.kg.close()

def main():
    """Main function to build pattern extraction pipeline"""
    
    print("🔍 Building Pattern Extraction Pipeline")
    print("=" * 45)
    
    # Initialize extractor
    extractor = ARCPatternExtractor()
    
    # Extract patterns from training data
    training_file = "/workspaces/arc-prize-2025/arc-agi_training_challenges.json"
    patterns = extractor.process_all_tasks(training_file)
    
    # Save extracted patterns
    patterns_file = "extracted_patterns.pkl"
    with open(patterns_file, 'wb') as f:
        pickle.dump(patterns, f)
    logger.info(f"✅ Saved extracted patterns to {patterns_file}")
    
    # Analyze pattern distribution
    pattern_counts = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type.value
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print("\\n📊 Extracted Pattern Distribution:")
    print("-" * 35)
    total = len(patterns)
    for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"{pattern_type:12}: {count:4d} ({percentage:5.1f}%)")
    
    # Initialize KG populator
    print("\\n🗄️ Populating Neo4j Knowledge Graph...")
    populator = KnowledgeGraphPopulator()
    
    # Populate knowledge graph
    populator.populate_from_patterns(patterns)
    
    # Get KG statistics
    kg_stats = populator.get_statistics()
    
    print("\\n📈 Knowledge Graph Statistics:")
    print("-" * 32)
    print(f"Total patterns: {kg_stats.get('total_patterns', 0)}")
    
    if 'by_type' in kg_stats:
        print("\\nBy pattern type:")
        for pattern_type, stats in kg_stats['by_type'].items():
            count = stats['count']
            avg_complexity = stats.get('avg_complexity', 0)
            print(f"  {pattern_type:12}: {count:4d} patterns (avg complexity: {avg_complexity:.2f})")
    
    if 'overall_stats' in kg_stats:
        overall = kg_stats['overall_stats']
        print(f"\\nOverall statistics:")
        avg_size = overall.get('avg_size_ratio', 0)
        avg_complexity = overall.get('avg_complexity', 0)
        min_size = overall.get('min_size_ratio', 0)
        max_size = overall.get('max_size_ratio', 0)
        
        print(f"  Average size ratio: {avg_size:.2f}" if avg_size is not None else "  Average size ratio: N/A")
        print(f"  Average complexity: {avg_complexity:.2f}" if avg_complexity is not None else "  Average complexity: N/A")
        print(f"  Size ratio range: {min_size:.2f} - {max_size:.2f}" if min_size is not None and max_size is not None else "  Size ratio range: N/A")
    
    # Save pattern analysis
    analysis = {
        'total_patterns': len(patterns),
        'pattern_distribution': pattern_counts,
        'kg_statistics': kg_stats,
        'complexity_stats': {
            'min_complexity': min(p.complexity_score for p in patterns),
            'max_complexity': max(p.complexity_score for p in patterns),
            'avg_complexity': np.mean([p.complexity_score for p in patterns])
        }
    }
    
    with open('pattern_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info("✅ Saved pattern analysis to pattern_analysis.json")
    
    # Close KG connection
    populator.close()
    
    print("\\n🎉 Pattern Extraction Pipeline Complete!")
    print(f"✅ Extracted {len(patterns)} patterns")
    print(f"✅ Populated knowledge graph with {kg_stats['total_patterns']} patterns")
    print(f"✅ Created analysis files:")
    print(f"   - extracted_patterns.pkl")
    print(f"   - pattern_analysis.json")

if __name__ == "__main__":
    main()