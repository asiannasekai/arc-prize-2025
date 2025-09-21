"""
Federated Small Language Models for ARC Prize 2025
Implementation based on "Small Language Models as Federated Learners" paper

This module provides the core architecture for applying federated learning
to the Abstract Reasoning Corpus (ARC) challenge using specialized small
language models for different pattern transformation types.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict

class PatternType(Enum):
    """Classification of ARC transformation types"""
    TILING = "tiling"
    EXTRACTION = "extraction"
    MAPPING = "mapping"
    SYMMETRY = "symmetry"
    COMPLETION = "completion"
    RULE_BASED = "rule_based"

@dataclass
class ARCTask:
    """Structure for ARC task data"""
    task_id: str
    train_examples: List[Dict[str, List[List[int]]]]
    test_examples: List[Dict[str, List[List[int]]]]
    
    def get_pattern_features(self) -> Dict[str, Any]:
        """Extract pattern features for routing"""
        if not self.train_examples:
            return {}
            
        first_example = self.train_examples[0]
        input_grid = first_example['input']
        output_grid = first_example['output']
        
        input_size = (len(input_grid), len(input_grid[0]))
        output_size = (len(output_grid), len(output_grid[0]))
        
        # Calculate key features
        size_ratio = (output_size[0] * output_size[1]) / (input_size[0] * input_size[1])
        
        # Count unique colors
        input_colors = set()
        output_colors = set()
        for row in input_grid:
            input_colors.update(row)
        for row in output_grid:
            output_colors.update(row)
            
        return {
            'input_size': input_size,
            'output_size': output_size,
            'size_ratio': size_ratio,
            'input_colors': len(input_colors),
            'output_colors': len(output_colors),
            'color_change': len(output_colors - input_colors) > 0,
            'grid_area': input_size[0] * input_size[1]
        }

class PatternKnowledgeGraph:
    """Knowledge graph for storing and retrieving pattern transformations"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.pattern_embeddings = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_pattern(self, pattern_id: str, pattern_type: PatternType, 
                   features: Dict[str, Any], transformation_rule: str):
        """Add a pattern to the knowledge graph"""
        self.graph.add_node(pattern_id, 
                           pattern_type=pattern_type,
                           features=features,
                           transformation_rule=transformation_rule)
        
        # Create embedding for semantic similarity
        pattern_description = f"{pattern_type.value} size_ratio:{features.get('size_ratio', 1)} colors:{features.get('input_colors', 0)}"
        embedding = self.embedding_model.encode(pattern_description)
        self.pattern_embeddings[pattern_id] = embedding
        
    def find_similar_patterns(self, query_features: Dict[str, Any], top_k: int = 5) -> List[str]:
        """Find similar patterns using embedding similarity"""
        if not self.pattern_embeddings:
            return []
            
        query_desc = f"unknown size_ratio:{query_features.get('size_ratio', 1)} colors:{query_features.get('input_colors', 0)}"
        query_embedding = self.embedding_model.encode(query_desc)
        
        similarities = []
        for pattern_id, embedding in self.pattern_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((pattern_id, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [pattern_id for pattern_id, _ in similarities[:top_k]]

class TaskRouter:
    """Routes ARC tasks to appropriate expert models"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        # In practice, this would be a fine-tuned classification model
        
    def classify_task(self, task: ARCTask) -> Tuple[PatternType, float]:
        """Classify task and return primary pattern type with confidence"""
        features = task.get_pattern_features()
        
        # Rule-based classification (would be replaced with trained model)
        size_ratio = features.get('size_ratio', 1.0)
        color_change = features.get('color_change', False)
        grid_area = features.get('grid_area', 1)
        
        if size_ratio >= 4:
            return PatternType.TILING, 0.9
        elif size_ratio <= 0.5:
            return PatternType.EXTRACTION, 0.8
        elif color_change and size_ratio == 1.0:
            return PatternType.MAPPING, 0.85
        elif grid_area <= 25:
            return PatternType.COMPLETION, 0.7
        else:
            return PatternType.RULE_BASED, 0.6
            
    def select_experts(self, task: ARCTask) -> List[str]:
        """Select which expert models should handle this task"""
        primary_type, confidence = self.classify_task(task)
        features = task.get_pattern_features()
        
        experts = []
        
        # Primary expert based on pattern type
        experts.append(f"{primary_type.value}_expert")
        
        # Size-based expert
        grid_area = features.get('grid_area', 1)
        if grid_area <= 25:
            experts.append("micro_expert")
        elif grid_area <= 225:
            experts.append("standard_expert")
        else:
            experts.append("macro_expert")
            
        # Add fallback for low confidence
        if confidence < 0.7:
            experts.append("general_expert")
            
        return experts

class PatternExpert:
    """Base class for specialized pattern experts"""
    
    def __init__(self, expert_type: str, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.expert_type = expert_type
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # In practice, this would be the fine-tuned expert model
        self.model = None  # AutoModelForCausalLM.from_pretrained(model_name)
        
    def generate_transformation(self, task: ARCTask, kg_context: List[str]) -> Dict[str, Any]:
        """Generate transformation based on task and KG context"""
        # Create prompt with training examples and context
        prompt = self._create_prompt(task, kg_context)
        
        # In practice, would use the actual model
        # output = self.model.generate(prompt)
        
        # Mock output for demonstration
        return {
            'transformation': [[0, 1], [1, 0]],  # Mock output grid
            'confidence': 0.8,
            'reasoning': f"Applied {self.expert_type} transformation based on pattern {kg_context[0] if kg_context else 'default'}"
        }
        
    def _create_prompt(self, task: ARCTask, kg_context: List[str]) -> str:
        """Create prompt for the expert model"""
        prompt = f"You are a {self.expert_type} expert. "
        prompt += "Analyze these training examples and apply the transformation:\n\n"
        
        # Add training examples
        for i, example in enumerate(task.train_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
            
        # Add KG context
        if kg_context:
            prompt += f"Similar patterns: {kg_context}\n\n"
            
        # Add test input
        if task.test_examples:
            prompt += f"Apply transformation to: {task.test_examples[0]['input']}\n"
            
        return prompt

class FederatedARCSystem:
    """Main system orchestrating federated experts for ARC solving"""
    
    def __init__(self):
        self.router = TaskRouter()
        self.pattern_kg = PatternKnowledgeGraph()
        self.experts = self._initialize_experts()
        
    def _initialize_experts(self) -> Dict[str, PatternExpert]:
        """Initialize all expert models"""
        experts = {}
        
        # Pattern-specific experts
        for pattern_type in PatternType:
            experts[f"{pattern_type.value}_expert"] = PatternExpert(pattern_type.value)
            
        # Size-specific experts
        for size_type in ["micro", "standard", "macro"]:
            experts[f"{size_type}_expert"] = PatternExpert(size_type)
            
        # General fallback expert
        experts["general_expert"] = PatternExpert("general")
        
        return experts
        
    def train_on_dataset(self, training_data: Dict[str, Any]):
        """Train the system on ARC training data"""
        print(f"Training on {len(training_data)} tasks...")
        
        # Populate knowledge graph with training patterns
        for task_id, task_data in training_data.items():
            task = ARCTask(
                task_id=task_id,
                train_examples=task_data['train'],
                test_examples=task_data['test']
            )
            
            # Classify and store pattern
            pattern_type, confidence = self.router.classify_task(task)
            features = task.get_pattern_features()
            
            # Extract transformation rule (simplified)
            transformation_rule = self._extract_transformation_rule(task)
            
            # Add to knowledge graph
            self.pattern_kg.add_pattern(
                pattern_id=task_id,
                pattern_type=pattern_type,
                features=features,
                transformation_rule=transformation_rule
            )
            
        print("Training completed. Knowledge graph populated.")
        
    def _extract_transformation_rule(self, task: ARCTask) -> str:
        """Extract transformation rule from training examples (simplified)"""
        features = task.get_pattern_features()
        size_ratio = features.get('size_ratio', 1)
        
        if size_ratio > 4:
            return f"tile_pattern_{int(size_ratio)}x"
        elif size_ratio < 0.5:
            return f"extract_pattern_{1/size_ratio:.1f}x_reduction"
        else:
            return "apply_rule_transformation"
            
    def solve_task(self, task: ARCTask) -> List[List[int]]:
        """Solve an ARC task using federated experts"""
        
        # 1. Route to appropriate experts
        selected_experts = self.router.select_experts(task)
        print(f"Selected experts: {selected_experts}")
        
        # 2. Query knowledge graph for similar patterns
        features = task.get_pattern_features()
        similar_patterns = self.pattern_kg.find_similar_patterns(features)
        
        # 3. Generate solutions from each expert
        expert_outputs = []
        for expert_name in selected_experts:
            if expert_name in self.experts:
                expert = self.experts[expert_name]
                output = expert.generate_transformation(task, similar_patterns)
                expert_outputs.append((expert_name, output))
                
        # 4. Aggregate outputs using confidence weighting
        final_solution = self._aggregate_outputs(expert_outputs)
        
        return final_solution
        
    def _aggregate_outputs(self, expert_outputs: List[Tuple[str, Dict]]) -> List[List[int]]:
        """Aggregate outputs from multiple experts"""
        if not expert_outputs:
            return [[0, 0], [0, 0]]  # Default fallback
            
        # Simple confidence-weighted selection (would be more sophisticated)
        best_output = max(expert_outputs, key=lambda x: x[1]['confidence'])
        
        return best_output[1]['transformation']

def main():
    """Demonstration of the federated ARC system"""
    
    # Initialize system
    system = FederatedARCSystem()
    
    # Load training data
    print("Loading ARC training data...")
    with open('arc-agi_training_challenges.json', 'r') as f:
        training_data = json.load(f)
    
    # Train system (populate knowledge graph)
    system.train_on_dataset(training_data)
    
    # Test on a sample task
    sample_task_id = list(training_data.keys())[0]
    sample_task_data = training_data[sample_task_id]
    
    test_task = ARCTask(
        task_id=sample_task_id,
        train_examples=sample_task_data['train'],
        test_examples=sample_task_data['test']
    )
    
    print(f"\nSolving task {sample_task_id}...")
    solution = system.solve_task(test_task)
    print(f"Generated solution: {solution}")

if __name__ == "__main__":
    main()