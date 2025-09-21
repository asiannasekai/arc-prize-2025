"""
Full Inference Pipeline for ARC-AGI Challenge
Integrates router, expert models, knowledge graph retrieval, and response aggregation
for end-to-end ARC task solving targeting 85% accuracy for $600K prize.

Key insight: To beat the current 4% AI performance, we need a sophisticated pipeline
that can handle novel patterns through expert specialization, knowledge retrieval,
and uncertainty-aware aggregation.
"""

import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import our federated system components
from multi_expert_aggregation import (
    ExpertPrediction, AggregatedPrediction, 
    MultiExpertAggregationSystem, WeightedVotingAggregator,
    BayesianModelAveraging, ConsensusAggregator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ARCTaskInput:
    """Input data for an ARC task"""
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_input: np.ndarray
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ARCTaskOutput:
    """Output from the inference pipeline"""
    task_id: str
    prediction: np.ndarray
    confidence: float
    expert_contributions: Dict[str, float]
    reasoning: str
    processing_time: float
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline"""
    # Router settings
    router_confidence_threshold: float = 0.6
    max_experts_per_task: int = 3
    
    # Expert settings
    expert_timeout: float = 30.0  # seconds
    parallel_expert_inference: bool = True
    
    # Knowledge graph settings
    kg_similarity_threshold: float = 0.7
    max_similar_patterns: int = 10
    
    # Aggregation settings
    aggregation_strategy: str = "weighted_voting"  # "weighted_voting", "bayesian", "consensus"
    confidence_threshold: float = 0.3
    
    # Caching settings
    enable_caching: bool = True
    cache_dir: str = "./cache"
    
    # Performance settings
    enable_profiling: bool = False
    log_level: str = "INFO"

class KnowledgeGraphRetriever:
    """Retrieves relevant patterns from Neo4j knowledge graph"""
    
    def __init__(self, connection_params: Dict[str, Any] = None):
        self.connection_params = connection_params or {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'neo4jpassword'
        }
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.connection_params['uri'],
                auth=(self.connection_params['user'], self.connection_params['password'])
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j knowledge graph")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Using mock retrieval.")
            self.driver = None
    
    def retrieve_similar_patterns(self, 
                                 task: ARCTaskInput, 
                                 similarity_threshold: float = 0.7,
                                 max_patterns: int = 10) -> List[Dict[str, Any]]:
        """Retrieve patterns similar to the input task"""
        if not self.driver:
            return self._mock_pattern_retrieval(task, max_patterns)
        
        try:
            with self.driver.session() as session:
                # Compute task features for similarity search
                task_features = self._extract_task_features(task)
                
                # Cypher query to find similar patterns
                query = """
                MATCH (p:Pattern)
                WHERE p.grid_size = $grid_size
                AND p.num_colors = $num_colors
                AND p.complexity_score >= $min_complexity
                AND p.complexity_score <= $max_complexity
                RETURN p.pattern_id as pattern_id,
                       p.transformation_type as transformation_type,
                       p.description as description,
                       p.confidence_score as confidence,
                       p.frequency as frequency
                ORDER BY p.confidence_score DESC
                LIMIT $max_patterns
                """
                
                result = session.run(
                    query,
                    grid_size=task_features['grid_size'],
                    num_colors=task_features['num_colors'],
                    min_complexity=task_features['complexity'] - 0.2,
                    max_complexity=task_features['complexity'] + 0.2,
                    max_patterns=max_patterns
                )
                
                patterns = []
                for record in result:
                    patterns.append({
                        'pattern_id': record['pattern_id'],
                        'transformation_type': record['transformation_type'],
                        'description': record['description'],
                        'confidence': record['confidence'],
                        'frequency': record['frequency']
                    })
                
                return patterns
                
        except Exception as e:
            logger.warning(f"Error retrieving patterns from Neo4j: {e}")
            return self._mock_pattern_retrieval(task, max_patterns)
    
    def _extract_task_features(self, task: ARCTaskInput) -> Dict[str, Any]:
        """Extract features from task for similarity matching"""
        # Use first training example for feature extraction
        if task.train_examples:
            input_grid, output_grid = task.train_examples[0]
        else:
            input_grid = task.test_input
            output_grid = task.test_input
        
        return {
            'grid_size': f"{input_grid.shape[0]}x{input_grid.shape[1]}",
            'num_colors': len(np.unique(input_grid)),
            'complexity': np.std(input_grid.flatten()) / np.mean(input_grid.flatten() + 1)
        }
    
    def _mock_pattern_retrieval(self, task: ARCTaskInput, max_patterns: int) -> List[Dict[str, Any]]:
        """Mock pattern retrieval when Neo4j is not available"""
        mock_patterns = [
            {
                'pattern_id': 'mock_tiling_001',
                'transformation_type': 'tiling',
                'description': 'Repeating 2x2 pattern across grid',
                'confidence': 0.85,
                'frequency': 45
            },
            {
                'pattern_id': 'mock_symmetry_001', 
                'transformation_type': 'symmetry',
                'description': 'Horizontal reflection transformation',
                'confidence': 0.78,
                'frequency': 32
            },
            {
                'pattern_id': 'mock_mapping_001',
                'transformation_type': 'mapping',
                'description': 'Color value increment by 1',
                'confidence': 0.72,
                'frequency': 28
            }
        ]
        return mock_patterns[:max_patterns]
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

class TaskRouter:
    """Routes tasks to appropriate expert models"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "best_router_model.pkl"
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained router model"""
        try:
            # Load the trained model components
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open("best_router_model_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open("best_router_model_label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            with open("best_router_model_features.json", 'r') as f:
                self.feature_names = json.load(f)
                
            logger.info("Router model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load router model: {e}. Using mock router.")
            self.model = None
    
    def route_task(self, 
                   task: ARCTaskInput, 
                   kg_patterns: List[Dict[str, Any]] = None,
                   max_experts: int = 3) -> List[str]:
        """Route task to most appropriate expert models"""
        if not self.model:
            return self._mock_routing(task, max_experts)
        
        try:
            # Extract features from task
            features = self._extract_routing_features(task, kg_patterns)
            
            # Scale features
            feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            # Get top expert types
            expert_types = self.label_encoder.classes_
            expert_probs = list(zip(expert_types, probabilities))
            expert_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top experts above threshold
            selected_experts = []
            for expert_type, prob in expert_probs[:max_experts]:
                if prob > 0.1:  # Minimum probability threshold
                    selected_experts.append(expert_type)
            
            return selected_experts if selected_experts else [expert_probs[0][0]]
            
        except Exception as e:
            logger.warning(f"Error in task routing: {e}. Using fallback routing.")
            return self._mock_routing(task, max_experts)
    
    def _extract_routing_features(self, 
                                  task: ARCTaskInput, 
                                  kg_patterns: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract features for router decision making"""
        # Use first training example or test input
        if task.train_examples:
            input_grid, output_grid = task.train_examples[0]
        else:
            input_grid = task.test_input
            output_grid = task.test_input
        
        features = {}
        
        # Basic grid features
        features['grid_height'] = float(input_grid.shape[0])
        features['grid_width'] = float(input_grid.shape[1])
        features['grid_area'] = float(input_grid.size)
        features['aspect_ratio'] = float(input_grid.shape[1] / input_grid.shape[0])
        
        # Color features
        features['num_colors'] = float(len(np.unique(input_grid)))
        features['color_diversity'] = float(len(np.unique(input_grid)) / input_grid.size)
        
        # Pattern features
        features['has_symmetry'] = float(self._check_symmetry(input_grid))
        features['has_repetition'] = float(self._check_repetition(input_grid))
        features['complexity_score'] = float(self._compute_complexity(input_grid))
        
        # Knowledge graph features
        if kg_patterns:
            features['kg_pattern_count'] = float(len(kg_patterns))
            features['kg_max_confidence'] = float(max([p['confidence'] for p in kg_patterns], default=0))
            features['kg_avg_confidence'] = float(np.mean([p['confidence'] for p in kg_patterns]) if kg_patterns else 0)
        else:
            features['kg_pattern_count'] = 0.0
            features['kg_max_confidence'] = 0.0
            features['kg_avg_confidence'] = 0.0
        
        return features
    
    def _check_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has symmetry"""
        return (np.array_equal(grid, np.fliplr(grid)) or 
                np.array_equal(grid, np.flipud(grid)) or
                np.array_equal(grid, grid.T))
    
    def _check_repetition(self, grid: np.ndarray) -> bool:
        """Check if grid has repeating patterns"""
        h, w = grid.shape
        # Check for 2x2 repetition
        if h >= 4 and w >= 4:
            top_left = grid[:2, :2]
            return (np.array_equal(top_left, grid[2:4, :2]) and
                    np.array_equal(top_left, grid[:2, 2:4]) and
                    np.array_equal(top_left, grid[2:4, 2:4]))
        return False
    
    def _compute_complexity(self, grid: np.ndarray) -> float:
        """Compute complexity score for grid"""
        # Based on color distribution and spatial patterns
        unique_colors = len(np.unique(grid))
        color_entropy = -np.sum([(np.sum(grid == c) / grid.size) * 
                                np.log2((np.sum(grid == c) / grid.size) + 1e-10) 
                                for c in np.unique(grid)])
        return color_entropy / np.log2(unique_colors + 1)
    
    def _mock_routing(self, task: ARCTaskInput, max_experts: int) -> List[str]:
        """Mock routing when model is not available"""
        # Simple heuristic routing
        input_grid = task.test_input
        experts = []
        
        # Check for tiling patterns
        if input_grid.shape[0] % 2 == 0 and input_grid.shape[1] % 2 == 0:
            experts.append("tiling")
        
        # Check for symmetry
        if (np.array_equal(input_grid, np.fliplr(input_grid)) or 
            np.array_equal(input_grid, np.flipud(input_grid))):
            experts.append("symmetry")
        
        # Check for color complexity
        if len(np.unique(input_grid)) > 3:
            experts.append("mapping")
        else:
            experts.append("extraction")
        
        return experts[:max_experts]

class ExpertModelManager:
    """Manages multiple expert models and their inference"""
    
    def __init__(self, expert_configs: Dict[str, Dict[str, Any]] = None):
        self.expert_configs = expert_configs or self._default_expert_configs()
        self.expert_models = {}
        self.expert_stats = {expert: {'calls': 0, 'avg_time': 0.0, 'success_rate': 1.0} 
                           for expert in self.expert_configs.keys()}
        
    def _default_expert_configs(self) -> Dict[str, Dict[str, Any]]:
        """Default configuration for expert models"""
        return {
            'tiling': {
                'model_path': 'expert_models/tiling_expert',
                'base_confidence': 0.8,
                'timeout': 30.0
            },
            'symmetry': {
                'model_path': 'expert_models/symmetry_expert', 
                'base_confidence': 0.75,
                'timeout': 25.0
            },
            'extraction': {
                'model_path': 'expert_models/extraction_expert',
                'base_confidence': 0.7,
                'timeout': 35.0
            },
            'mapping': {
                'model_path': 'expert_models/mapping_expert',
                'base_confidence': 0.65,
                'timeout': 30.0
            },
            'rule': {
                'model_path': 'expert_models/rule_expert',
                'base_confidence': 0.6,
                'timeout': 40.0
            }
        }
    
    def get_expert_predictions(self, 
                              task: ARCTaskInput,
                              selected_experts: List[str],
                              kg_patterns: List[Dict[str, Any]] = None,
                              parallel: bool = True) -> List[ExpertPrediction]:
        """Get predictions from selected expert models"""
        predictions = []
        
        if parallel and len(selected_experts) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=len(selected_experts)) as executor:
                future_to_expert = {
                    executor.submit(self._get_single_expert_prediction, task, expert, kg_patterns): expert
                    for expert in selected_experts
                }
                
                for future in as_completed(future_to_expert):
                    expert = future_to_expert[future]
                    try:
                        prediction = future.result(timeout=self.expert_configs[expert]['timeout'])
                        if prediction:
                            predictions.append(prediction)
                    except Exception as e:
                        logger.warning(f"Expert {expert} failed: {e}")
                        self._update_expert_stats(expert, success=False)
        else:
            # Sequential execution
            for expert in selected_experts:
                try:
                    prediction = self._get_single_expert_prediction(task, expert, kg_patterns)
                    if prediction:
                        predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Expert {expert} failed: {e}")
                    self._update_expert_stats(expert, success=False)
        
        return predictions
    
    def _get_single_expert_prediction(self, 
                                     task: ARCTaskInput,
                                     expert_type: str,
                                     kg_patterns: List[Dict[str, Any]] = None) -> Optional[ExpertPrediction]:
        """Get prediction from a single expert model"""
        start_time = time.time()
        
        try:
            # For now, use mock expert models (in production, load actual fine-tuned models)
            prediction = self._mock_expert_inference(task, expert_type, kg_patterns)
            
            processing_time = time.time() - start_time
            self._update_expert_stats(expert_type, processing_time, success=True)
            
            return ExpertPrediction(
                expert_type=expert_type,
                prediction=prediction['output'],
                confidence=prediction['confidence'],
                reasoning=prediction['reasoning'],
                processing_time=processing_time,
                metadata=prediction.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error in expert {expert_type}: {e}")
            self._update_expert_stats(expert_type, success=False)
            return None
    
    def _mock_expert_inference(self, 
                              task: ARCTaskInput,
                              expert_type: str,
                              kg_patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock expert inference (replace with actual model calls in production)"""
        input_grid = task.test_input
        base_confidence = self.expert_configs[expert_type]['base_confidence']
        
        # Add knowledge graph influence
        kg_boost = 0.0
        if kg_patterns:
            relevant_patterns = [p for p in kg_patterns if p['transformation_type'] == expert_type]
            if relevant_patterns:
                kg_boost = 0.1 * len(relevant_patterns) / len(kg_patterns)
        
        if expert_type == "tiling":
            # Tiling pattern prediction
            output = self._apply_tiling_transformation(input_grid)
            confidence = min(base_confidence + kg_boost, 0.95)
            reasoning = f"Applied {expert_type} transformation based on repeating patterns"
            
        elif expert_type == "symmetry":
            # Symmetry transformation
            output = self._apply_symmetry_transformation(input_grid)
            confidence = min(base_confidence + kg_boost, 0.95)
            reasoning = f"Applied {expert_type} transformation with reflection/rotation"
            
        elif expert_type == "extraction":
            # Object extraction and manipulation
            output = self._apply_extraction_transformation(input_grid)
            confidence = min(base_confidence + kg_boost, 0.95)
            reasoning = f"Applied {expert_type} transformation with object manipulation"
            
        elif expert_type == "mapping":
            # Color/shape mapping
            output = self._apply_mapping_transformation(input_grid)
            confidence = min(base_confidence + kg_boost, 0.95)
            reasoning = f"Applied {expert_type} transformation with color/shape mapping"
            
        else:  # rule
            # Complex rule application
            output = self._apply_rule_transformation(input_grid)
            confidence = min(base_confidence + kg_boost, 0.95)
            reasoning = f"Applied {expert_type} transformation with logical rules"
        
        return {
            'output': output,
            'confidence': confidence,
            'reasoning': reasoning,
            'metadata': {'kg_patterns_used': len(kg_patterns) if kg_patterns else 0}
        }
    
    def _apply_tiling_transformation(self, grid: np.ndarray) -> np.ndarray:
        """Apply tiling transformation"""
        # Create 2x2 tiling pattern
        h, w = grid.shape
        output = np.zeros_like(grid)
        
        # Extract base pattern from top-left
        pattern_size = min(2, h//2, w//2)
        if pattern_size > 0:
            base_pattern = grid[:pattern_size, :pattern_size]
            
            # Tile the pattern
            for i in range(0, h, pattern_size):
                for j in range(0, w, pattern_size):
                    end_i = min(i + pattern_size, h)
                    end_j = min(j + pattern_size, w)
                    output[i:end_i, j:end_j] = base_pattern[:end_i-i, :end_j-j]
        
        return output
    
    def _apply_symmetry_transformation(self, grid: np.ndarray) -> np.ndarray:
        """Apply symmetry transformation"""
        # Random choice of symmetry operation
        operations = [np.fliplr, np.flipud, lambda x: np.rot90(x), lambda x: x.T]
        operation = np.random.choice(operations)
        return operation(grid)
    
    def _apply_extraction_transformation(self, grid: np.ndarray) -> np.ndarray:
        """Apply extraction transformation"""
        # Simple object extraction: increment non-zero values
        output = grid.copy()
        mask = grid > 0
        output[mask] = (grid[mask] + 1) % 10
        return output
    
    def _apply_mapping_transformation(self, grid: np.ndarray) -> np.ndarray:
        """Apply mapping transformation"""
        # Color mapping transformation
        color_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}
        output = grid.copy()
        for old_color, new_color in color_map.items():
            output[grid == old_color] = new_color
        return output
    
    def _apply_rule_transformation(self, grid: np.ndarray) -> np.ndarray:
        """Apply rule-based transformation"""
        # Complex rule: rotate and modify based on position
        rotated = np.rot90(grid)
        output = (rotated + np.arange(grid.shape[0])[:, None]) % 10
        return output
    
    def _update_expert_stats(self, expert_type: str, processing_time: float = 0.0, success: bool = True):
        """Update performance statistics for an expert"""
        stats = self.expert_stats[expert_type]
        stats['calls'] += 1
        
        if success:
            # Update average processing time
            stats['avg_time'] = ((stats['avg_time'] * (stats['calls'] - 1)) + processing_time) / stats['calls']
        
        # Update success rate (with exponential decay for recent performance)
        decay_factor = 0.9
        stats['success_rate'] = decay_factor * stats['success_rate'] + (1 - decay_factor) * (1.0 if success else 0.0)
    
    def get_expert_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all experts"""
        return self.expert_stats.copy()

class InferencePipelineCache:
    """Caching system for inference pipeline"""
    
    def __init__(self, cache_dir: str = "./cache", max_cache_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from disk"""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_path = self.cache_dir / "cache_index.json"
        with open(index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _compute_task_hash(self, task: ARCTaskInput) -> str:
        """Compute hash for task caching"""
        # Create hash from task content
        content = {
            'train_examples': [(inp.tolist(), out.tolist()) for inp, out in task.train_examples],
            'test_input': task.test_input.tolist()
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def get_cached_result(self, task: ARCTaskInput) -> Optional[ARCTaskOutput]:
        """Get cached result for task"""
        task_hash = self._compute_task_hash(task)
        
        if task_hash in self.cache_index:
            cache_file = self.cache_dir / f"{task_hash}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Reconstruct ARCTaskOutput
                    return ARCTaskOutput(
                        task_id=cached_data['task_id'],
                        prediction=np.array(cached_data['prediction']),
                        confidence=cached_data['confidence'],
                        expert_contributions=cached_data['expert_contributions'],
                        reasoning=cached_data['reasoning'],
                        processing_time=cached_data['processing_time'],
                        pipeline_metadata=cached_data['pipeline_metadata']
                    )
                except Exception as e:
                    logger.warning(f"Error loading cached result: {e}")
        
        return None
    
    def cache_result(self, task: ARCTaskInput, result: ARCTaskOutput):
        """Cache result for task"""
        if len(self.cache_index) >= self.max_cache_size:
            # Remove oldest entry
            oldest_task = min(self.cache_index.items(), key=lambda x: x[1]['timestamp'])[0]
            self._remove_cached_result(oldest_task)
        
        task_hash = self._compute_task_hash(task)
        
        # Save result to file
        cache_file = self.cache_dir / f"{task_hash}.json"
        cached_data = {
            'task_id': result.task_id,
            'prediction': result.prediction.tolist(),
            'confidence': result.confidence,
            'expert_contributions': result.expert_contributions,
            'reasoning': result.reasoning,
            'processing_time': result.processing_time,
            'pipeline_metadata': result.pipeline_metadata
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f, indent=2)
        
        # Update index
        self.cache_index[task_hash] = {
            'timestamp': time.time(),
            'task_id': result.task_id
        }
        self._save_cache_index()
    
    def _remove_cached_result(self, task_hash: str):
        """Remove cached result"""
        cache_file = self.cache_dir / f"{task_hash}.json"
        if cache_file.exists():
            cache_file.unlink()
        if task_hash in self.cache_index:
            del self.cache_index[task_hash]

class FullInferencePipeline:
    """Complete end-to-end inference pipeline for ARC tasks"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.kg_retriever = KnowledgeGraphRetriever()
        self.router = TaskRouter()
        self.expert_manager = ExpertModelManager()
        self.aggregation_system = MultiExpertAggregationSystem()
        
        # Initialize caching if enabled
        if self.config.enable_caching:
            self.cache = InferencePipelineCache(self.config.cache_dir)
        else:
            self.cache = None
        
        # Set aggregation strategy
        self._set_aggregation_strategy()
        
        # Performance tracking
        self.pipeline_stats = {
            'total_tasks': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'success_rate': 1.0
        }
        
        logger.info("Full inference pipeline initialized")
    
    def _set_aggregation_strategy(self):
        """Set the aggregation strategy based on config"""
        if self.config.aggregation_strategy == "bayesian":
            strategy = BayesianModelAveraging()
        elif self.config.aggregation_strategy == "consensus":
            strategy = ConsensusAggregator()
        else:
            strategy = WeightedVotingAggregator(
                confidence_threshold=self.config.confidence_threshold
            )
        
        self.aggregation_system.set_strategy(strategy)
    
    def predict(self, task: ARCTaskInput) -> ARCTaskOutput:
        """Main prediction method for ARC tasks"""
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get_cached_result(task)
            if cached_result:
                self.pipeline_stats['cache_hits'] += 1
                self.pipeline_stats['total_tasks'] += 1
                logger.info(f"Cache hit for task {task.task_id}")
                return cached_result
        
        try:
            # Step 1: Knowledge Graph Retrieval
            kg_patterns = self.kg_retriever.retrieve_similar_patterns(
                task, 
                self.config.kg_similarity_threshold,
                self.config.max_similar_patterns
            )
            logger.info(f"Retrieved {len(kg_patterns)} similar patterns from KG")
            
            # Step 2: Task Routing
            selected_experts = self.router.route_task(
                task, kg_patterns, self.config.max_experts_per_task
            )
            logger.info(f"Router selected experts: {selected_experts}")
            
            # Step 3: Expert Inference
            expert_predictions = self.expert_manager.get_expert_predictions(
                task, selected_experts, kg_patterns, self.config.parallel_expert_inference
            )
            logger.info(f"Got {len(expert_predictions)} expert predictions")
            
            if not expert_predictions:
                raise ValueError("No expert predictions available")
            
            # Step 4: Prediction Aggregation
            aggregated_result = self.aggregation_system.aggregate_predictions(
                expert_predictions,
                task_context={
                    'task_id': task.task_id,
                    'kg_patterns': kg_patterns,
                    'selected_experts': selected_experts
                }
            )
            
            # Step 5: Create final output
            processing_time = time.time() - start_time
            
            result = ARCTaskOutput(
                task_id=task.task_id,
                prediction=aggregated_result.final_prediction,
                confidence=aggregated_result.confidence,
                expert_contributions=aggregated_result.expert_contributions,
                reasoning=aggregated_result.reasoning,
                processing_time=processing_time,
                pipeline_metadata={
                    'kg_patterns_count': len(kg_patterns),
                    'selected_experts': selected_experts,
                    'consensus_score': aggregated_result.consensus_score,
                    'uncertainty_score': aggregated_result.uncertainty_score,
                    'aggregation_strategy': self.config.aggregation_strategy
                }
            )
            
            # Cache result if enabled
            if self.cache:
                self.cache.cache_result(task, result)
            
            # Update pipeline statistics
            self._update_pipeline_stats(processing_time, success=True)
            
            logger.info(f"Pipeline completed for task {task.task_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Pipeline failed for task {task.task_id}: {e}")
            
            # Create error result with fallback prediction
            fallback_prediction = self._generate_fallback_prediction(task)
            
            result = ARCTaskOutput(
                task_id=task.task_id,
                prediction=fallback_prediction,
                confidence=0.1,  # Low confidence for fallback
                expert_contributions={'fallback': 1.0},
                reasoning=f"Pipeline failed: {str(e)}. Using fallback prediction.",
                processing_time=processing_time,
                pipeline_metadata={'error': str(e), 'fallback_used': True}
            )
            
            self._update_pipeline_stats(processing_time, success=False)
            return result
    
    def _generate_fallback_prediction(self, task: ARCTaskInput) -> np.ndarray:
        """Generate fallback prediction when pipeline fails"""
        # Simple fallback: return input grid or create basic transformation
        if task.train_examples:
            # Use output from first training example as template
            _, first_output = task.train_examples[0]
            if first_output.shape == task.test_input.shape:
                return first_output
        
        # Last resort: return modified input
        return (task.test_input + 1) % 10
    
    def _update_pipeline_stats(self, processing_time: float, success: bool):
        """Update pipeline performance statistics"""
        self.pipeline_stats['total_tasks'] += 1
        
        if success:
            # Update average processing time
            total = self.pipeline_stats['total_tasks']
            current_avg = self.pipeline_stats['avg_processing_time']
            self.pipeline_stats['avg_processing_time'] = ((current_avg * (total - 1)) + processing_time) / total
        
        # Update success rate with exponential decay
        decay_factor = 0.95
        current_rate = self.pipeline_stats['success_rate']
        self.pipeline_stats['success_rate'] = decay_factor * current_rate + (1 - decay_factor) * (1.0 if success else 0.0)
    
    def batch_predict(self, tasks: List[ARCTaskInput]) -> List[ARCTaskOutput]:
        """Batch prediction for multiple tasks"""
        results = []
        
        logger.info(f"Starting batch prediction for {len(tasks)} tasks")
        
        for i, task in enumerate(tasks):
            logger.info(f"Processing task {i+1}/{len(tasks)}: {task.task_id}")
            result = self.predict(task)
            results.append(result)
        
        logger.info(f"Batch prediction completed. Success rate: {self.get_pipeline_stats()['success_rate']:.3f}")
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = self.pipeline_stats.copy()
        stats['expert_statistics'] = self.expert_manager.get_expert_statistics()
        
        if self.cache:
            stats['cache_size'] = len(self.cache.cache_index)
            stats['cache_hit_rate'] = stats['cache_hits'] / max(stats['total_tasks'], 1)
        
        return stats
    
    def close(self):
        """Clean up resources"""
        self.kg_retriever.close()
        logger.info("Pipeline resources cleaned up")

def create_demo_task() -> ARCTaskInput:
    """Create a demo ARC task for testing the pipeline"""
    # Pattern recognition task: color inversion
    train_input_1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    train_output_1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    train_input_2 = np.array([[2, 0, 2], [0, 2, 0], [2, 0, 2]]) 
    train_output_2 = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])
    
    test_input = np.array([[3, 0, 3], [0, 3, 0], [3, 0, 3]])
    
    return ARCTaskInput(
        train_examples=[(train_input_1, train_output_1), (train_input_2, train_output_2)],
        test_input=test_input,
        task_id="demo_pipeline_001"
    )

def main():
    """Demo of the full inference pipeline"""
    print("🚀 Full ARC Inference Pipeline Demo")
    print("=" * 40)
    
    # Initialize pipeline with configuration
    config = PipelineConfig(
        aggregation_strategy="weighted_voting",
        enable_caching=True,
        parallel_expert_inference=True
    )
    
    pipeline = FullInferencePipeline(config)
    
    # Create demo task
    task = create_demo_task()
    
    print(f"\\n📋 Demo Task: {task.task_id}")
    print(f"Input shape: {task.test_input.shape}")
    print("Test input:")
    print(task.test_input)
    
    # Run inference
    print("\\n🔄 Running inference pipeline...")
    result = pipeline.predict(task)
    
    print(f"\\n✅ Pipeline Results:")
    print(f"Task ID: {result.task_id}")
    print(f"Prediction:")
    print(result.prediction)
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Processing time: {result.processing_time:.3f}s")
    
    print("\\nExpert contributions:")
    for expert, contribution in result.expert_contributions.items():
        print(f"  - {expert}: {contribution:.3f}")
    
    print(f"\\nReasoning: {result.reasoning}")
    
    # Pipeline statistics
    print("\\n📊 Pipeline Statistics:")
    stats = pipeline.get_pipeline_stats()
    print(f"Total tasks processed: {stats['total_tasks']}")
    print(f"Success rate: {stats['success_rate']:.3f}")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    
    if 'cache_hit_rate' in stats:
        print(f"Cache hit rate: {stats['cache_hit_rate']:.3f}")
    
    print("\\nExpert performance:")
    for expert, expert_stats in stats['expert_statistics'].items():
        print(f"  - {expert}: {expert_stats['calls']} calls, "
              f"{expert_stats['success_rate']:.3f} success rate, "
              f"{expert_stats['avg_time']:.3f}s avg time")
    
    # Test caching with same task
    print("\\n🔄 Testing cache with same task...")
    start_time = time.time()
    cached_result = pipeline.predict(task)
    cache_time = time.time() - start_time
    
    print(f"Cached result time: {cache_time:.6f}s (vs {result.processing_time:.3f}s original)")
    print(f"Results match: {np.array_equal(result.prediction, cached_result.prediction)}")
    
    # Cleanup
    pipeline.close()
    
    print("\\n✅ Full Inference Pipeline Demo Complete!")
    print("   Key capabilities demonstrated:")
    print("   - End-to-end ARC task processing")
    print("   - Knowledge graph pattern retrieval")
    print("   - Intelligent expert routing")
    print("   - Parallel expert inference")
    print("   - Sophisticated prediction aggregation")
    print("   - Result caching and performance tracking")
    print("   - Comprehensive error handling and fallbacks")

if __name__ == "__main__":
    main()