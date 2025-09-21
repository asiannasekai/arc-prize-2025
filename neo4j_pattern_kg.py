"""
Neo4j Pattern Knowledge Graph for ARC Challenges
Implements the knowledge graph infrastructure for storing and retrieving transformation patterns
"""

import json
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatternSignature:
    """Unique signature for ARC patterns"""
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    size_ratio: float
    color_change: bool
    pattern_hash: str
    complexity_score: float

class TransformationType(Enum):
    TILING = "tiling"
    EXTRACTION = "extraction"
    MAPPING = "mapping"
    SYMMETRY = "symmetry"
    COMPLETION = "completion"
    RULE_BASED = "rule_based"

class Neo4jPatternKG:
    """
    Neo4j-based Pattern Knowledge Graph for ARC challenges
    Stores transformation patterns, rules, and relationships
    """
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="neo4jpassword"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.setup_schema()
    
    def close(self):
        self.driver.close()
    
    def setup_schema(self):
        """Create Neo4j schema for ARC patterns"""
        with self.driver.session() as session:
            # Create constraints and indexes
            constraints = [
                "CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT rule_id_unique IF NOT EXISTS FOR (r:Rule) REQUIRE r.id IS UNIQUE",
                "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
                "CREATE INDEX pattern_type_idx IF NOT EXISTS FOR (p:Pattern) ON (p.type)",
                "CREATE INDEX rule_type_idx IF NOT EXISTS FOR (r:Rule) ON (r.type)",
                "CREATE INDEX size_ratio_idx IF NOT EXISTS FOR (p:Pattern) ON (p.size_ratio)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint/index: {constraint.split()[1]}")
                except Exception as e:
                    logger.warning(f"Constraint/index creation failed: {e}")
    
    def compute_pattern_signature(self, task_data: Dict) -> PatternSignature:
        """Compute unique signature for a pattern"""
        train_examples = task_data['train']
        
        # Analyze first training example
        input_grid = np.array(train_examples[0]['input'])
        output_grid = np.array(train_examples[0]['output'])
        
        input_shape = input_grid.shape
        output_shape = output_grid.shape
        size_ratio = (output_shape[0] * output_shape[1]) / (input_shape[0] * input_shape[1])
        
        # Check if colors change
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        color_change = input_colors != output_colors
        
        # Compute pattern hash
        pattern_str = f"{input_shape}_{output_shape}_{sorted(input_colors)}_{sorted(output_colors)}"
        pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:12]
        
        # Complexity score based on grid size and color variety
        complexity_score = (input_shape[0] * input_shape[1] + 
                          output_shape[0] * output_shape[1] + 
                          len(input_colors) + len(output_colors)) / 100.0
        
        return PatternSignature(
            input_shape=input_shape,
            output_shape=output_shape,
            size_ratio=size_ratio,
            color_change=color_change,
            pattern_hash=pattern_hash,
            complexity_score=complexity_score
        )
    
    def extract_transformation_rule(self, task_data: Dict) -> Dict[str, Any]:
        """Extract transformation rule from task examples"""
        train_examples = task_data['train']
        
        transformation_rule = {
            'type': 'unknown',
            'parameters': {},
            'examples': len(train_examples)
        }
        
        if len(train_examples) > 0:
            input_grid = np.array(train_examples[0]['input'])
            output_grid = np.array(train_examples[0]['output'])
            
            # Analyze transformation type
            if output_grid.shape[0] > input_grid.shape[0] or output_grid.shape[1] > input_grid.shape[1]:
                if np.array_equal(input_grid, output_grid[:input_grid.shape[0], :input_grid.shape[1]]):
                    transformation_rule['type'] = 'tiling'
                    transformation_rule['parameters'] = {
                        'scale_x': output_grid.shape[1] // input_grid.shape[1],
                        'scale_y': output_grid.shape[0] // input_grid.shape[0]
                    }
            
            elif output_grid.shape == input_grid.shape:
                if not np.array_equal(input_grid, output_grid):
                    unique_input = len(np.unique(input_grid))
                    unique_output = len(np.unique(output_grid))
                    
                    if unique_input == unique_output:
                        transformation_rule['type'] = 'mapping'
                        transformation_rule['parameters'] = {
                            'color_mapping': True,
                            'shape_preserved': True
                        }
                    else:
                        transformation_rule['type'] = 'rule_based'
                        transformation_rule['parameters'] = {
                            'rule_complexity': 'high'
                        }
            
            elif output_grid.shape[0] < input_grid.shape[0] or output_grid.shape[1] < input_grid.shape[1]:
                transformation_rule['type'] = 'extraction'
                transformation_rule['parameters'] = {
                    'extraction_ratio': (output_grid.shape[0] * output_grid.shape[1]) / 
                                      (input_grid.shape[0] * input_grid.shape[1])
                }
        
        return transformation_rule
    
    def encode_grid_pattern(self, grid: np.ndarray) -> List[float]:
        """Encode grid pattern as vector embedding"""
        # Convert grid to string representation
        grid_str = ' '.join([' '.join(map(str, row)) for row in grid])
        
        # Add spatial features
        height, width = grid.shape
        unique_colors = len(np.unique(grid))
        density = np.count_nonzero(grid) / (height * width)
        
        feature_str = f"grid {height}x{width} colors_{unique_colors} density_{density:.2f} pattern_{grid_str}"
        
        # Generate embedding
        embedding = self.encoder.encode([feature_str])[0]
        return embedding.tolist()
    
    def add_pattern(self, task_id: str, task_data: Dict, transformation_type: TransformationType):
        """Add a pattern to the knowledge graph"""
        with self.driver.session() as session:
            try:
                # Compute pattern signature
                signature = self.compute_pattern_signature(task_data)
                transformation_rule = self.extract_transformation_rule(task_data)
                
                # Encode pattern
                input_grid = np.array(task_data['train'][0]['input'])
                output_grid = np.array(task_data['train'][0]['output'])
                input_embedding = self.encode_grid_pattern(input_grid)
                output_embedding = self.encode_grid_pattern(output_grid)
                
                # Create pattern node
                pattern_query = """
                MERGE (p:Pattern {id: $task_id})
                SET p.type = $pattern_type,
                    p.input_shape = $input_shape,
                    p.output_shape = $output_shape,
                    p.size_ratio = $size_ratio,
                    p.color_change = $color_change,
                    p.pattern_hash = $pattern_hash,
                    p.complexity_score = $complexity_score,
                    p.input_embedding = $input_embedding,
                    p.output_embedding = $output_embedding,
                    p.examples_count = $examples_count
                RETURN p
                """
                
                session.run(pattern_query, 
                          task_id=task_id,
                          pattern_type=transformation_type.value,
                          input_shape=list(signature.input_shape),
                          output_shape=list(signature.output_shape),
                          size_ratio=signature.size_ratio,
                          color_change=signature.color_change,
                          pattern_hash=signature.pattern_hash,
                          complexity_score=signature.complexity_score,
                          input_embedding=input_embedding,
                          output_embedding=output_embedding,
                          examples_count=len(task_data['train']))
                
                # Create transformation rule node
                rule_id = f"rule_{task_id}"
                rule_query = """
                MERGE (r:Rule {id: $rule_id})
                SET r.type = $rule_type,
                    r.parameters = $parameters
                RETURN r
                """
                
                session.run(rule_query,
                          rule_id=rule_id,
                          rule_type=transformation_rule['type'],
                          parameters=json.dumps(transformation_rule['parameters']))
                
                # Create relationship
                relationship_query = """
                MATCH (p:Pattern {id: $task_id}), (r:Rule {id: $rule_id})
                MERGE (p)-[:USES_RULE]->(r)
                """
                
                session.run(relationship_query, task_id=task_id, rule_id=rule_id)
                
                logger.info(f"Added pattern {task_id} of type {transformation_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to add pattern {task_id}: {e}")
    
    def find_similar_patterns(self, query_signature: PatternSignature, 
                            transformation_type: Optional[TransformationType] = None,
                            limit: int = 5) -> List[Dict]:
        """Find similar patterns based on signature"""
        with self.driver.session() as session:
            # Build query
            where_clauses = [
                "ABS(p.size_ratio - $size_ratio) < 2.0",
                "ABS(p.complexity_score - $complexity_score) < 0.5"
            ]
            
            if transformation_type:
                where_clauses.append("p.type = $pattern_type")
            
            query = f"""
            MATCH (p:Pattern)-[:USES_RULE]->(r:Rule)
            WHERE {' AND '.join(where_clauses)}
            RETURN p.id as pattern_id, 
                   p.type as pattern_type,
                   p.size_ratio as size_ratio,
                   p.complexity_score as complexity_score,
                   r.type as rule_type,
                   r.parameters as rule_parameters
            ORDER BY ABS(p.size_ratio - $size_ratio) + ABS(p.complexity_score - $complexity_score)
            LIMIT $limit
            """
            
            params = {
                'size_ratio': query_signature.size_ratio,
                'complexity_score': query_signature.complexity_score,
                'limit': limit
            }
            
            if transformation_type:
                params['pattern_type'] = transformation_type.value
            
            result = session.run(query, **params)
            
            similar_patterns = []
            for record in result:
                similar_patterns.append({
                    'pattern_id': record['pattern_id'],
                    'pattern_type': record['pattern_type'],
                    'size_ratio': record['size_ratio'],
                    'complexity_score': record['complexity_score'],
                    'rule_type': record['rule_type'],
                    'rule_parameters': json.loads(record['rule_parameters']) if record['rule_parameters'] else {}
                })
            
            return similar_patterns
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about patterns in the knowledge graph"""
        with self.driver.session() as session:
            stats_query = """
            MATCH (p:Pattern)
            RETURN p.type as pattern_type, 
                   count(*) as count,
                   avg(p.size_ratio) as avg_size_ratio,
                   avg(p.complexity_score) as avg_complexity
            ORDER BY count DESC
            """
            
            result = session.run(stats_query)
            
            statistics = {
                'total_patterns': 0,
                'by_type': {},
                'overall_stats': {}
            }
            
            total_count = 0
            for record in result:
                pattern_type = record['pattern_type']
                count = record['count']
                total_count += count
                
                statistics['by_type'][pattern_type] = {
                    'count': count,
                    'avg_size_ratio': record['avg_size_ratio'],
                    'avg_complexity': record['avg_complexity']
                }
            
            statistics['total_patterns'] = total_count
            
            # Overall statistics
            overall_query = """
            MATCH (p:Pattern)
            RETURN avg(p.size_ratio) as avg_size_ratio,
                   avg(p.complexity_score) as avg_complexity,
                   min(p.size_ratio) as min_size_ratio,
                   max(p.size_ratio) as max_size_ratio
            """
            
            overall_result = session.run(overall_query).single()
            if overall_result:
                statistics['overall_stats'] = {
                    'avg_size_ratio': overall_result['avg_size_ratio'],
                    'avg_complexity': overall_result['avg_complexity'],
                    'min_size_ratio': overall_result['min_size_ratio'],
                    'max_size_ratio': overall_result['max_size_ratio']
                }
            
            return statistics

def create_neo4j_setup_script():
    """Create setup script for Neo4j"""
    setup_script = """#!/bin/bash
# Neo4j Setup Script for ARC Pattern KG

echo "🗄️ Setting up Neo4j for ARC Pattern Knowledge Graph"

# Set Neo4j home
export NEO4J_HOME=/workspaces/arc-prize-2025/neo4j-community-5.15.0

# Configure Neo4j
echo "Configuring Neo4j..."
cd $NEO4J_HOME

# Set initial password
echo "Setting initial password..."
./bin/neo4j-admin dbms set-initial-password neo4j

# Configure memory settings for development
echo "Configuring memory settings..."
cat >> conf/neo4j.conf << EOF

# Memory settings for ARC Pattern KG
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=1g
dbms.memory.pagecache.size=256m

# Network settings
dbms.default_listen_address=0.0.0.0
dbms.connector.bolt.listen_address=:7687
dbms.connector.http.listen_address=:7474

# APOC plugin settings
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

EOF

echo "✅ Neo4j configured successfully"
echo "🚀 Starting Neo4j..."

# Start Neo4j in background
./bin/neo4j start

echo "Waiting for Neo4j to start..."
sleep 10

# Check if Neo4j is running
if ./bin/neo4j status | grep -q "running"; then
    echo "✅ Neo4j is running successfully"
    echo "🌐 Web interface: http://localhost:7474"
    echo "🔌 Bolt endpoint: bolt://localhost:7687"
    echo "👤 Username: neo4j"
    echo "🔑 Password: neo4j"
else
    echo "❌ Failed to start Neo4j"
    ./bin/neo4j console
fi
"""
    
    with open('/workspaces/arc-prize-2025/setup_neo4j.sh', 'w') as f:
        f.write(setup_script)
    
    import os
    os.chmod('/workspaces/arc-prize-2025/setup_neo4j.sh', 0o755)
    
    print("📝 Created Neo4j setup script: setup_neo4j.sh")

if __name__ == "__main__":
    # Create setup script
    create_neo4j_setup_script()
    
    # Test the pattern KG (will fail if Neo4j not running)
    try:
        kg = Neo4jPatternKG()
        print("✅ Successfully connected to Neo4j")
        kg.close()
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("Run './setup_neo4j.sh' to start Neo4j first")