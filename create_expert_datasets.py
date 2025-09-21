"""
Expert Training Dataset Creator
Creates specialized instruction datasets for fine-tuning each expert model
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pickle
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpertInstruction:
    """Single instruction for expert training"""
    instruction: str
    input_context: str
    expected_output: str
    metadata: Dict[str, Any]

class ExpertDatasetCreator:
    """Creates training datasets for each pattern expert"""
    
    def __init__(self):
        self.expert_templates = self._create_instruction_templates()
        self.datasets = defaultdict(list)
    
    def _create_instruction_templates(self) -> Dict[str, List[str]]:
        """Create instruction templates for each expert type"""
        return {
            'tiling': [
                "Given this {input_height}x{input_width} pattern, tile it to create a {output_height}x{output_width} grid:",
                "Repeat this pattern to fill a {output_height}x{output_width} space:",
                "Scale this {input_height}x{input_width} pattern by a factor to create the output:",
                "Tile this base pattern multiple times to match the target size:",
                "Create a repeating pattern using this grid as the base unit:"
            ],
            'extraction': [
                "Extract the key pattern from this {input_height}x{input_width} grid:",
                "Identify and isolate the main shape from this complex grid:",
                "Filter out the noise and extract the core pattern:",
                "From this large grid, extract the essential {output_height}x{output_width} pattern:",
                "Simplify this grid by extracting only the important elements:"
            ],
            'mapping': [
                "Transform the colors in this grid according to the mapping rule:",
                "Apply color transformation to convert the input to the target output:",
                "Map each color value to its corresponding output value:",
                "Change colors following the pattern shown in the examples:",
                "Apply the learned color mapping rule to this grid:"
            ],
            'symmetry': [
                "Apply symmetry transformation to this grid:",
                "Rotate, flip, or reflect this pattern as needed:",
                "Transform this grid using geometric operations:",
                "Apply the symmetry rule shown in the examples:",
                "Perform geometric transformation to match the target pattern:"
            ],
            'completion': [
                "Complete this partial pattern:",
                "Fill in the missing parts of this grid:",
                "Extend this pattern to complete the full grid:",
                "Continue the pattern to fill the entire space:",
                "Complete the grid following the established pattern:"
            ],
            'rule_based': [
                "Apply the logical rule to transform this grid:",
                "Follow the conditional logic to generate the output:",
                "Use the pattern rule to convert input to output:",
                "Apply the learned transformation rule:",
                "Transform this grid according to the established rule:"
            ]
        }
    
    def grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert grid to string representation"""
        return '\\n'.join([' '.join(map(str, row)) for row in grid])
    
    def analyze_transformation(self, input_grid: List[List[int]], 
                             output_grid: List[List[int]]) -> Dict[str, Any]:
        """Analyze the transformation between input and output"""
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        analysis = {
            'input_shape': [int(x) for x in input_array.shape],
            'output_shape': [int(x) for x in output_array.shape],
            'size_ratio': float((output_array.shape[0] * output_array.shape[1]) / 
                               (input_array.shape[0] * input_array.shape[1])),
            'input_colors': [int(x) for x in np.unique(input_array)],
            'output_colors': [int(x) for x in np.unique(output_array)],
            'color_mapping': {},
            'transformation_type': 'unknown'
        }
        
        # Detect transformation type
        if output_array.shape == input_array.shape:
            if np.array_equal(input_array, output_array):
                analysis['transformation_type'] = 'identity'
            elif set(analysis['input_colors']) != set(analysis['output_colors']):
                analysis['transformation_type'] = 'color_mapping'
            else:
                analysis['transformation_type'] = 'geometric'
        elif analysis['size_ratio'] > 1:
            analysis['transformation_type'] = 'expansion'
        elif analysis['size_ratio'] < 1:
            analysis['transformation_type'] = 'reduction'
        
        return analysis
    
    def create_tiling_instructions(self, task_data: Dict, task_id: str) -> List[ExpertInstruction]:
        """Create instructions for tiling expert"""
        instructions = []
        
        for i, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            analysis = self.analyze_transformation(input_grid, output_grid)
            
            # Choose appropriate template
            template = np.random.choice(self.expert_templates['tiling'])
            
            instruction = template.format(
                input_height=analysis['input_shape'][0],
                input_width=analysis['input_shape'][1],
                output_height=analysis['output_shape'][0],
                output_width=analysis['output_shape'][1]
            )
            
            input_context = self.grid_to_string(input_grid)
            expected_output = self.grid_to_string(output_grid)
            
            metadata = {
                'task_id': task_id,
                'example_id': i,
                'expert_type': 'tiling',
                'analysis': analysis
            }
            
            instructions.append(ExpertInstruction(
                instruction=instruction,
                input_context=input_context,
                expected_output=expected_output,
                metadata=metadata
            ))
        
        return instructions
    
    def create_extraction_instructions(self, task_data: Dict, task_id: str) -> List[ExpertInstruction]:
        """Create instructions for extraction expert"""
        instructions = []
        
        for i, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            analysis = self.analyze_transformation(input_grid, output_grid)
            
            template = np.random.choice(self.expert_templates['extraction'])
            
            instruction = template.format(
                input_height=analysis['input_shape'][0],
                input_width=analysis['input_shape'][1],
                output_height=analysis['output_shape'][0],
                output_width=analysis['output_shape'][1]
            )
            
            input_context = self.grid_to_string(input_grid)
            expected_output = self.grid_to_string(output_grid)
            
            metadata = {
                'task_id': task_id,
                'example_id': i,
                'expert_type': 'extraction',
                'analysis': analysis
            }
            
            instructions.append(ExpertInstruction(
                instruction=instruction,
                input_context=input_context,
                expected_output=expected_output,
                metadata=metadata
            ))
        
        return instructions
    
    def create_mapping_instructions(self, task_data: Dict, task_id: str) -> List[ExpertInstruction]:
        """Create instructions for mapping expert"""
        instructions = []
        
        for i, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            analysis = self.analyze_transformation(input_grid, output_grid)
            
            # Detect color mapping
            input_array = np.array(input_grid)
            output_array = np.array(output_grid)
            
            color_mapping = {}
            if input_array.shape == output_array.shape:
                for inp_val, out_val in zip(input_array.flatten(), output_array.flatten()):
                    inp_val = int(inp_val)
                    out_val = int(out_val)
                    if inp_val not in color_mapping:
                        color_mapping[inp_val] = out_val
                    elif color_mapping[inp_val] != out_val:
                        # Inconsistent mapping, might be rule-based
                        color_mapping = {}
                        break
            
            template = np.random.choice(self.expert_templates['mapping'])
            instruction = template
            
            # Add color mapping context if detected
            if color_mapping:
                mapping_str = ', '.join([f"{k}→{v}" for k, v in color_mapping.items()])
                instruction += f" Color mapping: {mapping_str}"
            
            input_context = self.grid_to_string(input_grid)
            expected_output = self.grid_to_string(output_grid)
            
            metadata = {
                'task_id': task_id,
                'example_id': i,
                'expert_type': 'mapping',
                'analysis': analysis,
                'color_mapping': color_mapping
            }
            
            instructions.append(ExpertInstruction(
                instruction=instruction,
                input_context=input_context,
                expected_output=expected_output,
                metadata=metadata
            ))
        
        return instructions
    
    def create_symmetry_instructions(self, task_data: Dict, task_id: str) -> List[ExpertInstruction]:
        """Create instructions for symmetry expert"""
        instructions = []
        
        for i, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            analysis = self.analyze_transformation(input_grid, output_grid)
            
            template = np.random.choice(self.expert_templates['symmetry'])
            instruction = template
            
            input_context = self.grid_to_string(input_grid)
            expected_output = self.grid_to_string(output_grid)
            
            metadata = {
                'task_id': task_id,
                'example_id': i,
                'expert_type': 'symmetry',
                'analysis': analysis
            }
            
            instructions.append(ExpertInstruction(
                instruction=instruction,
                input_context=input_context,
                expected_output=expected_output,
                metadata=metadata
            ))
        
        return instructions
    
    def create_completion_instructions(self, task_data: Dict, task_id: str) -> List[ExpertInstruction]:
        """Create instructions for completion expert"""
        instructions = []
        
        for i, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            analysis = self.analyze_transformation(input_grid, output_grid)
            
            template = np.random.choice(self.expert_templates['completion'])
            instruction = template
            
            input_context = self.grid_to_string(input_grid)
            expected_output = self.grid_to_string(output_grid)
            
            metadata = {
                'task_id': task_id,
                'example_id': i,
                'expert_type': 'completion',
                'analysis': analysis
            }
            
            instructions.append(ExpertInstruction(
                instruction=instruction,
                input_context=input_context,
                expected_output=expected_output,
                metadata=metadata
            ))
        
        return instructions
    
    def create_rule_based_instructions(self, task_data: Dict, task_id: str) -> List[ExpertInstruction]:
        """Create instructions for rule-based expert"""
        instructions = []
        
        for i, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            analysis = self.analyze_transformation(input_grid, output_grid)
            
            template = np.random.choice(self.expert_templates['rule_based'])
            instruction = template
            
            input_context = self.grid_to_string(input_grid)
            expected_output = self.grid_to_string(output_grid)
            
            metadata = {
                'task_id': task_id,
                'example_id': i,
                'expert_type': 'rule_based',
                'analysis': analysis
            }
            
            instructions.append(ExpertInstruction(
                instruction=instruction,
                input_context=input_context,
                expected_output=expected_output,
                metadata=metadata
            ))
        
        return instructions
    
    def process_expert_dataset(self, expert_type: str, task_ids: List[str], 
                             training_data: Dict) -> List[ExpertInstruction]:
        """Process all tasks for a specific expert"""
        
        all_instructions = []
        
        logger.info(f"Processing {len(task_ids)} tasks for {expert_type}")
        
        for task_id in tqdm(task_ids, desc=f"Creating {expert_type} instructions"):
            if task_id not in training_data:
                logger.warning(f"Task {task_id} not found in training data")
                continue
            
            task_data = training_data[task_id]
            
            if expert_type == 'tiling_expert':
                instructions = self.create_tiling_instructions(task_data, task_id)
            elif expert_type == 'extraction_expert':
                instructions = self.create_extraction_instructions(task_data, task_id)
            elif expert_type == 'mapping_expert':
                instructions = self.create_mapping_instructions(task_data, task_id)
            elif expert_type == 'symmetry_expert':
                instructions = self.create_symmetry_instructions(task_data, task_id)
            elif expert_type == 'completion_expert':
                instructions = self.create_completion_instructions(task_data, task_id)
            elif expert_type == 'rule_expert':
                instructions = self.create_rule_based_instructions(task_data, task_id)
            else:
                logger.error(f"Unknown expert type: {expert_type}")
                continue
            
            all_instructions.extend(instructions)
        
        return all_instructions
    
    def save_expert_dataset(self, expert_type: str, instructions: List[ExpertInstruction]):
        """Save expert dataset in multiple formats"""
        
        # Create directory for expert datasets
        expert_dir = Path("expert_datasets")
        expert_dir.mkdir(exist_ok=True)
        
        # Save as JSON for human readability
        json_data = []
        for instruction in instructions:
            json_data.append({
                'instruction': instruction.instruction,
                'input': instruction.input_context,
                'output': instruction.expected_output,
                'metadata': instruction.metadata
            })
        
        json_path = expert_dir / f"{expert_type}_dataset.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save as pickle for fast loading
        pickle_path = expert_dir / f"{expert_type}_dataset.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(instructions, f)
        
        # Save in Hugging Face format for fine-tuning
        hf_data = []
        for instruction in instructions:
            # Format for instruction tuning
            text = f"### Instruction:\\n{instruction.instruction}\\n\\n### Input:\\n{instruction.input_context}\\n\\n### Response:\\n{instruction.expected_output}"
            
            hf_data.append({
                'text': text,
                'input': instruction.input_context,
                'instruction': instruction.instruction,
                'output': instruction.expected_output
            })
        
        hf_path = expert_dir / f"{expert_type}_hf_dataset.json"
        with open(hf_path, 'w') as f:
            json.dump(hf_data, f, indent=2)
        
        logger.info(f"✅ Saved {expert_type} dataset:")
        logger.info(f"   JSON: {json_path} ({len(json_data)} instructions)")
        logger.info(f"   Pickle: {pickle_path}")
        logger.info(f"   HuggingFace: {hf_path}")

def main():
    """Main function to create all expert datasets"""
    
    print("🎯 Creating Expert Training Datasets")
    print("=" * 40)
    
    # Load expert task assignments
    with open('arc_expert_datasets.json', 'r') as f:
        expert_datasets = json.load(f)
    
    # Load training data
    with open('arc-agi_training_challenges.json', 'r') as f:
        training_data = json.load(f)
    
    # Initialize dataset creator
    creator = ExpertDatasetCreator()
    
    # Process each expert
    total_instructions = 0
    
    for expert_type, task_ids in expert_datasets.items():
        if not task_ids:
            logger.info(f"Skipping {expert_type} - no tasks assigned")
            continue
        
        print(f"\\n📚 Processing {expert_type}...")
        print(f"   Tasks: {len(task_ids)}")
        
        # Create instructions
        instructions = creator.process_expert_dataset(expert_type, task_ids, training_data)
        
        # Save dataset
        creator.save_expert_dataset(expert_type, instructions)
        
        total_instructions += len(instructions)
        print(f"   Instructions created: {len(instructions)}")
    
    print(f"\\n🎉 Expert Dataset Creation Complete!")
    print(f"📊 Summary:")
    print(f"   Total experts: {len([e for e in expert_datasets.keys() if expert_datasets[e]])}")
    print(f"   Total instructions: {total_instructions}")
    print(f"   Average per expert: {total_instructions // len([e for e in expert_datasets.keys() if expert_datasets[e]])}")
    
    # Create overview file
    overview = {
        'total_instructions': total_instructions,
        'experts': {}
    }
    
    for expert_type, task_ids in expert_datasets.items():
        if task_ids:
            instructions_count = len(task_ids) * 3  # Rough estimate
            overview['experts'][expert_type] = {
                'tasks': len(task_ids),
                'estimated_instructions': instructions_count
            }
    
    with open('expert_datasets_overview.json', 'w') as f:
        json.dump(overview, f, indent=2)
    
    print(f"\\n📋 Files created:")
    print(f"   expert_datasets/ directory with individual expert datasets")
    print(f"   expert_datasets_overview.json")
    
    print(f"\\n🚀 Ready for expert model fine-tuning!")

if __name__ == "__main__":
    main()