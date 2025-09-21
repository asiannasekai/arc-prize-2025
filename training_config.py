"""
Training configuration for federated ARC experts
Based on QLoRA fine-tuning approach from the paper
"""

import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class ExpertConfig:
    """Configuration for individual expert training"""
    name: str
    base_model: str
    pattern_types: List[str]
    training_data_size: int
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_steps: int = 1000
    max_length: int = 512
    
@dataclass
class SystemConfig:
    """Overall system configuration"""
    # Model configurations
    router_model: str = "microsoft/Phi-3-mini-4k-instruct"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Training settings
    output_dir: str = "./arc_experts"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    
    # Knowledge graph settings
    kg_backend: str = "networkx"  # or "neo4j" for production
    embedding_dim: int = 384
    
    # Inference settings
    max_experts_per_task: int = 3
    confidence_threshold: float = 0.7
    consensus_threshold: float = 0.6

def create_expert_configs() -> Dict[str, ExpertConfig]:
    """Create training configurations for each expert"""
    
    configs = {}
    
    # Pattern-specific experts
    pattern_experts = [
        ("tiling", ["microsoft/Phi-3-mini-4k-instruct"], ["tiling"], 60),
        ("extraction", ["microsoft/Phi-3-mini-4k-instruct"], ["extraction"], 236),
        ("mapping", ["microsoft/Phi-3-mini-4k-instruct"], ["mapping"], 231),
        ("symmetry", ["microsoft/Phi-3-mini-4k-instruct"], ["symmetry"], 32),
        ("completion", ["microsoft/Phi-3-mini-4k-instruct"], ["completion"], 22),
        ("rule_based", ["microsoft/Phi-3-mini-4k-instruct"], ["rule_based"], 419),
    ]
    
    for name, base_model, pattern_types, data_size in pattern_experts:
        configs[f"{name}_expert"] = ExpertConfig(
            name=f"{name}_expert",
            base_model=base_model[0],
            pattern_types=pattern_types,
            training_data_size=data_size,
            max_steps=min(1000, data_size * 2)  # Scale training steps with data
        )
    
    # Size-specific experts
    size_experts = [
        ("micro", ["microsoft/Phi-3-mini-4k-instruct"], ["small_grids"], 200),
        ("standard", ["microsoft/Phi-3-mini-4k-instruct"], ["medium_grids"], 600),
        ("macro", ["microsoft/Phi-3-mini-4k-instruct"], ["large_grids"], 200),
    ]
    
    for name, base_model, grid_types, data_size in size_experts:
        configs[f"{name}_expert"] = ExpertConfig(
            name=f"{name}_expert",
            base_model=base_model[0],
            pattern_types=grid_types,
            training_data_size=data_size,
            max_length=1024 if name == "macro" else 512  # Larger context for big grids
        )
    
    # Router configuration
    configs["router"] = ExpertConfig(
        name="router",
        base_model="microsoft/Phi-3-mini-4k-instruct",
        pattern_types=["classification"],
        training_data_size=1000,  # All training tasks for classification
        max_length=256,  # Shorter context for classification
        max_steps=2000
    )
    
    return configs

def save_configs(output_path: str = "training_configs.yaml"):
    """Save all configurations to YAML file"""
    
    system_config = SystemConfig()
    expert_configs = create_expert_configs()
    
    config_dict = {
        "system": asdict(system_config),
        "experts": {name: asdict(config) for name, config in expert_configs.items()}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Configurations saved to {output_path}")
    return config_dict

def load_configs(config_path: str = "training_configs.yaml") -> Dict:
    """Load configurations from YAML file"""
    
    try:
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        return configs
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Creating default configs...")
        return save_configs(config_path)

if __name__ == "__main__":
    # Generate and save configurations
    configs = save_configs()
    
    print("\\n=== SYSTEM CONFIGURATION ===")
    system_config = configs["system"]
    for key, value in system_config.items():
        print(f"{key}: {value}")
    
    print("\\n=== EXPERT CONFIGURATIONS ===")
    for expert_name, expert_config in configs["experts"].items():
        print(f"\\n{expert_name}:")
        print(f"  Base model: {expert_config['base_model']}")
        print(f"  Training data size: {expert_config['training_data_size']}")
        print(f"  Max steps: {expert_config['max_steps']}")
        print(f"  Pattern types: {expert_config['pattern_types']}")