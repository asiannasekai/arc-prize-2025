#!/bin/bash

echo "🚀 Setting up Federated ARC System Demo"
echo "=================================="

# Check if data files exist
if [ ! -f "arc-agi_training_challenges.json" ]; then
    echo "❌ ARC data files not found. Please make sure you have:"
    echo "  - arc-agi_training_challenges.json"
    echo "  - arc-agi_training_solutions.json"
    echo "  - arc-agi_evaluation_challenges.json"
    echo "  - arc-agi_evaluation_solutions.json"
    echo "  - arc-agi_test_challenges.json"
    echo ""
    echo "These should be in the current directory."
    exit 1
fi

echo "✅ ARC data files found"

# Install required packages
echo "📦 Installing required packages..."
pip install torch transformers sentence-transformers networkx numpy matplotlib scikit-learn

echo "🔬 Running Federated ARC System Demo..."
python demo_federated_arc.py

echo ""
echo "🎉 Demo completed!"
echo ""
echo "Next steps to implement the full system:"
echo "1. Set up Neo4j database for pattern knowledge graph"
echo "2. Fine-tune expert models using the training configurations"
echo "3. Implement proper model aggregation mechanisms"
echo "4. Evaluate on the test set"