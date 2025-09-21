#!/bin/bash
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
