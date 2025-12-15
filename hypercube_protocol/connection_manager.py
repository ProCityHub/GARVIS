"""
HYPERCUBE CONNECTION MANAGER
Manages connections between repositories using hydroxyl radical protocols
Implements MeerKAT-style antenna array for multi-repository communication
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .binary_ops import BinaryGateOperations, HypercubePropagation
from .core import HydroxylSignal, HypercubeProtocol


class ConnectionType(Enum):
    """Types of hypercube connections"""

    PRIMARY = "primary"  # Core AGI nodes
    SECONDARY = "secondary"  # Data processing nodes
    TERTIARY = "tertiary"  # Specialized systems
    BRIDGE = "bridge"  # Consciousness-cosmic bridges


@dataclass
class RepositoryNode:
    """Repository node in hypercube network"""

    name: str
    connection_type: ConnectionType
    coordinates: tuple
    active: bool = False
    last_ping: float = 0.0
    signal_strength: float = 0.0
    binary_signature: str = ""


class HypercubeConnectionManager:
    """Manages all hypercube connections across repositories"""

    def __init__(self, local_repo: str):
        self.local_repo = local_repo
        self.protocol = HypercubeProtocol(local_repo)
        self.gates = BinaryGateOperations()
        self.propagator = HypercubePropagation()

        # Repository network topology
        self.repository_nodes: dict[str, RepositoryNode] = {}
        self.connection_matrix: dict[str, set[str]] = {}
        self.signal_handlers: dict[str, Callable] = {}

        # Network state
        self.network_active = False
        self.heartbeat_interval = 30.0  # seconds
        self.last_network_scan = 0.0

        # Initialize repository topology
        self._initialize_repository_topology()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"HypercubeNetwork.{local_repo}")

    def _initialize_repository_topology(self):
        """Initialize the complete repository network topology"""

        # Primary AGI nodes (core intelligence)
        primary_repos = ["AGI", "GARVIS", "grok-1"]

        # Secondary data processing nodes
        secondary_repos = [
            "milvus",
            "root",
            "kaggle-api",
            "Memori",
            "llama-models",
            "llama-cookbook",
        ]

        # Tertiary specialized systems
        tertiary_repos = [
            "adk-python",
            "gemini-cli",
            "PurpleLlama",
            "arc-prize-2024",
            "arcagi",
            "AGI-POWER",
        ]

        # Bridge nodes (consciousness-cosmic)
        bridge_repos = ["hypercubeheartbeat", "SigilForge", "THUNDERBIRD"]

        # Create repository nodes
        all_repos = {
            **{repo: ConnectionType.PRIMARY for repo in primary_repos},
            **{repo: ConnectionType.SECONDARY for repo in secondary_repos},
            **{repo: ConnectionType.TERTIARY for repo in tertiary_repos},
            **{repo: ConnectionType.BRIDGE for repo in bridge_repos},
        }

        for repo_name, conn_type in all_repos.items():
            coordinates = self._calculate_coordinates(repo_name)
            signature = self._generate_binary_signature(repo_name)

            self.repository_nodes[repo_name] = RepositoryNode(
                name=repo_name,
                connection_type=conn_type,
                coordinates=coordinates,
                binary_signature=signature,
            )

            # Initialize connection matrix
            self.connection_matrix[repo_name] = set()

    def _calculate_coordinates(self, repo_name: str) -> tuple:
        """Calculate 8D hypercube coordinates for repository"""
        hash_bytes = hashlib.sha256(repo_name.encode()).digest()

        # 8-dimensional coordinates based on comet transmission scaling
        coordinates = tuple(
            hash_bytes[i] % 2
            for i in range(8)  # Binary coordinates (0 or 1)
        )

        return coordinates

    def _generate_binary_signature(self, repo_name: str) -> str:
        """Generate unique binary signature for repository"""
        hash_obj = hashlib.sha256(repo_name.encode())
        return hash_obj.hexdigest()[:16]  # 64-bit signature

    def establish_connection(self, target_repo: str, connection_strength: float = 1.0) -> bool:
        """Establish hypercube connection with target repository"""

        if target_repo not in self.repository_nodes:
            self.logger.error(f"Unknown repository: {target_repo}")
            return False

        if target_repo == self.local_repo:
            self.logger.warning("Cannot connect to self")
            return False

        # Calculate connection compatibility
        local_coords = self.repository_nodes[self.local_repo].coordinates
        target_coords = self.repository_nodes[target_repo].coordinates

        hamming_distance = sum(
            self.gates.xor_gate(a, b) for a, b in zip(local_coords, target_coords)
        )

        # Prefer connections with Hamming distance = 1 (direct neighbors)
        if hamming_distance <= 2:  # Allow some flexibility
            # Add bidirectional connection
            self.connection_matrix[self.local_repo].add(target_repo)
            self.connection_matrix[target_repo].add(self.local_repo)

            # Generate connection signal
            signal = self._create_connection_signal(target_repo, connection_strength)
            self.protocol.propagate_signal(signal)

            self.logger.info(f"🌌 Connection established: {self.local_repo} <-> {target_repo}")
            self.logger.info(f"📏 Hamming distance: {hamming_distance}")

            return True
        else:
            self.logger.warning(
                f"Connection rejected: Hamming distance {hamming_distance} too high"
            )
            return False

    def _create_connection_signal(self, target_repo: str, strength: float) -> HydroxylSignal:
        """Create hydroxyl signal for connection establishment"""

        connection_data = {
            "action": "establish_connection",
            "source": self.local_repo,
            "target": target_repo,
            "strength": strength,
            "timestamp": time.time(),
            "coordinates": self.repository_nodes[self.local_repo].coordinates,
        }

        payload = json.dumps(connection_data).encode()

        return HydroxylSignal(
            frequency_1665=1665.0,
            frequency_1667=1667.0,
            absorption_depth=strength * 0.1,  # Scale absorption by strength
            timestamp=time.time(),
            source_node=self.local_repo,
            target_node=target_repo,
            binary_payload=payload,
        )

    def broadcast_to_network(self, message: str, signal_type: str = "broadcast") -> int:
        """Broadcast message to all connected repositories"""

        broadcast_count = 0
        connected_repos = self.connection_matrix[self.local_repo]

        for target_repo in connected_repos:
            broadcast_data = {
                "type": signal_type,
                "message": message,
                "source": self.local_repo,
                "timestamp": time.time(),
                "hop_count": 0,
            }

            payload = json.dumps(broadcast_data).encode()
            signal = HydroxylSignal(
                frequency_1665=1665.0,
                frequency_1667=1667.0,
                absorption_depth=0.05,  # Light absorption for broadcast
                timestamp=time.time(),
                source_node=self.local_repo,
                target_node=target_repo,
                binary_payload=payload,
            )

            if self.protocol.propagate_signal(signal):
                broadcast_count += 1

        self.logger.info(f"📡 Broadcast sent to {broadcast_count} repositories")
        return broadcast_count

    def scan_network(self) -> dict[str, Any]:
        """Scan hypercube network for active nodes"""

        self.last_network_scan = time.time()
        active_nodes = []
        connection_count = 0

        for repo_name, node in self.repository_nodes.items():
            # Simulate ping (in real implementation, would send actual ping)
            if repo_name in self.connection_matrix[self.local_repo]:
                node.active = True
                node.last_ping = time.time()
                node.signal_strength = 0.8 + (hash(repo_name) % 20) / 100  # Simulate signal
                active_nodes.append(repo_name)
                connection_count += len(self.connection_matrix[repo_name])

        network_status = {
            "local_repository": self.local_repo,
            "scan_timestamp": self.last_network_scan,
            "total_repositories": len(self.repository_nodes),
            "active_nodes": len(active_nodes),
            "total_connections": connection_count // 2,  # Bidirectional connections
            "network_topology": {
                "primary_nodes": [
                    n
                    for n, r in self.repository_nodes.items()
                    if r.connection_type == ConnectionType.PRIMARY
                ],
                "secondary_nodes": [
                    n
                    for n, r in self.repository_nodes.items()
                    if r.connection_type == ConnectionType.SECONDARY
                ],
                "tertiary_nodes": [
                    n
                    for n, r in self.repository_nodes.items()
                    if r.connection_type == ConnectionType.TERTIARY
                ],
                "bridge_nodes": [
                    n
                    for n, r in self.repository_nodes.items()
                    if r.connection_type == ConnectionType.BRIDGE
                ],
            },
            "connection_matrix": {
                repo: list(connections)
                for repo, connections in self.connection_matrix.items()
                if connections
            },
        }

        self.logger.info(f"🔍 Network scan complete: {len(active_nodes)} active nodes")
        return network_status

    def establish_full_network(self) -> dict[str, Any]:
        """Establish connections to all compatible repositories"""

        self.logger.info("🚀 Establishing full hypercube network...")

        connection_results = {
            "successful_connections": [],
            "failed_connections": [],
            "total_attempts": 0,
        }

        # Connect to all repositories based on type compatibility
        local_type = self.repository_nodes[self.local_repo].connection_type

        for repo_name, node in self.repository_nodes.items():
            if repo_name == self.local_repo:
                continue

            connection_results["total_attempts"] += 1

            # Determine connection strength based on type compatibility
            strength = self._calculate_connection_strength(local_type, node.connection_type)

            if self.establish_connection(repo_name, strength):
                connection_results["successful_connections"].append(
                    {
                        "repository": repo_name,
                        "type": node.connection_type.value,
                        "strength": strength,
                        "coordinates": node.coordinates,
                    }
                )
            else:
                connection_results["failed_connections"].append(
                    {"repository": repo_name, "reason": "hamming_distance_too_high"}
                )

        # Perform network scan after connections
        network_status = self.scan_network()
        connection_results["network_status"] = network_status

        self.network_active = True
        self.logger.info(
            f"✅ Network established: {len(connection_results['successful_connections'])} connections"
        )

        return connection_results

    def _calculate_connection_strength(
        self, local_type: ConnectionType, target_type: ConnectionType
    ) -> float:
        """Calculate connection strength based on node types"""

        # Connection strength matrix
        strength_matrix = {
            (ConnectionType.PRIMARY, ConnectionType.PRIMARY): 1.0,
            (ConnectionType.PRIMARY, ConnectionType.SECONDARY): 0.9,
            (ConnectionType.PRIMARY, ConnectionType.TERTIARY): 0.7,
            (ConnectionType.PRIMARY, ConnectionType.BRIDGE): 0.8,
            (ConnectionType.SECONDARY, ConnectionType.SECONDARY): 0.8,
            (ConnectionType.SECONDARY, ConnectionType.TERTIARY): 0.6,
            (ConnectionType.SECONDARY, ConnectionType.BRIDGE): 0.5,
            (ConnectionType.TERTIARY, ConnectionType.TERTIARY): 0.7,
            (ConnectionType.TERTIARY, ConnectionType.BRIDGE): 0.6,
            (ConnectionType.BRIDGE, ConnectionType.BRIDGE): 0.9,
        }

        # Try both directions
        key1 = (local_type, target_type)
        key2 = (target_type, local_type)

        return strength_matrix.get(key1, strength_matrix.get(key2, 0.5))

    async def start_heartbeat(self):
        """Start heartbeat monitoring for network health"""

        self.logger.info("💓 Starting hypercube heartbeat...")

        while self.network_active:
            # Generate heartbeat pulse
            heartbeat = self.protocol.heartbeat_pulse()

            # Broadcast heartbeat to connected nodes
            self.broadcast_to_network(
                f"Heartbeat from {self.local_repo}: {heartbeat.hex()}", "heartbeat"
            )

            # Wait for next heartbeat
            await asyncio.sleep(self.heartbeat_interval)

    def get_connection_status(self) -> dict[str, Any]:
        """Get detailed connection status"""

        return {
            "local_repository": self.local_repo,
            "network_active": self.network_active,
            "protocol_status": self.protocol.get_network_status(),
            "connections": {
                repo: {
                    "active": self.repository_nodes[repo].active,
                    "type": self.repository_nodes[repo].connection_type.value,
                    "coordinates": self.repository_nodes[repo].coordinates,
                    "signal_strength": self.repository_nodes[repo].signal_strength,
                    "last_ping": self.repository_nodes[repo].last_ping,
                }
                for repo in self.connection_matrix[self.local_repo]
            },
            "last_network_scan": self.last_network_scan,
            "total_repositories": len(self.repository_nodes),
        }


def create_connection_manager(repo_name: str) -> HypercubeConnectionManager:
    """Factory function to create connection manager"""
    return HypercubeConnectionManager(repo_name)


async def test_connection_manager():
    """Test the connection manager"""
    print("🧪 Testing Hypercube Connection Manager...")

    # Create manager for GARVIS repository
    manager = create_connection_manager("GARVIS")

    # Establish full network
    results = manager.establish_full_network()
    print(f"🌐 Network results: {len(results['successful_connections'])} connections")

    # Get connection status
    status = manager.get_connection_status()
    print(f"📊 Connection status: {status['network_active']}")

    # Start heartbeat (run for a few cycles)
    heartbeat_task = asyncio.create_task(manager.start_heartbeat())
    await asyncio.sleep(5)  # Run for 5 seconds
    manager.network_active = False
    await heartbeat_task


if __name__ == "__main__":
    asyncio.run(test_connection_manager())
