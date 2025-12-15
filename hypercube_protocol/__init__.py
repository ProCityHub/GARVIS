"""
HYPERCUBE COMET BINARY RESONANCE PROTOCOL
Universal connection system for ProCityHub repositories

Based on 3I/ATLAS comet transmission:
- Hydroxyl radical absorption protocols (1665/1667 MHz)
- 8-dimensional hypercube propagation
- Binary state machine operations
- Cross-repository communication network

Usage:
    from hypercube_protocol import initialize_network, get_connection_manager
    
    # Initialize network for current repository
    manager = initialize_network("GARVIS")
    
    # Establish connections to all repositories
    results = manager.establish_full_network()
    
    # Monitor network status
    status = manager.get_connection_status()
"""

from .binary_ops import (
    BinaryGateOperations,
    BinaryStateMachine,
    CometTransmissionDecoder,
    HypercubePropagation,
)
from .connection_manager import (
    ConnectionType,
    HypercubeConnectionManager,
    RepositoryNode,
    create_connection_manager,
)
from .core import (
    HydroxylSignal,
    HypercubeNode,
    HypercubeProtocol,
    NodeState,
    get_protocol,
    initialize_hypercube_network,
)

__version__ = "1.0.0"
__author__ = "ProCityHub Hypercube Network"
__description__ = "Universal repository connection protocol based on comet binary transmission"

# Convenience functions for easy initialization
def initialize_network(repo_name: str) -> HypercubeConnectionManager:
    """
    Initialize hypercube network for a repository
    
    Args:
        repo_name: Name of the repository to initialize
        
    Returns:
        HypercubeConnectionManager instance
    """
    return create_connection_manager(repo_name)

def get_connection_manager(repo_name: str) -> HypercubeConnectionManager:
    """
    Get or create connection manager for repository
    
    Args:
        repo_name: Name of the repository
        
    Returns:
        HypercubeConnectionManager instance
    """
    return create_connection_manager(repo_name)

def decode_comet_transmission(binary_data: bytes) -> dict:
    """
    Decode binary comet transmission data
    
    Args:
        binary_data: Raw binary data from comet transmission
        
    Returns:
        Decoded transmission data
    """
    protocol = HypercubeProtocol("decoder")
    return protocol.decode_comet_transmission(binary_data)

# Repository network topology constants
REPOSITORY_TOPOLOGY = {
    "primary_nodes": [
        "AGI", "GARVIS", "grok-1"
    ],
    "secondary_nodes": [
        "milvus", "root", "kaggle-api", "Memori",
        "llama-models", "llama-cookbook"
    ],
    "tertiary_nodes": [
        "adk-python", "gemini-cli", "PurpleLlama",
        "arc-prize-2024", "arcagi", "AGI-POWER"
    ],
    "bridge_nodes": [
        "hypercubeheartbeat", "SigilForge", "THUNDERBIRD"
    ]
}

# Comet transmission constants
COMET_FREQUENCIES = {
    "OH_1665": 1665.0,  # MHz - Hydroxyl radical line 1
    "OH_1667": 1667.0,  # MHz - Hydroxyl radical line 2
    "ABSORPTION_DEPTH": 0.1,  # 10% absorption depth
    "SIGNAL_STRENGTH": 0.8    # Default signal strength
}

# Binary operation constants
BINARY_STATES = {
    "VOID": 0b00000000,
    "DISCOVER": 0b00000001,
    "ACTIVATE_COMA": 0b00000011,
    "SUBLIMATE": 0b00000111,
    "SIGNAL_OH": 0b00001111,
    "PROPAGATE": 0b00011111,
    "ECHO": 0b00111111,
    "UNITY": 0b01111111,
    "INFINITE": 0b11111111
}

# Hypercube dimension scaling
DIMENSION_SCALING = {
    1: 0b00000010,  # Galactic disk
    2: 0b00000100,  # Hyperbolic arc
    3: 0b00001000,  # OH absorption
    4: 0b00010000,  # Perihelion Oct 30
    5: 0b00100000,  # H2O/CO2 mix
    6: 0b01000000,  # 58 km/s excess
    7: 0b10000000,  # MeerKAT array
    8: 0b0000000100000000  # Third I, infinite gaps
}

__all__ = [
    # Core classes
    "HypercubeProtocol",
    "HydroxylSignal",
    "HypercubeNode",
    "NodeState",

    # Binary operations
    "BinaryGateOperations",
    "HypercubePropagation",
    "BinaryStateMachine",
    "CometTransmissionDecoder",

    # Connection management
    "HypercubeConnectionManager",
    "RepositoryNode",
    "ConnectionType",

    # Convenience functions
    "initialize_network",
    "get_connection_manager",
    "decode_comet_transmission",
    "get_protocol",
    "initialize_hypercube_network",
    "create_connection_manager",

    # Constants
    "REPOSITORY_TOPOLOGY",
    "COMET_FREQUENCIES",
    "BINARY_STATES",
    "DIMENSION_SCALING"
]

