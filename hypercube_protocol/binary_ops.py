"""
BINARY OPERATIONS FOR HYPERCUBE COMET TRANSMISSION
Implements XOR, AND, OR gates and recursive spread functions
Based on 3I/ATLAS comet binary resonance patterns
"""

import struct
import time
from typing import Dict, List, Optional, Set, Tuple


class BinaryGateOperations:
    """Binary gate operations for hypercube signal processing"""

    @staticmethod
    def xor_gate(a: int, b: int) -> int:
        """
        XOR GATE (FLIP):
        Sunlight XOR H2O = OH + H  // Radical birth
        0 XOR 1 = 1  // Ice to gas
        1 XOR 0 = 1  // Natural vs myth
        1 XOR 1 = 0  // Anomalies explained
        """
        return a ^ b

    @staticmethod
    def and_gate(a: int, b: int) -> int:
        """
        AND GATE (MASK):
        Comet AND Radio = Absorption  // 1665/1667 MHz lines
        1 AND 1 = 1  // MeerKAT masks the truth
        """
        return a & b

    @staticmethod
    def or_gate(a: int, b: int) -> int:
        """
        OR GATE (MERGE):
        Interstellar OR Solar = Shared chemistry  // CO2 rich, CN depleted
        0 OR 1 = 1  // Life's ingredients merge
        """
        return a | b

    @staticmethod
    def not_gate(a: int) -> int:
        """
        NOT GATE (INVERT):
        NOT Alien = Natural  // Avi's probe? Inverted to ice wanderer
        NOT 1 = 0  // No hostile recon, just cosmic breath
        """
        return ~a & 0xFF  # Keep within 8-bit range

class HypercubePropagation:
    """Hypercube propagation algorithms for network spread"""

    def __init__(self):
        self.visited_nodes: Set[str] = set()
        self.propagation_log: List[Dict] = []

    def propagate_recursive(self, comet_node: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """
        RECURSIVE SPREAD FUNCTION
        
        FUNCTION PROPAGATE(comet, visited):
          IF comet IN visited:
            RETURN "Echo from stars, no more"
          
          visited = visited OR (00000001 << comet)
          NODE[comet] = 00000001
          
          FOR bit IN [00000000, 00000001, 00000010]:  // OH lines
            neighbor = comet XOR (00000001 << bit)
            PROPAGATE(neighbor, visited)
          
          RETURN visited
        """
        if visited is None:
            visited = set()

        if comet_node in visited:
            return visited  # Echo from stars, no more

        # Mark node as visited
        visited.add(comet_node)

        # Log propagation step
        self.propagation_log.append({
            "node": comet_node,
            "timestamp": time.time(),
            "action": "activate",
            "state": "00000001"
        })

        # Propagate to neighbors (Hamming distance = 1)
        for bit_position in [0, 1, 2]:  # OH absorption lines
            neighbor_hash = hash(comet_node) ^ (1 << bit_position)
            neighbor_name = f"node_{neighbor_hash % 1000:03d}"

            if neighbor_name not in visited:
                visited = self.propagate_recursive(neighbor_name, visited)

        return visited

    def n_dimensional_scaling(self, dimensions: int) -> Dict[str, int]:
        """
        N-DIMENSIONAL SCALING
        
        1D: Origin = 00000010 (Galactic disk)
        2D: Path = 00000100 (Hyperbolic arc)  
        3D: Signal = 00001000 (OH absorption)
        4D: Time = 00010000 (Perihelion Oct 30)
        5D: Chem = 00100000 (H2O/CO2 mix)
        6D: Speed = 01000000 (58 km/s excess)
        7D: Scope = 10000000 (MeerKAT array)
        8D: Echo = 0000000100000000 (Third I, infinite gaps)
        
        EDGES = n * 2^(n-1)
        """
        scaling_map = {
            1: 0b00000010,  # Galactic disk
            2: 0b00000100,  # Hyperbolic arc
            3: 0b00001000,  # OH absorption
            4: 0b00010000,  # Perihelion Oct 30
            5: 0b00100000,  # H2O/CO2 mix
            6: 0b01000000,  # 58 km/s excess
            7: 0b10000000,  # MeerKAT array
            8: 0b0000000100000000  # Third I, infinite gaps
        }

        edges = dimensions * (2 ** (dimensions - 1))

        return {
            "dimensions": dimensions,
            "scaling_value": scaling_map.get(dimensions, 0),
            "edge_count": edges,
            "connections": f"{edges} connections: Ice to life"
        }

class BinaryStateMachine:
    """Binary state machine for hypercube nodes"""

    def __init__(self):
        self.current_state = 0b00000000
        self.state_history: List[Tuple[int, float]] = []

    def transition_state(self, new_state: int) -> bool:
        """
        BINARY STATE MACHINE
        
        STATE_0: 00000000 -> 00000001 (DISCOVER)
        STATE_1: 00000001 -> 00000011 (ACTIVATE COMA)
        STATE_2: 00000011 -> 00000111 (SUBLIMATE)
        STATE_3: 00000111 -> 00001111 (SIGNAL OH)
        ...
        STATE_n: 11111111 -> 00000001 (ECHO NATURAL)
        LOOP = 2^n cycles
        UNITY = 00000001 (Water returns, always)
        """
        if self._is_valid_transition(self.current_state, new_state):
            self.state_history.append((self.current_state, time.time()))
            self.current_state = new_state
            return True
        return False

    def _is_valid_transition(self, current: int, new: int) -> bool:
        """Validate state transition according to hypercube rules"""
        # Allow progression through binary states
        if new == (current << 1) | 1:  # Add next bit
            return True
        if new == 0b00000001 and current == 0b11111111:  # Unity cycle
            return True
        return False

    def get_state_name(self, state: int) -> str:
        """Get human-readable state name"""
        state_names = {
            0b00000000: "VOID",
            0b00000001: "DISCOVER",
            0b00000011: "ACTIVATE_COMA",
            0b00000111: "SUBLIMATE",
            0b00001111: "SIGNAL_OH",
            0b00011111: "PROPAGATE",
            0b00111111: "ECHO",
            0b01111111: "UNITY",
            0b11111111: "INFINITE"
        }
        return state_names.get(state, f"STATE_{state:08b}")

class CometTransmissionDecoder:
    """Decoder for pure binary comet transmissions"""

    @staticmethod
    def decode_binary_message(binary_data: bytes) -> str:
        """Decode binary message to ASCII text"""
        try:
            return binary_data.decode('ascii')
        except UnicodeDecodeError:
            return "Binary data (non-ASCII)"

    @staticmethod
    def encode_message_to_binary(message: str) -> bytes:
        """Encode ASCII message to binary"""
        return message.encode('ascii')

    @staticmethod
    def parse_comet_header(binary_data: bytes) -> Dict:
        """Parse comet transmission header"""
        if len(binary_data) < 8:
            return {"error": "Insufficient data for header"}

        return {
            "dimensions": binary_data[0] & 0x07,  # First 3 bits
            "source_type": (binary_data[0] & 0x38) >> 3,  # Next 3 bits
            "signal_strength": binary_data[1],
            "frequency_1665": struct.unpack('>H', binary_data[2:4])[0],
            "frequency_1667": struct.unpack('>H', binary_data[4:6])[0],
            "timestamp": struct.unpack('>H', binary_data[6:8])[0]
        }

    @staticmethod
    def create_hydroxyl_signature() -> bytes:
        """Create hydroxyl radical signature pattern"""
        # OH radical frequencies: 1665 MHz and 1667 MHz
        signature = bytearray()

        # Encode frequencies as binary patterns
        freq_1665 = 1665
        freq_1667 = 1667

        signature.extend(struct.pack('>H', freq_1665))
        signature.extend(struct.pack('>H', freq_1667))

        # Add absorption depth pattern
        absorption_pattern = [0x01, 0x02, 0x04, 0x08]  # Binary progression
        signature.extend(absorption_pattern)

        return bytes(signature)

def test_binary_operations():
    """Test binary operations and propagation"""
    print("🧪 Testing Binary Operations...")

    # Test gates
    gates = BinaryGateOperations()
    print(f"XOR(1,0): {gates.xor_gate(1, 0)} (Ice to gas)")
    print(f"AND(1,1): {gates.and_gate(1, 1)} (MeerKAT truth)")
    print(f"OR(0,1): {gates.or_gate(0, 1)} (Life ingredients merge)")
    print(f"NOT(1): {gates.not_gate(1)} (Natural, not alien)")

    # Test propagation
    propagator = HypercubePropagation()
    visited = propagator.propagate_recursive("3I_ATLAS")
    print(f"🌌 Propagated to {len(visited)} nodes")

    # Test state machine
    state_machine = BinaryStateMachine()
    state_machine.transition_state(0b00000001)  # DISCOVER
    state_machine.transition_state(0b00000011)  # ACTIVATE_COMA
    print(f"🔄 Current state: {state_machine.get_state_name(state_machine.current_state)}")

    # Test decoder
    decoder = CometTransmissionDecoder()
    oh_signature = decoder.create_hydroxyl_signature()
    print(f"💫 OH signature: {oh_signature.hex()}")

if __name__ == "__main__":
    test_binary_operations()

