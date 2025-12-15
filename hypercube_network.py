"""
Hypercube Network Implementation
Extracted from GARVIS Issue #2 - "Binary code"

This module implements a hypercube network with binary operations for
signal propagation and n-dimensional scaling. Based on pure binary mathematics
using bit operations and exponential scaling.

The hypercube network uses Hamming distance = 1 for edge connectivity,
allowing efficient signal propagation through all nodes in n hops.
"""

import time
from typing import Any


class HypercubeNetwork:
    """
    N-dimensional hypercube network with binary signal propagation

    Core Hypercube Protocol:
    - n-dimensional cube = 2^n nodes
    - Edge connectivity: Hamming distance = 1
    - Signal propagation: XOR-based bit flipping
    - Complete propagation: n hops for n-dimensional cube
    """

    def __init__(self, dimensions: int = 3):
        """
        Initialize hypercube network

        Args:
            dimensions: Number of dimensions (default: 3D = 8 nodes)
        """
        self.dimensions = dimensions
        self.num_nodes = 2 ** dimensions
        self.nodes = {}
        self.edges: list[tuple[int, int]] = []

        # Initialize all nodes as OFF (00000000)
        for i in range(self.num_nodes):
            self.nodes[i] = 0  # Node state: 0 = OFF, 1 = ON

        # Generate edges based on Hamming distance = 1
        self._generate_edges()

        # Propagation history
        self.propagation_history: list[dict[str, Any]] = []

    def _generate_edges(self) -> None:
        """Generate edges between nodes with Hamming distance = 1"""
        self.edges = []

        for i in range(self.num_nodes):
            for bit in range(self.dimensions):
                # Flip bit to get neighbor
                neighbor = i ^ (1 << bit)  # XOR with bit mask
                if neighbor < self.num_nodes:
                    edge = tuple(sorted([i, neighbor]))
                    if edge not in self.edges:
                        self.edges.append(edge)

    def get_neighbors(self, node: int) -> list[int]:
        """
        Get all neighbors of a node (Hamming distance = 1)

        Args:
            node: Node index

        Returns:
            List of neighbor node indices
        """
        neighbors = []
        for bit in range(self.dimensions):
            neighbor = node ^ (1 << bit)  # Flip bit
            if neighbor < self.num_nodes:
                neighbors.append(neighbor)
        return neighbors

    def activate_node(self, node: int) -> None:
        """Activate a node (set to ON)"""
        if 0 <= node < self.num_nodes:
            self.nodes[node] = 1

    def deactivate_node(self, node: int) -> None:
        """Deactivate a node (set to OFF)"""
        if 0 <= node < self.num_nodes:
            self.nodes[node] = 0

    def reset_network(self) -> None:
        """Reset all nodes to OFF state"""
        for i in range(self.num_nodes):
            self.nodes[i] = 0
        self.propagation_history = []

    def propagate_signal(self, source: int = 0, signal: int = 1) -> dict[str, Any]:
        """
        Propagate signal through hypercube network

        Args:
            source: Source node index (default: 0)
            signal: Signal value (default: 1 = ON)

        Returns:
            Dictionary with propagation results
        """
        self.reset_network()
        self.propagation_history = []

        # Step 0: Activate source
        self.nodes[source] = signal
        visited = {source}
        current_wave = {source}
        step = 0

        self.propagation_history.append({
            'step': step,
            'active_nodes': list(current_wave),
            'total_active': len([n for n in self.nodes.values() if n == signal]),
            'node_states': dict(self.nodes)
        })

        # Propagate until all nodes are reached
        while len(visited) < self.num_nodes:
            step += 1
            next_wave = set()

            # Spread from current wave to neighbors
            for node in current_wave:
                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        self.nodes[neighbor] = signal
                        next_wave.add(neighbor)
                        visited.add(neighbor)

            if not next_wave:  # No more nodes to activate
                break

            current_wave = next_wave

            self.propagation_history.append({
                'step': step,
                'active_nodes': list(current_wave),
                'total_active': len([n for n in self.nodes.values() if n == signal]),
                'node_states': dict(self.nodes)
            })

        return {
            'total_steps': step,
            'nodes_reached': len(visited),
            'complete_propagation': len(visited) == self.num_nodes,
            'expected_steps': self.dimensions,
            'efficiency': len(visited) / self.num_nodes,
            'propagation_history': self.propagation_history
        }

    def recursive_propagate(self, node: int, visited: set[int], signal: int = 1) -> set[int]:
        """
        Recursive signal propagation function

        Args:
            node: Current node
            visited: Set of visited nodes
            signal: Signal value

        Returns:
            Set of all visited nodes
        """
        if node in visited:
            return visited

        visited.add(node)
        self.nodes[node] = signal

        # Propagate to all neighbors
        neighbors = self.get_neighbors(node)
        for neighbor in neighbors:
            visited = self.recursive_propagate(neighbor, visited, signal)

        return visited

    def get_binary_representation(self, node: int) -> str:
        """
        Get binary representation of node index

        Args:
            node: Node index

        Returns:
            Binary string representation
        """
        return format(node, f'0{self.dimensions}b')

    def print_network_state(self) -> None:
        """Print current network state in binary format"""
        print(f"\nHypercube Network State ({self.dimensions}D, {self.num_nodes} nodes):")
        print("=" * 50)

        for i in range(self.num_nodes):
            binary = self.get_binary_representation(i)
            state = "ON " if self.nodes[i] == 1 else "OFF"
            print(f"Node {i:2d}: {binary} = {state}")

    def print_edges(self) -> None:
        """Print all edges in the hypercube"""
        print(f"\nHypercube Edges ({len(self.edges)} total):")
        print("=" * 40)

        for i, (node1, node2) in enumerate(self.edges):
            bin1 = self.get_binary_representation(node1)
            bin2 = self.get_binary_representation(node2)
            print(f"Edge {i+1:2d}: {node1}({bin1}) <-> {node2}({bin2})")

    def demonstrate_propagation(self, source: int = 0) -> None:
        """
        Demonstrate signal propagation with visual output

        Args:
            source: Source node for propagation
        """
        print("\n🔥 HYPERCUBE SIGNAL PROPAGATION DEMONSTRATION")
        print(f"Dimensions: {self.dimensions}D | Nodes: {self.num_nodes} | Source: {source}")
        print("=" * 60)

        result = self.propagate_signal(source)

        for step_data in self.propagation_history:
            step = step_data['step']
            active = step_data['active_nodes']
            total = step_data['total_active']

            print(f"\nSTEP {step}:")
            print(f"  Newly activated: {active}")
            print(f"  Total active nodes: {total}/{self.num_nodes}")

            # Show binary states
            active_binary = [self.get_binary_representation(n) for n in active]
            print(f"  Binary: {active_binary}")

            # Visual progress bar
            progress = "█" * total + "░" * (self.num_nodes - total)
            print(f"  Progress: [{progress}] {total/self.num_nodes*100:.1f}%")

            time.sleep(0.5)  # Pause for visualization

        print("\n✅ PROPAGATION COMPLETE!")
        print(f"Total steps: {result['total_steps']} (Expected: {result['expected_steps']})")
        print(f"Efficiency: {result['efficiency']*100:.1f}%")
        print(f"All nodes reached: {result['complete_propagation']}")


class BinaryOperations:
    """
    Binary operations for hypercube network
    """

    @staticmethod
    def xor_gate(a: int, b: int) -> int:
        """XOR gate (flip): 0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0"""
        return a ^ b

    @staticmethod
    def and_gate(a: int, b: int) -> int:
        """AND gate (mask): 0∧0=0, 0∧1=0, 1∧0=0, 1∧1=1"""
        return a & b

    @staticmethod
    def or_gate(a: int, b: int) -> int:
        """OR gate (merge): 0∨0=0, 0∨1=1, 1∨0=1, 1∨1=1"""
        return a | b

    @staticmethod
    def not_gate(a: int) -> int:
        """NOT gate (invert): ¬0=1, ¬1=0"""
        return 1 - a  # For single bit

    @staticmethod
    def hamming_distance(a: int, b: int) -> int:
        """Calculate Hamming distance between two integers"""
        return bin(a ^ b).count('1')

    @staticmethod
    def flip_bit(number: int, bit_position: int) -> int:
        """Flip a specific bit in a number"""
        return number ^ (1 << bit_position)

    @staticmethod
    def get_bit(number: int, bit_position: int) -> int:
        """Get value of specific bit"""
        return (number >> bit_position) & 1


class HypercubeScaling:
    """
    N-dimensional hypercube scaling calculations
    """

    @staticmethod
    def calculate_nodes(dimensions: int) -> int:
        """Calculate number of nodes: 2^n"""
        return 2 ** dimensions

    @staticmethod
    def calculate_edges(dimensions: int) -> int:
        """Calculate number of edges: n * 2^(n-1)"""
        return dimensions * (2 ** (dimensions - 1))

    @staticmethod
    def scaling_table(max_dimensions: int = 8) -> None:
        """Print scaling table for different dimensions"""
        print("\n📊 HYPERCUBE SCALING TABLE")
        print("=" * 40)
        print("Dim | Nodes | Edges | Memory")
        print("-" * 40)

        for n in range(1, max_dimensions + 1):
            nodes = HypercubeScaling.calculate_nodes(n)
            edges = HypercubeScaling.calculate_edges(n)
            memory = f"{nodes * 4}B"  # Assuming 4 bytes per node

            print(f"{n:2d}D | {nodes:5d} | {edges:5d} | {memory:>6s}")


# Example usage and demonstration
if __name__ == "__main__":
    print("🔥 HYPERCUBE NETWORK BINARY PROTOCOL")
    print("=" * 50)

    # Create 3D hypercube (8 nodes)
    cube = HypercubeNetwork(dimensions=3)

    # Show network structure
    cube.print_network_state()
    cube.print_edges()

    # Demonstrate propagation
    cube.demonstrate_propagation(source=0)

    # Show scaling table
    HypercubeScaling.scaling_table()

    # Binary operations demonstration
    print("\n🔧 BINARY OPERATIONS")
    print("=" * 30)
    ops = BinaryOperations()

    print("XOR Gate (FLIP):")
    for a in [0, 1]:
        for b in [0, 1]:
            result = ops.xor_gate(a, b)
            print(f"  {a} ⊕ {b} = {result}")

    print("\nHamming Distance Examples:")
    examples = [(0b000, 0b001), (0b000, 0b111), (0b101, 0b110)]
    for a, b in examples:
        distance = ops.hamming_distance(a, b)
        print(f"  {a:03b} ↔ {b:03b} = {distance}")

    print("\n" + "="*50)
    print("PURE BINARY TRANSMISSION:")
    print("01010000 01010010 01001111 01010000 01000001 01000111 01000001 01010100 01000101")
    print("Translation: PROPAGATE THROUGH ALL NODES")
    print("\nThis is pure binary mathematics—hypercube network propagation")
    print("using bit operations and exponential scaling. 🔥")
