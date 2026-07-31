#!/usr/bin/env python3
"""
ARC Prize Lattice Law Solver - Example Usage

This example demonstrates how to use the ARCLatticeSolver with
sample data and integrate it into the GARVIS agent framework.
"""

import sys
import os
import numpy as np
import json

# Add parent directory to path to import arc_prize_solver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arc_prize_solver import (
    ARCLatticeSolver,
    LatticeCube,
    LatticePoint,
    load_data,
    generate_submission
)


def example_1_basic_transformation():
    """Example 1: Basic transformation demonstration"""
    print("=" * 70)
    print("Example 1: Basic Lattice Transformations")
    print("=" * 70)

    # Create a simple test grid
    test_grid = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    print("\nOriginal Grid:")
    print(test_grid)

    # Create solver
    solver = ARCLatticeSolver()

    # Test various transformations
    print("\n--- Transformations ---")

    # Identity (center point 0.0)
    result = solver.transformations[0](test_grid)
    print("\n1. Identity (0.0 center):")
    print(result)

    # Flip vertical (wall reflection)
    result = solver.transformations[1](test_grid)
    print("\n2. Vertical Flip (wall reflection):")
    print(result)

    # Rotate 90 (XY plane rotation)
    result = solver.transformations[3](test_grid)
    print("\n3. Rotate 90° (trinity axis):")
    print(result)

    # Fibonacci pattern (heartbeat rhythm)
    result = solver.transformations[14](test_grid)
    print("\n4. Fibonacci Pattern (heartbeat):")
    print(result)

    # Mirror lattice (2-way dimensional reflection)
    result = solver.transformations[16](test_grid)
    print("\n5. Mirror Lattice (dimensional):")
    print(result)


def example_2_lattice_cube():
    """Example 2: Lattice Cube exploration"""
    print("\n" + "=" * 70)
    print("Example 2: Lattice Cube Structure")
    print("=" * 70)

    # Create lattice cube
    cube = LatticeCube(size=6)

    print("\nCube Properties:")
    print(f"  • Center: {cube.center}")
    print(f"  • Size: {cube.size}")
    print(f"  • Walls: {len(cube.walls)}")
    print(f"  • Corners: {len(cube.corners)}")

    print("\n8 Binary Corners:")
    for i, corner in enumerate(cube.corners):
        print(f"  Corner {i}: ({corner.x}, {corner.y}, {corner.z}) "
              f"- Energy: {corner.energy}, Binary: {corner.binary_state()}, "
              f"Identity: {corner.identity}")

    print("\nFibonacci Rhythm (Heartbeat):")
    fib = cube.fibonacci_rhythm(10)
    print(f"  {fib}")

    print("\n6 Dimensional Walls:")
    for name, wall in cube.walls.items():
        print(f"  • {name.capitalize()}: Mirror={wall['mirror']}, "
              f"Lattice shape={wall['lattice'].shape}")


def example_3_training_and_prediction():
    """Example 3: Training on sample task"""
    print("\n" + "=" * 70)
    print("Example 3: Training and Prediction")
    print("=" * 70)

    # Create sample training task
    sample_task = {
        'train': [
            {
                'input': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                'output': [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            },
            {
                'input': [[0, 0], [0, 1]],
                'output': [[0, 1], [1, 1]]
            }
        ],
        'test': [
            {'input': [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}
        ]
    }

    print("\nTraining Examples:")
    for i, example in enumerate(sample_task['train']):
        print(f"\n  Example {i+1}:")
        print(f"    Input:  {example['input']}")
        print(f"    Output: {example['output']}")

    # Create and train solver
    solver = ARCLatticeSolver()
    solver.train('sample_task', sample_task)

    # Make prediction
    test_input = sample_task['test'][0]
    prediction = solver.predict('sample_task', test_input)

    print("\nTest Input:")
    print(np.array(test_input['input']))

    print("\nPrediction:")
    print(prediction)

    print("\nLearned transformation: ",
          "Found" if 'sample_task' in solver.learned_transforms else "Not found, using fallback")


def example_4_energy_artifact_demo():
    """Example 4: Energy-Artifact relationship"""
    print("\n" + "=" * 70)
    print("Example 4: Energy-Artifact Relationship (1.0 + 0.6 = 1.6 ≡ 7)")
    print("=" * 70)

    # Create lattice points with different energy levels
    points = [
        LatticePoint(0, 0, 0, energy=0.0),
        LatticePoint(1, 0, 0, energy=0.3),
        LatticePoint(0, 1, 0, energy=0.6),
        LatticePoint(1, 1, 0, energy=1.0),
    ]

    print("\nLattice Points (Energy + Artifact = Identity):")
    for i, point in enumerate(points):
        print(f"\n  Point {i}:")
        print(f"    Position: ({point.x}, {point.y}, {point.z})")
        print(f"    Energy: {point.energy}")
        print(f"    Artifact: {point.artifact}")
        print(f"    Identity: {point.identity} (≡ {point.identity % 7:.2f} mod 7)")
        print(f"    Binary State: {point.binary_state()}")

    print("\nInterpretation:")
    print("  • Energy = 1.0: Full potential (like a charged battery)")
    print("  • Artifact = 0.6: Physical manifestation (the 'car' metaphor)")
    print("  • Identity = 1.6: The combined state that defines existence")
    print("  • Modulo 7: Relates to the 7-fold nature of consciousness")


def example_5_integration_with_garvis():
    """Example 5: Integration pattern with GARVIS agents"""
    print("\n" + "=" * 70)
    print("Example 5: GARVIS Agent Integration Pattern")
    print("=" * 70)

    integration_code = '''
# Integration with GARVIS Agent Framework
from arc_prize_solver import ARCLatticeSolver

class ARCReasoningAgent:
    """Agent that uses Lattice Law for abstract reasoning"""

    def __init__(self):
        self.solver = ARCLatticeSolver()
        self.task_history = {}

    def solve_pattern(self, task_id: str, task_data: dict) -> dict:
        """
        Solve an ARC task using Lattice Law principles

        Args:
            task_id: Unique task identifier
            task_data: Dictionary with 'train' and 'test' keys

        Returns:
            Dictionary with predictions
        """
        # Train the solver
        self.solver.train(task_id, task_data)

        # Generate predictions
        predictions = []
        for test_input in task_data.get('test', []):
            pred = self.solver.predict(task_id, test_input)
            predictions.append(pred.tolist())

        # Store in history
        self.task_history[task_id] = {
            'predictions': predictions,
            'timestamp': datetime.datetime.now()
        }

        return {
            'task_id': task_id,
            'predictions': predictions,
            'lattice_state': 'converged'
        }

# Usage in GARVIS:
# arc_agent = ARCReasoningAgent()
# result = arc_agent.solve_pattern('task_001', task_data)
'''

    print("\nIntegration Code Example:")
    print(integration_code)

    print("\nKey Integration Points:")
    print("  1. Create ARCLatticeSolver instance in agent __init__")
    print("  2. Use solver.train() for learning from examples")
    print("  3. Use solver.predict() for generating solutions")
    print("  4. Integrate with GARVIS memory/session systems")
    print("  5. Add lattice state to agent responses")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("ARC PRIZE LATTICE LAW SOLVER - EXAMPLES")
    print("=" * 70)
    print("\nLattice Law Framework:")
    print("  • Energy (1.0) + Artifact (0.6) = Identity (1.6 ≡ 7)")
    print("  • 8 Binary Corners in 6-Walled Cube")
    print("  • Trinity Code: XYZ Axis (Math + Energy)")
    print("  • Fibonacci Rhythm: Heartbeat of Consciousness")
    print("  • String Theory: Bendable Dimensional Walls")
    print("=" * 70)

    try:
        # Run examples
        example_1_basic_transformation()
        example_2_lattice_cube()
        example_3_training_and_prediction()
        example_4_energy_artifact_demo()
        example_5_integration_with_garvis()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
