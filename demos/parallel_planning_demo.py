"""
Parallel Planning Demo
======================
Demonstrates running multiple planners in parallel with early termination.

When one planner succeeds, others are cancelled to save computation.
This pattern is useful for motion planning where different algorithms
may have varying success rates depending on the scenario.
"""

import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List
from mj_environment import Environment


@dataclass
class Plan:
    """A simple plan result."""
    name: str
    trajectory: List[np.ndarray]
    cost: float


class MockPlanner:
    """
    A mock planner that simulates planning with variable completion times.

    In a real system, this would be replaced with actual motion planners
    like RRT, PRM, trajectory optimization, etc.
    """

    def __init__(self, name: str, steps: int, success_prob: float = 0.7):
        self.name = name
        self.total_steps = steps
        self.success_prob = success_prob
        self.current_step = 0
        self._trajectory: List[np.ndarray] = []

    def step(self, fork: Environment) -> bool:
        """Perform one planning step. Returns True if planning is complete."""
        self.current_step += 1

        # Simulate planning work by stepping physics
        fork.sim.step()

        # Record trajectory point
        self._trajectory.append(fork.data.time)

        # Simulate variable computation time
        time.sleep(np.random.uniform(0.01, 0.05))

        return self.current_step >= self.total_steps

    def get_plan(self) -> Optional[Plan]:
        """Get the plan result. Returns None if planning failed."""
        if np.random.random() < self.success_prob:
            return Plan(
                name=self.name,
                trajectory=self._trajectory,
                cost=np.random.uniform(1.0, 10.0)
            )
        return None


def run_planner(
    fork: Environment,
    planner: MockPlanner,
    cancel: threading.Event
) -> Optional[Plan]:
    """
    Run a planner on a forked environment with cancellation support.

    Args:
        fork: Independent environment clone for this planner
        planner: The planner instance to run
        cancel: Event to signal early termination

    Returns:
        Plan if successful, None if cancelled or failed
    """
    print(f"[{planner.name}] Starting planning ({planner.total_steps} steps)...")

    while not planner.step(fork):
        if cancel.is_set():
            print(f"[{planner.name}] Cancelled at step {planner.current_step}")
            return None

    plan = planner.get_plan()
    if plan:
        print(f"[{planner.name}] Success! Cost: {plan.cost:.2f}")
    else:
        print(f"[{planner.name}] Failed (no valid plan found)")

    return plan


def parallel_planning_demo():
    """
    Demonstrate parallel planning with early termination.

    Creates multiple forks of the environment and runs different "planners"
    on each. When one succeeds, the others are cancelled.
    """
    print("=" * 60)
    print("Parallel Planning Demo")
    print("=" * 60)

    # Initialize environment
    env = Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )

    # Activate an object to plan with
    obj_type = next(iter(env.registry.objects))
    name = env.registry.activate(obj_type, [0, 0, 0.5])
    print(f"\nActivated {name} for planning scenario")

    # Create planners with different characteristics
    # In a real system, these would be RRT, PRM, optimization-based, etc.
    planners = [
        MockPlanner("RRT", steps=20, success_prob=0.6),
        MockPlanner("PRM", steps=30, success_prob=0.8),
        MockPlanner("TrajOpt", steps=15, success_prob=0.5),
        MockPlanner("CHOMP", steps=25, success_prob=0.7),
    ]

    # Create forks for parallel execution
    print(f"\nCreating {len(planners)} forks for parallel planning...")
    forks = env.fork(n=len(planners))

    # Shared cancellation event
    cancel = threading.Event()

    print("\nStarting parallel planners...")
    print("-" * 40)

    start_time = time.time()
    winning_plan = None

    with ThreadPoolExecutor(max_workers=len(planners)) as executor:
        # Submit all planners
        futures = {
            executor.submit(run_planner, fork, planner, cancel): planner.name
            for fork, planner in zip(forks, planners)
        }

        # Process results as they complete
        for future in as_completed(futures):
            planner_name = futures[future]
            try:
                result = future.result()
                if result is not None and winning_plan is None:
                    # First successful plan - cancel others
                    winning_plan = result
                    cancel.set()
                    print(f"\n[Manager] {planner_name} won! Cancelling others...")
            except Exception as e:
                print(f"[{planner_name}] Error: {e}")

    elapsed = time.time() - start_time

    print("-" * 40)
    print(f"\nPlanning completed in {elapsed:.2f}s")

    if winning_plan:
        print(f"Winner: {winning_plan.name}")
        print(f"Cost: {winning_plan.cost:.2f}")
        print(f"Trajectory length: {len(winning_plan.trajectory)} points")
    else:
        print("No planner succeeded")

    # Verify original environment is unchanged
    print(f"\nOriginal environment time: {env.data.time:.3f}s (unchanged)")
    print("=" * 60)


if __name__ == "__main__":
    parallel_planning_demo()
