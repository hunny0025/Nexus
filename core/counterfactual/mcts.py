"""NEXUS — Monte Carlo Tree Search engine for intervention selection."""

import math
import random
from typing import Optional


class MCTSNode:
    """A single node in the MCTS search tree.

    Attributes
    ----------
    intervention : dict or None
        The intervention this node represents (None for root).
    parent : MCTSNode or None
    children : list[MCTSNode]
    visits : int
    total_reward : float
    """

    def __init__(self, intervention: Optional[dict] = None, parent=None):
        self.intervention = intervention
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.visits: int = 0
        self.total_reward: float = 0.0

    def ucb1(self, exploration: float = 1.41) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_reward / self.visits
        if self.parent is None or self.parent.visits == 0:
            return exploitation
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term

    def best_child(self, exploration: float = 1.41) -> "MCTSNode":
        """Return the child with highest UCB1 score."""
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return len(self.children) == 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)


class MCTSEngine:
    """Monte Carlo Tree Search for optimal intervention selection.

    Parameters
    ----------
    simulator : InterventionSimulator
        Simulator for running intervention outcomes.
    n_simulations : int
        Number of MCTS simulations to run (default 50).
    exploration : float
        UCB1 exploration constant (default 1.41).
    """

    def __init__(self, simulator, n_simulations: int = 50, exploration: float = 1.41):
        self.simulator = simulator
        self.n_simulations = n_simulations
        self.exploration = exploration

    def _reward(self, outcome: dict) -> float:
        """Compute reward from simulation outcome.

        Formula: 2000 - delay_penalty*0.5 - fuel_penalty*0.1
                      - cascade_penalty*1000 - complexity_penalty*50
        """
        delay_penalty = outcome.get("total_delay_minutes", 0) * 0.5
        fuel_penalty = outcome.get("fuel_delta_kg", 0) * 0.1
        cascade_penalty = outcome.get("cascade_probability_90min", 0) * 1000
        complexity_penalty = outcome.get("intervention_complexity", 1) * 50
        return 2000 - delay_penalty - fuel_penalty - cascade_penalty - complexity_penalty

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1 tree policy."""
        node = root
        while not node.is_leaf():
            node = node.best_child(self.exploration)
        return node

    def _expand(self, node: MCTSNode, intervention_space: list[dict]):
        """Expand a leaf node by adding children for each intervention."""
        for intervention in intervention_space:
            child = MCTSNode(intervention=intervention, parent=node)
            node.children.append(child)

    def _simulate(self, node: MCTSNode, current_state: dict) -> float:
        """Run one simulation and return the reward."""
        if node.intervention is None:
            return 0.0
        outcome = self.simulator.run(current_state, node.intervention)
        return self._reward(outcome)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Propagate reward up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def search(
        self,
        current_state: dict,
        intervention_space: list[dict],
    ) -> list[dict]:
        """Run MCTS and return interventions sorted by expected reward.

        Parameters
        ----------
        current_state : dict
            Current network state.
        intervention_space : list[dict]
            Available interventions.

        Returns
        -------
        list[dict]
            Sorted list of {intervention, expected_reward,
            simulations_run, confidence}
        """
        if not intervention_space:
            return []

        root = MCTSNode()
        self._expand(root, intervention_space)

        for _ in range(self.n_simulations):
            # Select
            leaf = self._select(root)

            # If leaf has been visited, expand and select a child
            if leaf.visits > 0 and leaf.intervention is not None:
                self._expand(leaf, intervention_space)
                if leaf.children:
                    leaf = leaf.children[0]

            # Simulate
            reward = self._simulate(leaf, current_state)

            # Backpropagate
            self._backpropagate(leaf, reward)

        # Collect results from root's direct children
        results = []
        for child in root.children:
            if child.visits == 0:
                continue
            avg = child.avg_reward
            # Confidence based on visit count relative to total
            confidence = min(1.0, child.visits / max(self.n_simulations * 0.3, 1))
            results.append(
                {
                    "intervention": child.intervention,
                    "expected_reward": round(avg, 2),
                    "simulations_run": child.visits,
                    "confidence": round(confidence, 4),
                }
            )

        results.sort(key=lambda r: r["expected_reward"], reverse=True)
        return results
