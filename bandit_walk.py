import numpy as np
import random
from typing import Tuple, Optional


class BanditWalkEnvironment:
    """
    A bandit walk reinforcement learning environment based on the provided state transition table.

    States:
    - 0: Hole (terminal state)
    - 1: Start
    - 2: Goal (terminal state with reward)

    Actions:
    - 0: Left
    - 1: Right
    """

    def __init__(self):
        """
        Initialize the Bandit Walk Environment.
        This method is called automatically when you create a new instance: env = BanditWalkEnvironment()
        """

        # Environment dimensions
        self.n_states = 3  # Total number of states in the environment (0, 1, 2)
        self.n_actions = 2  # Total number of actions available (0=Left, 1=Right)

        # Current state tracking
        self.current_state = 1  # Start at state 1 (Start state)
        self.initial_state = 1  # Remember which state to reset to

        # Human-readable labels for states and actions (for debugging/visualization)
        self.state_names = {0: "Hole", 1: "Start", 2: "Goal"}
        self.action_names = {0: "Left", 1: "Right"}

        # Core transition dynamics: defines how the environment behaves
        # Structure: self.transitions[current_state][action] = (next_state, probability, reward)
        self.transitions = {
            0: {  # From Hole state (terminal - agent is "stuck")
                0: (0, 1.0, 0),  # Left action -> stay in Hole, 100% probability, 0 reward
                1: (0, 1.0, 0)  # Right action -> stay in Hole, 100% probability, 0 reward
            },
            1: {  # From Start state (the only interesting state)
                0: (0, 1.0, 0),  # Left action -> go to Hole, 100% probability, 0 reward (bad!)
                1: (2, 1.0, 1)  # Right action -> go to Goal, 100% probability, +1 reward (good!)
            },
            2: {  # From Goal state (terminal - agent succeeded)
                0: (2, 1.0, 0),  # Left action -> stay in Goal, 100% probability, 0 reward
                1: (2, 1.0, 0)  # Right action -> stay in Goal, 100% probability, 0 reward
            }
        }

        # Define which states end the episode (no more decisions to make)
        self.terminal_states = {0, 2}  # Both Hole and Goal are terminal

    def reset(self) -> int:
        """Reset the environment to the initial state."""
        self.current_state = self.initial_state
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take an action in the environment.

        Args:
            action (int): Action to take (0=Left, 1=Right)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action not in [0, 1]:
            raise ValueError(f"Invalid action {action}. Must be 0 (Left) or 1 (Right)")

        # Get transition information
        next_state, probability, reward = self.transitions[self.current_state][action]

        # Update current state
        previous_state = self.current_state
        self.current_state = next_state

        # Check if episode is done (reached terminal state)
        done = self.current_state in self.terminal_states

        # Additional info
        info = {
            'previous_state': previous_state,
            'action_taken': self.action_names[action],
            'state_name': self.state_names[self.current_state],
            'transition_probability': probability
        }

        return self.current_state, reward, done, info

    def get_state_name(self, state: int) -> str:
        """Get the name of a state."""
        return self.state_names.get(state, f"Unknown state {state}")

    def get_action_name(self, action: int) -> str:
        """Get the name of an action."""
        return self.action_names.get(action, f"Unknown action {action}")

    def is_terminal(self, state: Optional[int] = None) -> bool:
        """Check if a state is terminal."""
        if state is None:
            state = self.current_state
        return state in self.terminal_states

    def get_possible_actions(self, state: Optional[int] = None) -> list:
        """Get list of possible actions from a state."""
        if state is None:
            state = self.current_state
        return list(self.transitions[state].keys())

    def render(self):
        """Print the current state of the environment."""
        state_repr = ["[ ]", "[ ]", "[ ]"]
        state_repr[self.current_state] = "[X]"

        print(f"Hole    Start   Goal")
        print(f"{state_repr[0]}    {state_repr[1]}    {state_repr[2]}")
        print(f"Current state: {self.current_state} ({self.get_state_name(self.current_state)})")
        print(f"Terminal: {self.is_terminal()}")
        print()


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = BanditWalkEnvironment()

    print("=== Bandit Walk Environment Demo ===\n")

    # Example 1: Single episode
    print("Example 1: Single episode")
    print("-" * 30)

    state = env.reset()
    env.render()

    # Take action Right (should go to Goal and get reward)
    action = 1  # Right
    next_state, reward, done, info = env.step(action)

    print(f"Action taken: {env.get_action_name(action)}")
    print(f"Reward received: {reward}")
    print(f"Episode done: {done}")
    print(f"Info: {info}")
    env.render()

    print("\n" + "=" * 50 + "\n")

    # Example 2: Multiple episodes
    print("Example 2: Multiple random episodes")
    print("-" * 30)

    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        state = env.reset()
        total_reward = 0
        step_count = 0

        while not env.is_terminal() and step_count < 10:  # Safety limit
            # Choose random action
            possible_actions = env.get_possible_actions()
            action = random.choice(possible_actions)

            print(
                f"  Step {step_count + 1}: State {state} ({env.get_state_name(state)}) -> Action {action} ({env.get_action_name(action)})")

            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            print(f"    -> Next state: {state} ({env.get_state_name(state)}), Reward: {reward}")

            if done:
                break

        print(f"  Episode finished! Total reward: {total_reward}, Steps: {step_count}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: Demonstrate optimal policy
    print("Example 3: Optimal policy demonstration")
    print("-" * 30)

    state = env.reset()
    print("Starting from Start state, optimal action is Right to reach Goal:")
    env.render()

    # Optimal action: Right
    action = 1
    next_state, reward, done, info = env.step(action)

    print(f"Taking optimal action: {env.get_action_name(action)}")
    print(f"Result: Reward = {reward}, Reached {env.get_state_name(next_state)}")
    env.render()