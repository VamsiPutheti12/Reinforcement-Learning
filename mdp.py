import time

def solve_mdp_example():
    """
    Implements the value iteration algorithm for the simple
    'Good'/'Bad' state MDP example.
    """
    # 1. Define the MDP components
    # -------------------------------------
    states = ["Good", "Bad"]
    actions = ["Stay", "Go"]
    gamma = 0.9  # Discount factor for future rewards
    
    # Transition probabilities: P(s' | s, a)
    # Describes where you land after taking an action in a state.
    transitions = {
        "Good": {
            "Stay": {"Good": 1.0},  # Staying in 'Good' keeps you in 'Good'
            "Go": {"Bad": 1.0}      # Going from 'Good' takes you to 'Bad'
        },
        "Bad": { # The 'Bad' state is a terminal state; you can't leave.
            "Stay": {"Bad": 1.0},
            "Go": {"Bad": 1.0}
        }
    }
    
    # Rewards R(s, a)
    # The immediate reward for taking an action from a state.
    rewards = {
        "Good": {"Stay": 1, "Go": 5},
        "Bad": {"Stay": 0, "Go": 0}
    }

    # 2. Initialize the algorithm
    # ---------------------------------
    V = {s: 0 for s in states}  # Start with value of all states as 0
    epsilon = 0.001  # Convergence threshold
    iteration = 0

    print("--- Starting Value Iteration ---")
    print(f"Initial Values: {V}\n")

    # 3. The main loop
    # -------------------
    while True:
        V_old = V.copy()
        max_change = 0  # Tracks the biggest change in value in this iteration
        iteration += 1

        # Update the value for each state
        for s in states:
            # The value of the terminal 'Bad' state is always 0
            if s == "Bad":
                continue

            # Calculate the value of taking each action from the current state 's'
            action_values = {}
            for a in actions:
                # The Bellman equation part: immediate reward + discounted future value
                next_state = list(transitions[s][a].keys())[0] # Get the resulting state
                reward = rewards[s][a]
                
                # This is the "value of stay" or "value of go" calculation
                action_values[a] = reward + gamma * V_old[next_state]

            # The new value of the state is the best you can do from it
            V[s] = max(action_values.values())
            
            # Check how much the value of this state has changed
            max_change = max(max_change, abs(V[s] - V_old[s]))
        
        print(f"Iteration {iteration}: Values = {{'Good': {V['Good']:.3f}, 'Bad': {V['Bad']:.1f}}}, Change = {max_change:.4f}")
        time.sleep(0.5) # Pause for half a second to make it readable

        # 4. Check for convergence
        # ---------------------------
        if max_change < epsilon:
            print("\n--- Convergence Reached! ---")
            break

    # 5. Extract the Optimal Policy
    # --------------------------------
    policy = {}
    for s in states:
        if s == "Bad":
            policy[s] = "N/A (Terminal)"
            continue
            
        # For the 'Good' state, find which action yields the highest value
        action_values = {}
        for a in actions:
            next_state = list(transitions[s][a].keys())[0]
            reward = rewards[s][a]
            action_values[a] = reward + gamma * V[next_state]

        # The best policy is to choose the action with the highest value
        best_action = max(action_values, key=action_values.get)
        policy[s] = best_action

    print(f"\nFinal Converged Values: {V}")
    print(f"Optimal Policy: {policy} 💡")

# Run the algorithm
solve_mdp_example()
