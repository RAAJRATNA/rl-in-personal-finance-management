import numpy as np
import csv
import os
import pickle

# Parameters
states = list(range(0, 110000, 10000))  # total expense buckets: 0–100k
actions = list(range(0, 6000, 1000))    # suggested savings: 0–5k
q_table = np.zeros((len(states), len(actions)))
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
episodes = 500

# Load monthly expenses
def get_monthly_expenses():
    current_month = datetime.now().strftime("%Y-%m")
    total = 0
    try:
        with open("data/expenses.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[2].startswith(current_month):
                    total += float(row[1])
    except:
        pass
    return total

# Reward function: less expense → higher reward
def get_reward(expense, savings):
    return max(0, 5000 - abs(expense - savings))

# Training
for episode in range(episodes):
    expense = np.random.choice(states)
    state_idx = states.index(expense)

    for _ in range(10):
        action_idx = np.random.choice(len(actions))
        savings = actions[action_idx]
        reward = get_reward(expense, savings)

        # Update Q-Value
        q_table[state_idx, action_idx] = q_table[state_idx, action_idx] + alpha * (
            reward + gamma * np.max(q_table[state_idx]) - q_table[state_idx, action_idx]
        )

# Save Q-table
os.makedirs("model", exist_ok=True)
with open("model/q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("✅ Training complete. Q-table saved.")
