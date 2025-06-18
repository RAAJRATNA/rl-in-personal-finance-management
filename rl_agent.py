import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from finance_env import FinanceEnv
from q_learning_agent import QLearningAgent

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)

    # Drop date for now
    df = df.drop(columns=["Date"], errors="ignore")

    # Normalize Income, Expense, Goal Progress
    df["Income"] = df["Income"] / df["Income"].max()
    df["Expense"] = df["Expense"] / df["Expense"].max()
    df["Goal_Progress"] = df["Goal_Progress"] / 100.0

    # Encode actions
    le = LabelEncoder()
    df["Action_Index"] = le.fit_transform(df["Recommended_Action"])

    return df, le

def load_trained_agent():
    """Load the trained Q-learning agent"""
    try:
        with open("model/best_q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        env = FinanceEnv()
        agent = QLearningAgent(env)
        agent.q_table = q_table
        return agent
    except FileNotFoundError:
        return None

def get_rl_recommendation(income, expenses, savings, investments=None, emergency_fund=0, total_wealth=None):
    """Get RL-based financial recommendations"""
    try:
        agent = load_trained_agent()
        if agent is None:
            return {
                "budget_category": "N/A",
                "action": "N/A",
                "priority_goal": "N/A",
                "error": "Model not trained yet"
            }

        # Prepare state vector for the new environment
        if investments is None:
            investments = {"Stocks": 0, "Bonds": 0, "Real Estate": 0}
        if total_wealth is None:
            total_wealth = income
        max_expense = 50000  # Should match env
        emergency_fund_target = 100000
        
        total_investments = sum(investments.values())
        risk_score = 0.5  # Placeholder, env will update
        state = np.array([
            expenses / max_expense if max_expense else 0,
            savings / income if income else 0,
            total_investments / income if income else 0,
            emergency_fund / emergency_fund_target if emergency_fund_target else 0,
            income / income if income else 0,
            risk_score,
            total_wealth / (income * 12) if income else 0,
            total_investments / (total_wealth + 1e-6) if total_wealth else 0
        ])

        # Get action from agent
        action = agent.get_recommendation(state)

        # Map action to recommendation
        action_map = {
            0: "Save more cash",
            1: "Invest in Stocks",
            2: "Invest in Bonds",
            3: "Invest in Real Estate",
            4: "Reduce Spending",
            5: "Build Emergency Fund"
        }
        priority_goal = get_priority_goal(action, savings, income, emergency_fund, emergency_fund_target)
        
        # Add descriptive suggestions based on action
        suggestion = ""
        if action == 0:
            suggestion = "Consider setting up an automatic savings plan to build your cash reserves."
        elif action == 1:
            suggestion = "Consider diversifying your stock portfolio for better risk management."
        elif action == 2:
            suggestion = "Bonds can provide stable returns. Consider government or corporate bonds."
        elif action == 3:
            suggestion = "Real estate can be a good long-term investment. Research local market trends."
        elif action == 4:
            suggestion = "Review your spending habits and identify areas where you can cut back."
        elif action == 5:
            suggestion = "Aim to build an emergency fund that covers 3-6 months of expenses."

        # Calculate daily spend limit if action is 'Reduce Spending'
        daily_spend_limit = 0
        if action == 4 and income > 0:
            monthly_available = income - expenses
            daily_spend_limit = max(0, monthly_available / 30)

        recommendations = {
            "budget_category": action_map.get(action, "Balanced Budget"),
            "action": action,
            "priority_goal": priority_goal,
            "suggestion": suggestion,
            "daily_spend_limit": round(daily_spend_limit, 2)
        }
        return recommendations
    except Exception as e:
        return {
            "budget_category": "N/A",
            "action": "N/A",
            "priority_goal": "N/A",
            "error": str(e)
        }

def get_priority_goal(action, savings, income, emergency_fund, emergency_fund_target):
    if income <= 0:
        return "Increase Income"
    savings_ratio = savings / income
    emergency_ratio = emergency_fund / emergency_fund_target if emergency_fund_target else 0
    if emergency_ratio < 0.5:
        return "Build Emergency Fund"
    elif savings_ratio < 0.1:
        return "Increase Savings Rate"
    elif action == 1:
        return "Start/Increase Stock Investments"
    elif action == 2:
        return "Start/Increase Bond Investments"
    elif action == 3:
        return "Consider Real Estate Investment"
    elif action == 4:
        return "Reduce Expenses"
    else:
        return "Maintain Current Strategy"
