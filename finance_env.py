import numpy as np
import gym
from gym import spaces

class FinanceEnv(gym.Env):
    def __init__(self):
        super(FinanceEnv, self).__init__()
        
        # Enhanced financial parameters
        self.income = 50000
        self.max_expense = 50000
        self.emergency_fund_target = 100000
        self.inflation_rate = 0.02  # 2% annual inflation
        self.tax_rate = 0.20  # 20% tax rate
        
        # Enhanced investment opportunities with more metrics
        self.investment_opportunities = [
            {
                "name": "Stocks",
                "risk": 0.7,
                "return": 0.12,
                "volatility": 0.15,
                "liquidity": 0.8
            },
            {
                "name": "Bonds",
                "risk": 0.3,
                "return": 0.06,
                "volatility": 0.05,
                "liquidity": 0.6
            },
            {
                "name": "Savings",
                "risk": 0.1,
                "return": 0.03,
                "volatility": 0.01,
                "liquidity": 1.0
            },
            {
                "name": "Real Estate",
                "risk": 0.5,
                "return": 0.08,
                "volatility": 0.10,
                "liquidity": 0.3
            }
        ]

        # Action space: 0=save, 1=invest_stocks, 2=invest_bonds, 3=invest_real_estate, 4=spend, 5=emergency_fund
        self.action_space = spaces.Discrete(6)
        
        # Enhanced state space with more financial metrics
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(8,),  # Increased state dimensions
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.expenses = 0
        self.savings = 0
        self.investments = {
            "Stocks": 0,
            "Bonds": 0,
            "Real Estate": 0
        }
        self.emergency_fund = 0
        self.monthly_income = self.income
        self.total_wealth = self.income
        self.risk_score = 0.5
        return self._get_state()

    def _get_state(self):
        total_investments = sum(self.investments.values())
        return np.array([
            self.expenses / self.max_expense,
            self.savings / self.income,
            total_investments / self.income,
            self.emergency_fund / self.emergency_fund_target,
            self.monthly_income / self.income,
            self.risk_score,
            self.total_wealth / (self.income * 12),  # Wealth to annual income ratio
            sum(self.investments.values()) / (self.total_wealth + 1e-6)  # Investment ratio
        ], dtype=np.float32)

    def _calculate_risk_score(self):
        """Calculate portfolio risk score based on investments"""
        risk_score = 0
        total_invested = sum(self.investments.values())
        if total_invested > 0:
            for inv_type, amount in self.investments.items():
                weight = amount / total_invested
                for opp in self.investment_opportunities:
                    if opp["name"] == inv_type:
                        risk_score += weight * opp["risk"]
        return risk_score

    def _calculate_reward(self, action):
        base_reward = 0
        
        # Emergency fund reward with diminishing returns
        emergency_ratio = self.emergency_fund / self.emergency_fund_target
        base_reward += 2 * (1 - np.exp(-emergency_ratio))
        
        # Portfolio diversification reward
        total_investments = sum(self.investments.values())
        if total_investments > 0:
            diversification = 1 - sum((v/total_investments)**2 for v in self.investments.values())
            base_reward += diversification * 2
        
        # Risk-adjusted return reward
        risk_score = self._calculate_risk_score()
        expected_return = sum(
            self.investments[inv["name"]] * inv["return"] 
            for inv in self.investment_opportunities 
            if inv["name"] in self.investments
        ) / (total_investments + 1e-6)
        
        risk_adjusted_return = expected_return / (risk_score + 1e-6)
        base_reward += risk_adjusted_return * 3
        
        # Action-specific rewards with more sophisticated logic
        if action == 0:  # Save
            base_reward += 1 if self.savings < self.income * 0.3 else 0.5
        elif action == 1:  # Invest Stocks
            base_reward += 1.5 if self.investments["Stocks"] < self.income * 0.4 else 0.5
        elif action == 2:  # Invest Bonds
            base_reward += 1.2 if self.investments["Bonds"] < self.income * 0.3 else 0.5
        elif action == 3:  # Invest Real Estate
            base_reward += 1.3 if self.investments["Real Estate"] < self.income * 0.2 else 0.5
        elif action == 4:  # Spend
            expense_ratio = self.expenses / self.income
            base_reward += 0.5 if expense_ratio < 0.5 else -1
        elif action == 5:  # Emergency fund
            base_reward += 2 if self.emergency_fund < self.emergency_fund_target else 0.5
            
        # Penalize excessive risk
        if risk_score > 0.7:
            base_reward -= 1
            
        return base_reward

    def step(self, action):
        amount = 10000  # Base amount for actions

        if action == 0:  # Save
            self.savings += amount
        elif action == 1:  # Invest Stocks
            self.investments["Stocks"] += amount
        elif action == 2:  # Invest Bonds
            self.investments["Bonds"] += amount
        elif action == 3:  # Invest Real Estate
            self.investments["Real Estate"] += amount
        elif action == 4:  # Spend
            self.expenses += amount
        elif action == 5:  # Emergency fund
            self.emergency_fund += amount

        # Update total wealth
        self.total_wealth = (
            self.savings + 
            sum(self.investments.values()) + 
            self.emergency_fund - 
            self.expenses
        )
        
        # Update risk score
        self.risk_score = self._calculate_risk_score()
        
        reward = self._calculate_reward(action)
        
        # Episode ends if total allocations exceed income or if wealth drops too low
        done = (
            (self.expenses + self.savings + sum(self.investments.values()) + self.emergency_fund >= self.income) or
            (self.total_wealth < self.income * 0.5)  # Bankruptcy condition
        )
        
        return self._get_state(), reward, done, {}

