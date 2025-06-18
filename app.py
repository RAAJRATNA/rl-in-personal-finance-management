from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import csv
import sqlite3
from datetime import datetime
from finance_env import FinanceEnv
from q_learning_agent import QLearningAgent
import pandas as pd
from stock_analyzer import StockAnalyzer
import shutil
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong key in production
stock_analyzer = StockAnalyzer()

# Ensure data folder and required files exist
os.makedirs("data", exist_ok=True)

# Initialize CSV files
for filename, headers in [
    ("expenses.csv", ["Category", "Amount", "Date"]),
    ("income.csv", ["Source", "Amount", "Date"]),
    ("goals.csv", ["Goal", "Target Amount", "Saved Amount", "Deadline"])
]:
    filepath = f"data/{filename}"
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

# Create SQLite database for users
conn = sqlite3.connect("data/users.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
""")
conn.commit()
conn.close()

# -------------------- ROUTES --------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect("data/users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0] == password:
            session['username'] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect("data/users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="Username already exists.")
        conn.close()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if 'username' not in session:
        return redirect(url_for("login"))

    current_month = datetime.now().strftime("%Y-%m")
    total_expense = 0
    total_income = 0
    total_savings = 0
    investments = {"Stocks": 0, "Bonds": 0, "Real Estate": 0}
    emergency_fund = 0
    total_wealth = 0

    # Calculate expenses
    try:
        with open("data/expenses.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                category, amount, date = row
                if date.startswith(current_month):
                    total_expense += float(amount)
    except FileNotFoundError:
        pass

    # Calculate income
    try:
        with open("data/income.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                source, amount, date = row
                if date.startswith(current_month):
                    total_income += float(amount)
    except FileNotFoundError:
        pass

    # Calculate savings and investments from goals (if tracked)
    try:
        with open("data/goals.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 4:
                    goal, target, saved, deadline = row
                    if "emergency" in goal.lower():
                        emergency_fund += float(saved)
                    elif "stock" in goal.lower():
                        investments["Stocks"] += float(saved)
                    elif "bond" in goal.lower():
                        investments["Bonds"] += float(saved)
                    elif "real estate" in goal.lower():
                        investments["Real Estate"] += float(saved)
                    else:
                        total_savings += float(saved)
    except FileNotFoundError:
        pass

    # Estimate total wealth
    total_wealth = total_income - total_expense + total_savings + sum(investments.values()) + emergency_fund
    savings = total_income - total_expense
    ai_suggestion = round(total_income * 0.1)

    # ✅ RL integration (updated)
    try:
        from rl_agent import get_rl_recommendation
        rl_recommendation = get_rl_recommendation(
            income=total_income,
            expenses=total_expense,
            savings=total_savings,
            investments=investments,
            emergency_fund=emergency_fund,
            total_wealth=total_wealth
        )
    except Exception as e:
        rl_recommendation = {
            "budget_category": "N/A",
            "action": "N/A",
            "priority_goal": "N/A",
            "error": str(e)
        }

    return render_template(
        "dashboard.html",
        total_income=total_income,
        total_expense=total_expense,
        savings=savings,
        ai_suggestion=ai_suggestion,
        rl_recommendation=rl_recommendation
    )

@app.route("/expenses", methods=["GET", "POST"])
def expenses():
    if 'username' not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        category = request.form.get("category")
        amount = request.form.get("amount")
        date = datetime.now().strftime("%Y-%m-%d")
        if category and amount:
            with open("data/expenses.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([category, amount, date])

    expenses_list = []
    total_monthly = 0
    current_month = datetime.now().strftime("%Y-%m")

    try:
        with open("data/expenses.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                category, amount, date = row
                expenses_list.append([category, amount, date])
                if date.startswith(current_month):
                    total_monthly += float(amount)
    except FileNotFoundError:
        pass

    # RL recommendation for expenses
    rl_recommendation = get_page_rl_recommendation()

    return render_template("expenses.html", expenses=expenses_list, total_monthly=total_monthly, rl_recommendation=rl_recommendation)

@app.route("/income", methods=["GET", "POST"])
def income():
    if 'username' not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        source = request.form.get("source")
        amount = request.form.get("amount")
        date = datetime.now().strftime("%Y-%m-%d")
        if source and amount:
            with open("data/income.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([source, amount, date])

    income_list = []
    try:
        with open("data/income.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                source, amount, date = row
                income_list.append([source, amount, date])
    except FileNotFoundError:
        pass

    # RL recommendation for income
    rl_recommendation = get_page_rl_recommendation()

    return render_template("income.html", incomes=income_list, rl_recommendation=rl_recommendation)

@app.route("/savings", methods=["GET", "POST"])
def goals():
    if 'username' not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        goal = request.form.get("goal")
        target = request.form.get("target")
        saved = request.form.get("saved") or "0"
        deadline = request.form.get("deadline")
        if goal and target and deadline:
            with open("data/goals.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([goal, target, saved, deadline])

    goals_list = []
    try:
        with open("data/goals.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 4:
                    goals_list.append(row)
    except FileNotFoundError:
        pass

    # RL recommendation for savings
    rl_recommendation = get_page_rl_recommendation()

    return render_template("savings.html", savings=goals_list, rl_recommendation=rl_recommendation)

@app.route("/analysis")
def analysis():
    # Get financial data
    income_data = pd.read_csv("data/income.csv")
    expenses_data = pd.read_csv("data/expenses.csv")
    goals_data = pd.read_csv("data/goals.csv")
    
    # Calculate totals
    total_income = income_data['Amount'].sum()
    total_expenses = expenses_data['Amount'].sum()
    total_savings = total_income - total_expenses
    
    # Get goals data
    goals = goals_data.to_dict('records')
    
    # Get RL recommendation
    rl_recommendation = get_rl_recommendation(total_income, total_expenses, total_savings)
    
    return render_template("analysis.html", 
                         income=total_income,
                         expenses=total_expenses,
                         savings=total_savings,
                         goals=goals,
                         rl_recommendation=rl_recommendation)

def get_top_stocks(income, expenses, savings):
    """Get personalized stock recommendations based on advanced financial analysis"""
    try:
        # Popular Indian stocks with updated live prices and enhanced metrics
        stocks = [
            {
                'symbol': 'RELIANCE',
                'name': 'Reliance Industries',
                'price': 1430.10,  # Updated live price
                'recommendation': 'BUY',
                'confidence': 0.85,
                'rsi': 55.08,
                'volatility': 0.0136,
                'sma_20': 1420.50,
                'sma_50': 1380.25,
                'risk_level': 'medium',
                'min_investment': 5000,  # Reduced for accessibility
                'sector': 'Energy',
                'market_cap': 'Large',
                'dividend_yield': 0.8,
                'description': 'Diversified conglomerate with strong market presence'
            },
            {
                'symbol': 'TCS',
                'name': 'Tata Consultancy Services',
                'price': 3451.40,  # Updated live price
                'recommendation': 'BUY',
                'confidence': 0.95,
                'rsi': 47.97,
                'volatility': 0.0138,
                'sma_20': 3420.30,
                'sma_50': 3380.15,
                'risk_level': 'low',
                'min_investment': 8000,
                'sector': 'Technology',
                'market_cap': 'Large',
                'dividend_yield': 1.2,
                'description': 'Leading IT services company with global presence'
            },
            {
                'symbol': 'HDFCBANK',
                'name': 'HDFC Bank',
                'price': 1933.90,  # Updated live price
                'recommendation': 'HOLD',
                'confidence': 0.70,
                'rsi': 51.77,
                'volatility': 0.0123,
                'sma_20': 1920.40,
                'sma_50': 1890.60,
                'risk_level': 'low',
                'min_investment': 4000,
                'sector': 'Banking',
                'market_cap': 'Large',
                'dividend_yield': 1.5,
                'description': 'Premier private sector bank with strong fundamentals'
            },
            {
                'symbol': 'INFY',
                'name': 'Infosys',
                'price': 1632.90,  # Updated live price
                'recommendation': 'BUY',
                'confidence': 0.95,
                'rsi': 68.97,
                'volatility': 0.0160,
                'sma_20': 1610.20,
                'sma_50': 1580.45,
                'risk_level': 'low',
                'min_investment': 4000,
                'sector': 'Technology',
                'market_cap': 'Large',
                'dividend_yield': 2.1,
                'description': 'Global technology consulting and services company'
            },
            {
                'symbol': 'ICICIBANK',
                'name': 'ICICI Bank',
                'price': 1050.00,
                'recommendation': 'BUY',
                'confidence': 0.80,
                'rsi': 58.30,
                'volatility': 0.0140,
                'sma_20': 1040.50,
                'sma_50': 1020.30,
                'risk_level': 'medium',
                'min_investment': 3000,
                'sector': 'Banking',
                'market_cap': 'Large',
                'dividend_yield': 1.8,
                'description': 'Leading private sector bank with strong retail focus'
            },
            {
                'symbol': 'BAJFINANCE',
                'name': 'Bajaj Finance',
                'price': 7200.00,
                'recommendation': 'HOLD',
                'confidence': 0.65,
                'rsi': 62.40,
                'volatility': 0.0200,
                'sma_20': 7150.00,
                'sma_50': 7000.00,
                'risk_level': 'high',
                'min_investment': 15000,
                'sector': 'Finance',
                'market_cap': 'Large',
                'dividend_yield': 0.5,
                'description': 'Leading NBFC with strong growth in consumer finance'
            },
            {
                'symbol': 'BHARTIARTL',
                'name': 'Bharti Airtel',
                'price': 1200.00,
                'recommendation': 'BUY',
                'confidence': 0.75,
                'rsi': 45.20,
                'volatility': 0.0150,
                'sma_20': 1180.00,
                'sma_50': 1150.00,
                'risk_level': 'medium',
                'min_investment': 3000,
                'sector': 'Telecommunications',
                'market_cap': 'Large',
                'dividend_yield': 0.3,
                'description': 'Leading telecommunications company with pan-India presence'
            }
        ]
        
        # Advanced financial analysis
        monthly_savings = income - expenses
        savings_rate = (monthly_savings / income) * 100 if income > 0 else 0
        emergency_fund_ratio = savings / (expenses * 6) if expenses > 0 else 0  # 6 months emergency fund
        
        # Enhanced risk tolerance assessment
        risk_tolerance = calculate_risk_tolerance(income, expenses, savings, savings_rate, emergency_fund_ratio)
        
        # Market condition assessment (simplified)
        market_condition = assess_market_condition()
        
        # Calculate optimal investment capacity
        investment_capacity = calculate_investment_capacity(income, expenses, savings, savings_rate, emergency_fund_ratio)
        
        # Score and filter stocks with advanced criteria
        scored_stocks = []
        for stock in stocks:
            score = calculate_stock_score(
                stock, risk_tolerance, market_condition, 
                investment_capacity, savings_rate, emergency_fund_ratio
            )
            stock['score'] = score
            stock['investment_suitability'] = assess_investment_suitability(stock, investment_capacity)
            scored_stocks.append(stock)
        
        # Sort by score and return top recommendations
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # Return different number of stocks based on investment capacity
        if investment_capacity >= 50000:
            return scored_stocks[:7]  # More options for high capacity
        elif investment_capacity >= 20000:
            return scored_stocks[:5]  # Standard recommendations
        elif investment_capacity >= 10000:
            return scored_stocks[:3]  # Conservative for lower capacity
        else:
            return scored_stocks[:2]  # Minimal for very low capacity
            
    except Exception as e:
        print(f"Error getting top stocks: {e}")
        return []

def calculate_risk_tolerance(income, expenses, savings, savings_rate, emergency_fund_ratio):
    """Calculate personalized risk tolerance based on multiple factors"""
    risk_score = 0
    
    # Income stability factor
    if income >= 100000:  # High income
        risk_score += 2
    elif income >= 50000:  # Medium income
        risk_score += 1
    
    # Savings rate factor
    if savings_rate >= 40:
        risk_score += 3
    elif savings_rate >= 30:
        risk_score += 2
    elif savings_rate >= 20:
        risk_score += 1
    
    # Emergency fund factor
    if emergency_fund_ratio >= 1.5:  # More than 9 months of expenses
        risk_score += 2
    elif emergency_fund_ratio >= 1.0:  # 6 months of expenses
        risk_score += 1
    
    # Age and experience factor (simplified)
    risk_score += 1  # Assume moderate experience
    
    # Determine risk tolerance
    if risk_score >= 7:
        return 'high'
    elif risk_score >= 4:
        return 'medium'
    else:
        return 'low'

def assess_market_condition():
    """Assess current market conditions (simplified)"""
    # In a real implementation, this would analyze market indicators
    # For now, return a neutral condition
    return {
        'trend': 'neutral',  # bullish, bearish, neutral
        'volatility': 'medium',
        'confidence': 0.6
    }

def calculate_investment_capacity(income, expenses, savings, savings_rate, emergency_fund_ratio):
    """Calculate optimal investment capacity with safety considerations"""
    
    # Base capacity calculation
    monthly_savings = income - expenses
    
    # Conservative approach: Only invest what you can afford to lose
    if emergency_fund_ratio < 1.0:
        # Don't invest if emergency fund is insufficient
        return 0
    
    # Calculate safe investment amount
    if savings_rate >= 40:
        # High savings rate - can invest more
        safe_investment = min(monthly_savings * 0.4, savings * 0.15)
    elif savings_rate >= 30:
        # Good savings rate - moderate investment
        safe_investment = min(monthly_savings * 0.3, savings * 0.10)
    elif savings_rate >= 20:
        # Moderate savings rate - conservative investment
        safe_investment = min(monthly_savings * 0.2, savings * 0.05)
    else:
        # Low savings rate - very conservative
        safe_investment = min(monthly_savings * 0.1, savings * 0.02)
    
    # Apply emergency fund safety factor
    if emergency_fund_ratio < 1.5:
        safe_investment *= 0.5  # Reduce by 50% if emergency fund is borderline
    
    return max(0, safe_investment)

def calculate_stock_score(stock, risk_tolerance, market_condition, investment_capacity, savings_rate, emergency_fund_ratio):
    """Calculate comprehensive stock score with multiple factors"""
    score = 0
    
    # Base recommendation score
    if stock['recommendation'] == 'BUY':
        score += stock['confidence'] * 100
    elif stock['recommendation'] == 'HOLD':
        score += stock['confidence'] * 50
    
    # Risk tolerance alignment
    if stock['risk_level'] == risk_tolerance:
        score += 30
    elif (risk_tolerance == 'high' and stock['risk_level'] == 'medium') or \
         (risk_tolerance == 'medium' and stock['risk_level'] == 'low'):
        score += 15
    elif risk_tolerance == 'low' and stock['risk_level'] == 'high':
        score -= 20  # Penalty for high risk when user is conservative
    
    # Investment capacity consideration
    if stock['min_investment'] <= investment_capacity:
        score += 25
    else:
        score -= 30  # Heavy penalty if can't afford
    
    # Technical indicators
    if stock['rsi'] < 70 and stock['rsi'] > 30:  # Not overbought or oversold
        score += 15
    if stock['volatility'] < 0.02:  # Low volatility preferred
        score += 10
    elif stock['volatility'] > 0.03:  # High volatility penalty
        score -= 10
    
    # Moving average analysis
    if stock['sma_20'] > stock['sma_50']:  # Bullish trend
        score += 10
    
    # Sector diversification bonus
    if stock['sector'] in ['Technology', 'Banking']:  # Preferred sectors
        score += 5
    
    # Dividend yield consideration
    if stock['dividend_yield'] > 1.0:  # Good dividend
        score += 5
    
    # Market condition adjustment
    if market_condition['trend'] == 'bullish':
        score += 10
    elif market_condition['trend'] == 'bearish':
        score -= 10
    
    # Emergency fund safety factor
    if emergency_fund_ratio < 1.5:
        # Prefer lower risk stocks when emergency fund is borderline
        if stock['risk_level'] == 'low':
            score += 10
        elif stock['risk_level'] == 'high':
            score -= 15
    
    return score

def assess_investment_suitability(stock, investment_capacity):
    """Assess if stock is suitable for current investment capacity"""
    if investment_capacity == 0:
        return 'not_recommended'
    elif stock['min_investment'] > investment_capacity:
        return 'too_expensive'
    elif stock['min_investment'] <= investment_capacity * 0.3:
        return 'excellent_fit'
    elif stock['min_investment'] <= investment_capacity * 0.6:
        return 'good_fit'
    else:
        return 'moderate_fit'

def get_portfolio_analysis(income, expenses, savings):
    """Get advanced portfolio analysis with personalized allocation strategy"""
    try:
        # Calculate financial health metrics
        monthly_savings = income - expenses
        savings_rate = (monthly_savings / income) * 100 if income > 0 else 0
        emergency_fund_ratio = savings / (expenses * 6) if expenses > 0 else 0
        
        # Calculate optimal investment capacity
        investment_capacity = calculate_investment_capacity(income, expenses, savings, savings_rate, emergency_fund_ratio)
        
        # Determine personalized asset allocation
        allocation = calculate_asset_allocation(savings_rate, emergency_fund_ratio, investment_capacity)
        
        # Calculate amounts for each asset class
        amounts = {
            'stocks': investment_capacity * allocation['stocks'],
            'bonds': investment_capacity * allocation['bonds'],
            'cash': investment_capacity * allocation['cash'],
            'emergency_fund': savings * 0.5  # Keep 50% as emergency fund
        }
        
        # Get personalized stock recommendations
        recommended_stocks = get_top_stocks(income, expenses, savings)
        
        # Calculate optimal stock allocations
        stock_allocations = calculate_stock_allocations(amounts['stocks'], recommended_stocks, investment_capacity)
        
        # Generate investment strategy
        strategy = generate_investment_strategy(income, expenses, savings, savings_rate, emergency_fund_ratio, allocation)
        
        return {
            'total_investable': investment_capacity,
            'allocation': allocation,
            'amounts': amounts,
            'stock_allocations': stock_allocations,
            'strategy': strategy,
            'financial_health': {
                'savings_rate': savings_rate,
                'emergency_fund_ratio': emergency_fund_ratio,
                'investment_readiness': assess_investment_readiness(emergency_fund_ratio, savings_rate)
            }
        }
    except Exception as e:
        print(f"Error in portfolio analysis: {e}")
        return None

def calculate_asset_allocation(savings_rate, emergency_fund_ratio, investment_capacity):
    """Calculate personalized asset allocation based on financial health"""
    
    # Base allocation depends on savings rate and emergency fund
    if emergency_fund_ratio < 1.0:
        # Emergency fund insufficient - very conservative
        return {
            'stocks': 0.05,   # Only 5% in stocks
            'bonds': 0.25,    # 25% in bonds
            'cash': 0.70      # 70% in cash (build emergency fund)
        }
    elif emergency_fund_ratio < 1.5:
        # Emergency fund borderline - conservative
        return {
            'stocks': 0.10,   # 10% in stocks
            'bonds': 0.40,    # 40% in bonds
            'cash': 0.50      # 50% in cash
        }
    elif savings_rate >= 40:
        # High savings rate - moderate growth
        return {
            'stocks': 0.30,   # 30% in stocks
            'bonds': 0.50,    # 50% in bonds
            'cash': 0.20      # 20% in cash
        }
    elif savings_rate >= 30:
        # Good savings rate - balanced
        return {
            'stocks': 0.25,   # 25% in stocks
            'bonds': 0.55,    # 55% in bonds
            'cash': 0.20      # 20% in cash
        }
    elif savings_rate >= 20:
        # Moderate savings rate - conservative
        return {
            'stocks': 0.15,   # 15% in stocks
            'bonds': 0.60,    # 60% in bonds
            'cash': 0.25      # 25% in cash
        }
    else:
        # Low savings rate - very conservative
        return {
            'stocks': 0.05,   # 5% in stocks
            'bonds': 0.35,    # 35% in bonds
            'cash': 0.60      # 60% in cash
        }

def calculate_stock_allocations(stock_amount, recommended_stocks, investment_capacity):
    """Calculate optimal stock allocations with safety limits"""
    stock_allocations = []
    
    if stock_amount <= 0 or not recommended_stocks:
        return stock_allocations
    
    # Calculate equal distribution among recommended stocks
    per_stock_amount = stock_amount / len(recommended_stocks)
    
    for stock in recommended_stocks:
        if stock['price'] > 0 and stock['investment_suitability'] != 'too_expensive':
            # Calculate maximum shares based on price and safety limits
            max_shares = int(per_stock_amount / stock['price'])
            
            # Apply safety limits based on investment capacity
            if investment_capacity >= 50000:
                max_shares = min(max_shares, 5)  # Up to 5 shares for high capacity
            elif investment_capacity >= 20000:
                max_shares = min(max_shares, 3)  # Up to 3 shares for medium capacity
            else:
                max_shares = min(max_shares, 2)  # Up to 2 shares for low capacity
            
            # Ensure minimum investment requirement is met
            if max_shares * stock['price'] >= stock['min_investment']:
                investment = max_shares * stock['price']
                stock_allocations.append({
                    'symbol': stock['symbol'],
                    'name': stock['name'],
                    'shares': max_shares,
                    'investment': investment,
                    'percentage': (investment / stock_amount) * 100,
                    'suitability': stock['investment_suitability'],
                    'risk_level': stock['risk_level'],
                    'expected_return': calculate_expected_return(stock)
                })
    
    return stock_allocations

def calculate_expected_return(stock):
    """Calculate expected return based on dividend yield and growth potential"""
    base_return = stock.get('dividend_yield', 0)
    
    # Add growth potential based on sector and market cap
    if stock.get('sector') == 'Technology':
        base_return += 8  # Higher growth potential
    elif stock.get('sector') == 'Banking':
        base_return += 6  # Moderate growth
    else:
        base_return += 4  # Lower growth
    
    return round(base_return, 1)

def generate_investment_strategy(income, expenses, savings, savings_rate, emergency_fund_ratio, allocation):
    """Generate personalized investment strategy recommendations"""
    strategy = {
        'priority_actions': [],
        'risk_warnings': [],
        'growth_opportunities': [],
        'timeline': 'medium_term'
    }
    
    # Priority actions based on financial health
    if emergency_fund_ratio < 1.0:
        strategy['priority_actions'].append("Build emergency fund to 6 months of expenses before investing")
        strategy['risk_warnings'].append("Current emergency fund is insufficient for safe investing")
    elif emergency_fund_ratio < 1.5:
        strategy['priority_actions'].append("Continue building emergency fund while starting small investments")
        strategy['risk_warnings'].append("Consider conservative investments only")
    
    if savings_rate < 20:
        strategy['priority_actions'].append("Increase savings rate to at least 20% for better investment capacity")
        strategy['growth_opportunities'].append("Focus on reducing expenses to improve savings rate")
    elif savings_rate >= 30:
        strategy['growth_opportunities'].append("High savings rate allows for more aggressive investment strategy")
    
    # Investment timeline recommendation
    if allocation['stocks'] >= 0.25:
        strategy['timeline'] = 'long_term'
        strategy['growth_opportunities'].append("Consider long-term investment horizon for stock portfolio")
    elif allocation['stocks'] <= 0.10:
        strategy['timeline'] = 'short_term'
        strategy['priority_actions'].append("Focus on short-term goals and emergency fund building")
    
    return strategy

def assess_investment_readiness(emergency_fund_ratio, savings_rate):
    """Assess if user is ready for investing"""
    if emergency_fund_ratio < 1.0:
        return 'not_ready'
    elif emergency_fund_ratio < 1.5:
        return 'borderline'
    elif savings_rate < 20:
        return 'needs_improvement'
    else:
        return 'ready'

@app.route('/stocks', methods=['GET', 'POST'])
def stocks():
    if 'username' not in session:
        return redirect(url_for("login"))
        
    try:
        # Get financial data
        income = get_total_income()
        expenses = get_total_expenses()
        savings = get_total_savings()
        
        # Get portfolio analysis
        portfolio_analysis = get_portfolio_analysis(income, expenses, savings)
        
        # Get RL recommendation
        rl_recommendation = get_rl_recommendation(income, expenses, savings)
        
        # Get top 5 recommended stocks based on financial data
        top_stocks = get_top_stocks(income, expenses, savings)
        
        # Get market status
        market_status = stock_analyzer.get_market_status()
        
        if request.method == 'POST':
            symbol = request.form.get('symbol', '').strip().upper()
            if symbol:
                # Analyze the requested stock
                analysis = stock_analyzer.analyze_stock(symbol)
                if analysis.get('status') == 'success':
                    return jsonify({
                        'success': True,
                        'analysis': analysis,
                        'market_status': market_status
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': analysis.get('error', 'Analysis failed'),
                        'market_status': market_status
                    })
        
        return render_template('stocks.html',
                             portfolio_analysis=portfolio_analysis,
                             rl_recommendation=rl_recommendation,
                             top_stocks=top_stocks,
                             market_status=market_status)
    except Exception as e:
        print(f"Error in stocks route: {e}")
        return render_template('stocks.html',
                             portfolio_analysis=None,
                             rl_recommendation=None,
                             top_stocks=[],
                             market_status={'is_open': False, 'error': str(e)})

@app.route('/api/stock/<symbol>')
def api_stock_analysis(symbol):
    """API endpoint for stock analysis"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
        
    try:
        analysis = stock_analyzer.analyze_stock(symbol)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-status')
def api_market_status():
    """API endpoint for market status"""
    try:
        status = stock_analyzer.get_market_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_total_income():
    """Calculate total income from income.csv"""
    try:
        income_df = pd.read_csv('data/income.csv')
        return income_df['Amount'].sum()
    except Exception as e:
        print(f"Error reading income data: {e}")
        return 0

def get_total_expenses():
    """Calculate total expenses from expenses.csv"""
    try:
        expenses_df = pd.read_csv('data/expenses.csv')
        return expenses_df['Amount'].sum()
    except Exception as e:
        print(f"Error reading expenses data: {e}")
        return 0

def get_total_savings():
    """Calculate total savings from goals.csv"""
    try:
        goals_df = pd.read_csv('data/goals.csv')
        return goals_df['Saved'].sum()
    except Exception as e:
        print(f"Error reading savings data: {e}")
        return 0

@app.route("/contact")
def contact():
    if 'username' not in session:
        return redirect(url_for("login"))
    return render_template("contact.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/clear-data", methods=["POST"])
def clear_data():
    """Clear all financial data for the current user"""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    try:
        username = session['username']
        
        # Clear all CSV files
        csv_files = ['expenses.csv', 'income.csv', 'goals.csv']
        cleared_files = []
        
        for filename in csv_files:
            filepath = f"data/{filename}"
            if os.path.exists(filepath):
                # Create backup before clearing
                backup_path = f"data/{filename}.backup_{int(time.time())}"
                shutil.copy2(filepath, backup_path)
                
                # Clear the file but keep headers
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if filename == 'expenses.csv':
                        writer.writerow(['Category', 'Amount', 'Date'])
                    elif filename == 'income.csv':
                        writer.writerow(['Source', 'Amount', 'Date'])
                    elif filename == 'goals.csv':
                        writer.writerow(['Goal', 'Target Amount', 'Saved Amount', 'Deadline'])
                
                cleared_files.append(filename)
        
        # Clear stock analyzer cache
        try:
            stock_analyzer.clear_cache()
        except:
            pass  # Ignore cache clearing errors
        
        # Log the action
        print(f"User {username} cleared all financial data at {datetime.now()}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully cleared {len(cleared_files)} data files',
            'cleared_files': cleared_files
        })
        
    except Exception as e:
        print(f"Error clearing data: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to clear data: {str(e)}'
        }), 500

# Helper to get RL recommendation for all pages
from rl_agent import get_rl_recommendation

def get_page_rl_recommendation():
    # This logic mirrors the dashboard's state extraction
    current_month = datetime.now().strftime("%Y-%m")
    total_expense = 0
    total_income = 0
    total_savings = 0
    investments = {"Stocks": 0, "Bonds": 0, "Real Estate": 0}
    emergency_fund = 0
    total_wealth = 0
    try:
        with open("data/expenses.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                category, amount, date = row
                if date.startswith(current_month):
                    total_expense += float(amount)
    except FileNotFoundError:
        pass
    try:
        with open("data/income.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                source, amount, date = row
                if date.startswith(current_month):
                    total_income += float(amount)
    except FileNotFoundError:
        pass
    try:
        with open("data/goals.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 4:
                    goal, target, saved, deadline = row
                    if "emergency" in goal.lower():
                        emergency_fund += float(saved)
                    elif "stock" in goal.lower():
                        investments["Stocks"] += float(saved)
                    elif "bond" in goal.lower():
                        investments["Bonds"] += float(saved)
                    elif "real estate" in goal.lower():
                        investments["Real Estate"] += float(saved)
                    else:
                        total_savings += float(saved)
    except FileNotFoundError:
        pass
    total_wealth = total_income - total_expense + total_savings + sum(investments.values()) + emergency_fund
    return get_rl_recommendation(
        income=total_income,
        expenses=total_expense,
        savings=total_savings,
        investments=investments,
        emergency_fund=emergency_fund,
        total_wealth=total_wealth
    )

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
