import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import requests

class StockAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Cache for stock data to avoid repeated API calls
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance with improved error handling"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if time.time() - cached_time < self.cache_timeout:
                    return cached_data
            
            # Add .NS suffix for Indian stocks if not present
            if not symbol.endswith('.NS') and not '.' in symbol:
                symbol = f"{symbol}.NS"
            
            self.logger.info(f"Fetching data for {symbol}")
            stock = yf.Ticker(symbol)
            
            # Try to get historical data
            hist = stock.history(period=period)
            
            if hist.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
                
            # Cache the data
            self.cache[cache_key] = (time.time(), hist)
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_metrics(self, data):
        """Calculate key financial metrics with improved error handling"""
        if data is None or len(data) < 20:  # Need at least 20 days for calculations
            return None

        try:
            metrics = {}
            
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            if len(returns) == 0:
                return None
                
            metrics['daily_returns'] = returns.mean()
            metrics['volatility'] = returns.std()
            
            # Calculate moving averages
            if len(data) >= 20:
                metrics['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            else:
                metrics['sma_20'] = data['Close'].mean()
                
            if len(data) >= 50:
                metrics['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
            else:
                metrics['sma_50'] = data['Close'].mean()
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            metrics['rsi'] = rsi if not np.isnan(rsi) else 50
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            metrics['macd'] = macd.iloc[-1]
            metrics['macd_signal'] = signal.iloc[-1]
            
            # Calculate additional metrics
            metrics['current_price'] = data['Close'].iloc[-1]
            metrics['price_change'] = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
            metrics['price_change_pct'] = (metrics['price_change'] / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
            
            # Volume analysis
            if 'Volume' in data.columns:
                metrics['avg_volume'] = data['Volume'].mean()
                metrics['current_volume'] = data['Volume'].iloc[-1]
                metrics['volume_ratio'] = metrics['current_volume'] / metrics['avg_volume'] if metrics['avg_volume'] > 0 else 1
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return None

    def generate_recommendation(self, metrics):
        """Generate stock recommendation based on metrics with improved logic"""
        if metrics is None:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': 'Insufficient data for analysis',
                'risk_level': 'unknown'
            }

        try:
            score = 0
            reasons = []
            risk_factors = []

            # RSI Analysis
            if metrics['rsi'] < 30:
                score += 2
                reasons.append("RSI indicates oversold conditions (bullish)")
            elif metrics['rsi'] > 70:
                score -= 2
                reasons.append("RSI indicates overbought conditions (bearish)")
            elif 40 <= metrics['rsi'] <= 60:
                score += 1
                reasons.append("RSI in neutral range")

            # Moving Average Analysis
            if metrics['sma_20'] > metrics['sma_50']:
                score += 1
                reasons.append("Short-term trend is bullish")
            else:
                score -= 1
                reasons.append("Short-term trend is bearish")

            # MACD Analysis
            if metrics['macd'] > metrics['macd_signal']:
                score += 1
                reasons.append("MACD indicates bullish momentum")
            else:
                score -= 1
                reasons.append("MACD indicates bearish momentum")

            # Volatility Analysis
            if metrics['volatility'] < 0.02:
                score += 1
                reasons.append("Low volatility suggests stability")
                risk_factors.append("low")
            elif metrics['volatility'] > 0.04:
                score -= 1
                reasons.append("High volatility suggests risk")
                risk_factors.append("high")
            else:
                risk_factors.append("medium")

            # Price Change Analysis
            if metrics['price_change_pct'] > 2:
                score += 1
                reasons.append("Strong positive price movement")
            elif metrics['price_change_pct'] < -2:
                score -= 1
                reasons.append("Strong negative price movement")

            # Volume Analysis
            if 'volume_ratio' in metrics:
                if metrics['volume_ratio'] > 1.5:
                    score += 1
                    reasons.append("High volume confirms price movement")
                elif metrics['volume_ratio'] < 0.5:
                    score -= 1
                    reasons.append("Low volume suggests weak conviction")

            # Generate recommendation
            if score >= 3:
                action = 'BUY'
                confidence = min(score * 15 + 50, 95)
            elif score <= -3:
                action = 'SELL'
                confidence = min(abs(score) * 15 + 50, 95)
            else:
                action = 'HOLD'
                confidence = 50 + abs(score) * 10

            # Determine risk level
            risk_level = 'medium'
            if len(risk_factors) > 0:
                if 'high' in risk_factors:
                    risk_level = 'high'
                elif 'low' in risk_factors:
                    risk_level = 'low'

            return {
                'action': action,
                'confidence': round(confidence, 1),
                'reasons': reasons,
                'risk_level': risk_level,
                'metrics': {
                    'rsi': round(metrics['rsi'], 2),
                    'volatility': round(metrics['volatility'] * 100, 2),
                    'sma_20': round(metrics['sma_20'], 2),
                    'sma_50': round(metrics['sma_50'], 2),
                    'macd': round(metrics['macd'], 2),
                    'current_price': round(metrics['current_price'], 2),
                    'price_change_pct': round(metrics['price_change_pct'], 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': f'Error in analysis: {str(e)}',
                'risk_level': 'unknown'
            }

    def analyze_stock(self, symbol):
        """Complete stock analysis with improved error handling"""
        try:
            self.logger.info(f"Starting analysis for {symbol}")
            
            # Get stock data
            data = self.get_stock_data(symbol)
            if data is None:
                return {
                    'symbol': symbol,
                    'error': 'Unable to fetch stock data. Please check the symbol and try again.',
                    'status': 'error'
                }

            # Calculate metrics
            metrics = self.calculate_metrics(data)
            if metrics is None:
                return {
                    'symbol': symbol,
                    'error': 'Unable to calculate metrics. Insufficient data.',
                    'status': 'error'
                }

            # Generate recommendation
            recommendation = self.generate_recommendation(metrics)

            # Prepare historical data for charts
            historical_data = {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'prices': data['Close'].round(2).tolist(),
                'volumes': data['Volume'].tolist() if 'Volume' in data.columns else []
            }

            return {
                'symbol': symbol,
                'current_price': round(metrics['current_price'], 2),
                'recommendation': recommendation,
                'historical_data': historical_data,
                'status': 'success',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error in complete stock analysis for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': f'Analysis failed: {str(e)}',
                'status': 'error'
            }

    def get_market_status(self):
        """Get current market status"""
        try:
            # Check if market is open (simplified check for Indian market)
            now = datetime.now()
            market_open = now.weekday() < 5 and 9 <= now.hour < 16  # Monday-Friday, 9 AM to 4 PM
            
            return {
                'is_open': market_open,
                'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                'timezone': 'IST'
            }
        except Exception as e:
            self.logger.error(f"Error getting market status: {str(e)}")
            return {
                'is_open': False,
                'error': str(e)
            }

    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        self.logger.info("Stock data cache cleared") 