import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import math
from datetime import datetime, timedelta, date
import warnings
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64

# Try to import QuantLib
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced ALM Teaching Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "almeducate2025!":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show login screen
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem; font-weight: 700;">
                🏦 Enhanced ALM Teaching Platform
            </h1>
            <p style="color: #e2e8f0; text-align: center; margin: 1rem 0 0 0; font-size: 1.2rem;">
                Professional Asset-Liability Management Education Tool
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Educational Disclaimer
        st.markdown("""
        <div style="background: #fef3c7; border: 2px solid #f59e0b; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
            <h3 style="color: #92400e; margin: 0 0 1rem 0;">📚 Educational Disclaimer</h3>
            <p style="color: #92400e; margin: 0; line-height: 1.6;">
                <strong>This tool is designed for educational purposes only.</strong> All market data, rates, and scenarios 
                used in this application are either simulated or based on publicly available historical information. 
                The calculations and analytics provided should not be used for actual trading, investment decisions, 
                or risk management purposes. This platform is intended to help students and professionals learn 
                about Asset-Liability Management concepts in a controlled educational environment.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login form
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-left: 5px solid #3b82f6;">
                <h3 style="color: #1e3a8a; text-align: center; margin-bottom: 1.5rem;">🔐 Secure Access</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_input("Password", type="password", on_change=password_entered, key="password")
            
            # Feature preview
            st.markdown("""
            <div style="margin-top: 2rem; padding: 1.5rem; background: #f8fafc; border-radius: 10px; border-left: 4px solid #3b82f6;">
                <h4 style="color: #1e3a8a; margin: 0 0 1rem 0;">🎯 What You'll Access:</h4>
                <ul style="color: #475569; margin: 0;">
                    <li><strong>ALM Sensitivity Analysis</strong> - Interactive portfolio building with real-time insights</li>
                    <li><strong>Duration Calculator</strong> - Bond price sensitivity analysis with visualizations</li>
                    <li><strong>Swap Analytics Tool</strong> - Complete swap pricing and risk analysis</li>
                    <li><strong>Historical Analysis</strong> - Learn from past interest rate cycles</li>
                    <li><strong>Monte Carlo Simulations</strong> - Advanced risk modeling capabilities</li>
                    <li><strong>Educational Resources</strong> - Comprehensive learning materials</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect
        st.error("😞 Password incorrect. Please try again.")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    else:
        # Password correct
        return True

# Custom CSS with Financial Blue Theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.3);
    }
    .section-header {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        padding: 1rem 2rem;
        margin: 2rem 0 1rem 0;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.2);
    }
    .metric-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .sensitivity-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        border-left: 5px solid #28a745;
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.2);
    }
    .sensitivity-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f1c0c7 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        border-left: 5px solid #dc3545;
        box-shadow: 0 3px 10px rgba(220, 53, 69, 0.2);
    }
    .alm-advisor {
        background: linear-gradient(135deg, #e8f4fd 0%, #dbeafe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(59, 130, 246, 0.1);
    }
    .recommendation {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(255, 193, 7, 0.2);
    }
    .swap-pricing {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1);
    }
    .risk-attribution {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #9c27b0;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(156, 39, 176, 0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #3b82f6;
        border-radius: 8px;
    }
    .stNumberInput > div > div > input {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #3b82f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions for Market Data
@st.cache_data(ttl=3600)
def fetch_fred_data(series_id, start_date=None, end_date=None):
    """Fetch data from FRED API"""
    try:
        base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {
            'id': series_id,
            'cosd': start_date or '2000-01-01',
            'coed': end_date or datetime.now().strftime('%Y-%m-%d')
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            data = []
            headers = lines[0].split(',')
            
            for line in lines[1:]:
                values = line.split(',')
                if len(values) >= 2 and values[1] != '.':
                    try:
                        data.append({
                            'date': pd.to_datetime(values[0]),
                            'value': float(values[1])
                        })
                    except:
                        continue
            
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_current_market_rates():
    """Get current market rates for yield curve"""
    # Simulated current market rates (realistic as of 2024)
    rates = {
        '3M': 5.25,
        '6M': 5.15,
        '1Y': 4.95,
        '2Y': 4.75,
        '3Y': 4.60,
        '5Y': 4.45,
        '7Y': 4.40,
        '10Y': 4.35,
        '20Y': 4.50,
        '30Y': 4.55
    }
    return rates

# Financial Calculation Functions
def calculate_bond_price(face_value, coupon_rate, ytm, years):
    """Calculate bond price using present value formula"""
    if ytm == 0:
        return face_value + (coupon_rate * face_value * years)
    
    coupon_payment = face_value * coupon_rate
    pv_coupons = coupon_payment * (1 - (1 + ytm) ** -years) / ytm
    pv_principal = face_value / (1 + ytm) ** years
    return pv_coupons + pv_principal

def calculate_macaulay_duration(face_value, coupon_rate, ytm, years):
    """Calculate Macaulay Duration"""
    if ytm == 0:
        return (years + 1) / 2
    
    coupon_payment = face_value * coupon_rate
    bond_price = calculate_bond_price(face_value, coupon_rate, ytm, years)
    
    weighted_cash_flows = 0
    for t in range(1, int(years) + 1):
        if t == years:
            cash_flow = coupon_payment + face_value
        else:
            cash_flow = coupon_payment
        
        pv_cash_flow = cash_flow / (1 + ytm) ** t
        weighted_cash_flows += t * pv_cash_flow
    
    return weighted_cash_flows / bond_price

def calculate_modified_duration(macaulay_duration, ytm):
    """Calculate Modified Duration"""
    return macaulay_duration / (1 + ytm)

def calculate_convexity(face_value, coupon_rate, ytm, years):
    """Calculate Convexity"""
    if ytm == 0:
        return 0
    
    coupon_payment = face_value * coupon_rate
    bond_price = calculate_bond_price(face_value, coupon_rate, ytm, years)
    
    convexity_sum = 0
    for t in range(1, int(years) + 1):
        if t == years:
            cash_flow = coupon_payment + face_value
        else:
            cash_flow = coupon_payment
        
        pv_cash_flow = cash_flow / (1 + ytm) ** t
        convexity_sum += t * (t + 1) * pv_cash_flow
    
    return convexity_sum / (bond_price * (1 + ytm) ** 2)

# ALM Analysis Functions
def analyze_portfolio_sensitivity(assets, liabilities):
    """Analyze portfolio sensitivity and provide educational insights"""
    if not assets and not liabilities:
        return None
    
    # Calculate portfolio metrics
    total_assets = sum([a['amount'] for a in assets]) if assets else 0
    total_liabilities = sum([l['amount'] for l in liabilities]) if liabilities else 0
    
    # Asset analysis
    fixed_assets = [a for a in assets if a['type'] == 'Fixed Asset']
    variable_assets = [a for a in assets if a['type'] == 'Variable Asset']
    
    fixed_asset_amount = sum([a['amount'] for a in fixed_assets])
    variable_asset_amount = sum([a['amount'] for a in variable_assets])
    
    avg_asset_duration = sum([a['amount'] * a['duration'] for a in assets]) / total_assets if total_assets > 0 else 0
    
    # Liability analysis
    fixed_liabilities = [l for l in liabilities if l['type'] == 'Fixed Liability']
    variable_liabilities = [l for l in liabilities if l['type'] == 'Variable Liability']
    
    fixed_liability_amount = sum([l['amount'] for l in fixed_liabilities])
    variable_liability_amount = sum([l['amount'] for l in variable_liabilities])
    
    avg_liability_duration = sum([l['amount'] * l['duration'] for l in liabilities]) / total_liabilities if total_liabilities > 0 else 0
    
    # Calculate sensitivity metrics
    duration_gap = avg_asset_duration - (total_liabilities / total_assets * avg_liability_duration) if total_assets > 0 else 0
    
    # Determine sensitivity
    if duration_gap > 0.5:
        sensitivity = "Asset Sensitive"
        direction = "benefits from rising rates"
    elif duration_gap < -0.5:
        sensitivity = "Liability Sensitive" 
        direction = "benefits from falling rates"
    else:
        sensitivity = "Neutral"
        direction = "has balanced rate sensitivity"
    
    # Generate educational insights
    insights = {
        'sensitivity': sensitivity,
        'direction': direction,
        'duration_gap': duration_gap,
        'asset_metrics': {
            'total': total_assets,
            'fixed_amount': fixed_asset_amount,
            'variable_amount': variable_asset_amount,
            'fixed_pct': (fixed_asset_amount / total_assets * 100) if total_assets > 0 else 0,
            'avg_duration': avg_asset_duration
        },
        'liability_metrics': {
            'total': total_liabilities,
            'fixed_amount': fixed_liability_amount,
            'variable_amount': variable_liability_amount,
            'fixed_pct': (fixed_liability_amount / total_liabilities * 100) if total_liabilities > 0 else 0,
            'avg_duration': avg_liability_duration
        }
    }
    
    return insights

def generate_alm_recommendations(insights):
    """Generate specific ALM recommendations based on portfolio analysis"""
    if not insights:
        return []
    
    recommendations = []
    sensitivity = insights['sensitivity']
    asset_metrics = insights['asset_metrics']
    liability_metrics = insights['liability_metrics']
    
    if sensitivity == "Asset Sensitive":
        recommendations.extend([
            "🎯 **Product Strategy**: Reduce long-term fixed-rate loan originations when expecting rate declines",
            "🔄 **Asset Mix**: Consider adding more variable-rate assets or shorter-duration securities",
            "🛡️ **Hedging**: Use receive-fixed interest rate swaps to reduce duration risk",
            "💰 **Funding**: Issue longer-term fixed-rate debt to extend liability duration"
        ])
        
        if asset_metrics['fixed_pct'] > 70:
            recommendations.append("⚠️ **High Risk**: Over 70% fixed-rate assets creates significant duration risk")
            
    elif sensitivity == "Liability Sensitive":
        recommendations.extend([
            "🎯 **Product Strategy**: Increase ARM loan originations and reduce short-term deposit rates",
            "🔄 **Asset Mix**: Add longer-duration assets or fixed-rate securities",
            "🛡️ **Hedging**: Use pay-fixed interest rate swaps to increase duration",
            "💰 **Funding**: Promote longer-term CDs to reduce liability repricing frequency"
        ])
        
        if liability_metrics['fixed_pct'] < 30:
            recommendations.append("⚠️ **High Risk**: Heavy reliance on variable-rate funding creates NIM pressure in rising rate environment")
    
    else:  # Neutral
        recommendations.extend([
            "✅ **Well Balanced**: Current portfolio has good interest rate risk balance",
            "🔍 **Monitor**: Watch for changes in product mix that could shift sensitivity",
            "🎯 **Optimize**: Consider tactical adjustments based on rate outlook"
        ])
    
    # Duration-specific recommendations
    if abs(insights['duration_gap']) > 2:
        recommendations.append(f"📊 **Duration Gap**: {insights['duration_gap']:.1f} years indicates significant rate risk")
    
    return recommendations

def simulate_rate_shock(portfolio, rate_change):
    """Simulate interest rate shock on portfolio"""
    results = []
    for item in portfolio:
        if item['type'] == 'Fixed Asset' or item['type'] == 'Fixed Liability':
            duration = item.get('duration', 5)
            price_change = -duration * rate_change
            new_value = item['amount'] * (1 + price_change)
        else:
            new_value = item['amount'] * (1 + 0.1 * rate_change)
        
        results.append({
            'name': item['name'],
            'type': item['type'],
            'original_value': item['amount'],
            'new_value': new_value,
            'change': new_value - item['amount'],
            'change_pct': (new_value - item['amount']) / item['amount'] * 100
        })
    
    return results

# Swap Analytics Functions
class SwapCalculator:
    def __init__(self):
        self.market_rates = get_current_market_rates()
    
    def calculate_swap_npv(self, notional, tenor_years, fixed_rate, floating_index, 
                          pay_fixed, fixed_freq, float_freq, fixed_daycount, float_daycount):
        """Calculate swap NPV using simplified approach"""
        
        # Get floating rate based on index
        floating_rates = {
            'SOFR': self.market_rates.get('3M', 5.25),
            '3M LIBOR': self.market_rates.get('3M', 5.25) + 0.1,  # LIBOR spread
            'Fed Funds': self.market_rates.get('3M', 5.25) - 0.05
        }
        
        current_float_rate = floating_rates.get(floating_index, 5.25) / 100
        fixed_rate_decimal = fixed_rate / 100
        
        # Calculate payment frequencies per year
        freq_map = {'Monthly': 12, 'Quarterly': 4, 'Semi-Annual': 2, 'Annual': 1}
        fixed_payments_per_year = freq_map[fixed_freq]
        float_payments_per_year = freq_map[float_freq]
        
        # Simplified present value calculation
        discount_rate = current_float_rate
        
        # Fixed leg present value
        fixed_payment = notional * fixed_rate_decimal / fixed_payments_per_year
        fixed_pv = 0
        for i in range(1, int(tenor_years * fixed_payments_per_year) + 1):
            t = i / fixed_payments_per_year
            fixed_pv += fixed_payment / (1 + discount_rate) ** t
        
        # Floating leg present value (simplified)
        float_payment = notional * current_float_rate / float_payments_per_year
        float_pv = 0
        for i in range(1, int(tenor_years * float_payments_per_year) + 1):
            t = i / float_payments_per_year
            float_pv += float_payment / (1 + discount_rate) ** t
        
        # NPV calculation based on pay/receive fixed
        if pay_fixed:
            npv = float_pv - fixed_pv  # Receive floating, pay fixed
        else:
            npv = fixed_pv - float_pv  # Receive fixed, pay floating
        
        # Calculate DV01 (simplified)
        shock = 0.0001  # 1 basis point
        npv_up = self._calculate_shocked_npv(notional, tenor_years, fixed_rate, 
                                           current_float_rate + shock, pay_fixed, 
                                           fixed_freq, float_freq)
        dv01 = abs(npv - npv_up)
        
        return {
            'npv': npv,
            'fixed_leg_pv': fixed_pv,
            'float_leg_pv': float_pv,
            'dv01': dv01,
            'current_float_rate': current_float_rate * 100
        }
    
    def _calculate_shocked_npv(self, notional, tenor_years, fixed_rate, shocked_float_rate, 
                              pay_fixed, fixed_freq, float_freq):
        """Helper function for DV01 calculation"""
        freq_map = {'Monthly': 12, 'Quarterly': 4, 'Semi-Annual': 2, 'Annual': 1}
        fixed_payments_per_year = freq_map[fixed_freq]
        float_payments_per_year = freq_map[float_freq]
        
        fixed_rate_decimal = fixed_rate / 100
        discount_rate = shocked_float_rate
        
        # Fixed leg PV
        fixed_payment = notional * fixed_rate_decimal / fixed_payments_per_year
        fixed_pv = sum([fixed_payment / (1 + discount_rate) ** (i / fixed_payments_per_year) 
                       for i in range(1, int(tenor_years * fixed_payments_per_year) + 1)])
        
        # Float leg PV
        float_payment = notional * shocked_float_rate / float_payments_per_year
        float_pv = sum([float_payment / (1 + discount_rate) ** (i / float_payments_per_year) 
                       for i in range(1, int(tenor_years * float_payments_per_year) + 1)])
        
        return float_pv - fixed_pv if pay_fixed else fixed_pv - float_pv
    
    def generate_cashflow_schedule(self, notional, tenor_years, fixed_rate, floating_index,
                                 pay_fixed, fixed_freq, float_freq, start_date):
        """Generate detailed cash flow schedule"""
        
        freq_map = {'Monthly': 12, 'Quarterly': 4, 'Semi-Annual': 2, 'Annual': 1}
        fixed_payments_per_year = freq_map[fixed_freq]
        float_payments_per_year = freq_map[float_freq]
        
        # Generate payment dates
        fixed_dates = pd.date_range(start=start_date, 
                                   periods=int(tenor_years * fixed_payments_per_year) + 1, 
                                   freq=f'{12//fixed_payments_per_year}M')[1:]
        
        float_dates = pd.date_range(start=start_date,
                                   periods=int(tenor_years * float_payments_per_year) + 1,
                                   freq=f'{12//float_payments_per_year}M')[1:]
        
        # Get floating rates
        floating_rates = {
            'SOFR': self.market_rates.get('3M', 5.25),
            '3M LIBOR': self.market_rates.get('3M', 5.25) + 0.1,
            'Fed Funds': self.market_rates.get('3M', 5.25) - 0.05
        }
        current_float_rate = floating_rates.get(floating_index, 5.25)
        
        # Create cash flow schedule
        schedule = []
        
        # Combine and sort all payment dates
        all_dates = sorted(set(list(fixed_dates) + list(float_dates)))
        
        running_npv = 0
        for i, payment_date in enumerate(all_dates):
            row = {
                'Payment_Date': payment_date,
                'Days_From_Start': (payment_date - pd.to_datetime(start_date)).days,
                'Notional': notional,
                'Fixed_Rate': fixed_rate,
                'Floating_Rate': current_float_rate,
                'Fixed_Payment': 0,
                'Floating_Payment': 0,
                'Net_Payment': 0
            }
            
            # Calculate fixed payment if it's a fixed payment date
            if payment_date in fixed_dates:
                days_in_period = 365 / fixed_payments_per_year
                row['Fixed_Payment'] = notional * (fixed_rate / 100) / fixed_payments_per_year
                if not pay_fixed:
                    row['Fixed_Payment'] *= -1  # Receiving fixed
            
            # Calculate floating payment if it's a floating payment date  
            if payment_date in float_dates:
                days_in_period = 365 / float_payments_per_year
                row['Floating_Payment'] = notional * (current_float_rate / 100) / float_payments_per_year
                if pay_fixed:
                    row['Floating_Payment'] *= -1  # Paying floating when pay_fixed=True
            
            row['Net_Payment'] = row['Fixed_Payment'] + row['Floating_Payment']
            
            # Calculate discount factor and present value
            years_from_start = row['Days_From_Start'] / 365
            discount_factor = 1 / (1 + current_float_rate / 100) ** years_from_start
            row['Discount_Factor'] = discount_factor
            row['Present_Value'] = row['Net_Payment'] * discount_factor
            
            running_npv += row['Present_Value']
            row['Running_NPV'] = running_npv
            
            schedule.append(row)
        
        return pd.DataFrame(schedule)

def create_pdf_report(swap_details, cash_flows, npv_analysis):
    """Create PDF swap confirmation report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Interest Rate Swap Confirmation", title_style))
    story.append(Spacer(1, 12))
    
    # Swap details table
    swap_data = [
        ['Trade Details', ''],
        ['Notional Amount', f"${swap_details['notional']:,.0f}"],
        ['Trade Date', swap_details['start_date']],
        ['Maturity Date', swap_details['maturity_date']],
        ['Tenor', f"{swap_details['tenor']} years"],
        ['Direction', 'Pay Fixed' if swap_details['pay_fixed'] else 'Receive Fixed'],
        ['', ''],
        ['Fixed Leg', ''],
        ['Fixed Rate', f"{swap_details['fixed_rate']:.3f}%"],
        ['Payment Frequency', swap_details['fixed_freq']],
        ['Day Count', swap_details['fixed_daycount']],
        ['', ''],
        ['Floating Leg', ''],
        ['Index', swap_details['floating_index']],
        ['Payment Frequency', swap_details['float_freq']],
        ['Day Count', swap_details['float_daycount']],
        ['', ''],
        ['Valuation', ''],
        ['NPV', f"${npv_analysis['npv']:,.0f}"],
        ['DV01', f"${npv_analysis['dv01']:,.0f}"],
    ]
    
    table = Table(swap_data, colWidths=[2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main Application
def main():
    if check_password():
        # Show QuantLib warning only after successful login
        if not QUANTLIB_AVAILABLE:
            st.warning("⚠️ QuantLib not installed. Swap Analytics will use simplified calculations. To get full functionality, install: pip install QuantLib")
        
        st.markdown('<div class="main-header">🏦 Enhanced ALM Teaching Tool with Swap Analytics</div>', unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a section:", [
            "🏦 ALM Sensitivity Analysis",
            "📊 Duration Calculator",
            "⚡ Swap Analytics Tool",
            "📈 Historical Rate Analysis", 
            "🎲 Monte Carlo Simulations",
            "🔍 Advanced Scenarios",
            "📚 Educational Resources"
        ])
        
        if page == "🏦 ALM Sensitivity Analysis":
            alm_sensitivity_page()
        elif page == "📊 Duration Calculator":
            duration_calculator_page()
        elif page == "⚡ Swap Analytics Tool":
            swap_analytics_page()
        elif page == "📈 Historical Rate Analysis":
            historical_analysis_page()
        elif page == "🎲 Monte Carlo Simulations":
            monte_carlo_page()
        elif page == "🔍 Advanced Scenarios":
            advanced_scenarios_page()
        else:
            educational_resources_page()

def alm_sensitivity_page():
    st.markdown('<div class="section-header">ALM Sensitivity Analysis with Intelligent Advisor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Learning Objective:** Understand how different asset and liability compositions affect an institution's 
    sensitivity to interest rate changes, with real-time educational guidance.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio builder
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            st.subheader("📈 Build Your Asset Portfolio")
            
            if 'assets' not in st.session_state:
                st.session_state.assets = []
            
            with st.expander("Add New Asset", expanded=True):
                asset_name = st.text_input("Asset Name", value="", key="asset_name")
                asset_type = st.selectbox("Asset Type", ["Fixed Rate Loan", "Variable Rate Loan", "Securities"], key="asset_type")
                asset_amount = st.number_input("Amount ($)", min_value=0, value=1000000, step=100000, key="asset_amount")
                asset_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=5.0, step=0.25, key="asset_rate")
                
                if asset_type in ["Fixed Rate Loan", "Securities"]:
                    asset_duration = st.number_input("Duration (years)", min_value=0.1, value=5.0, step=0.5, key="asset_duration")
                else:
                    asset_duration = 0.5
                
                if st.button("Add Asset"):
                    if asset_name:
                        st.session_state.assets.append({
                            'name': asset_name,
                            'type': 'Fixed Asset' if asset_type in ['Fixed Rate Loan', 'Securities'] else 'Variable Asset',
                            'amount': asset_amount,
                            'rate': asset_rate,
                            'duration': asset_duration
                        })
                        st.success(f"Added {asset_name}")
                        st.rerun()
        
        with subcol2:
            st.subheader("📉 Build Your Liability Portfolio")
            
            if 'liabilities' not in st.session_state:
                st.session_state.liabilities = []
            
            with st.expander("Add New Liability", expanded=True):
                liability_name = st.text_input("Liability Name", value="", key="liability_name")
                liability_type = st.selectbox("Liability Type", ["Fixed Rate Deposits", "Variable Rate Deposits", "Borrowings"], key="liability_type")
                liability_amount = st.number_input("Amount ($)", min_value=0, value=1000000, step=100000, key="liability_amount")
                liability_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=3.0, step=0.25, key="liability_rate")
                
                if liability_type in ["Fixed Rate Deposits", "Borrowings"]:
                    liability_duration = st.number_input("Duration (years)", min_value=0.1, value=3.0, step=0.5, key="liability_duration")
                else:
                    liability_duration = 0.3
                
                if st.button("Add Liability"):
                    if liability_name:
                        st.session_state.liabilities.append({
                            'name': liability_name,
                            'type': 'Fixed Liability' if liability_type in ['Fixed Rate Deposits', 'Borrowings'] else 'Variable Liability',
                            'amount': liability_amount,
                            'rate': liability_rate,
                            'duration': liability_duration
                        })
                        st.success(f"Added {liability_name}")
                        st.rerun()
    
    with col2:
        # Real-time ALM Advisor
        st.subheader("🎓 ALM Advisor")
        
        # Analyze current portfolio
        insights = analyze_portfolio_sensitivity(
            st.session_state.get('assets', []), 
            st.session_state.get('liabilities', [])
        )
        
        if insights:
            st.markdown('<div class="alm-advisor">', unsafe_allow_html=True)
            st.write("**🔍 Portfolio Analysis:**")
            st.write(f"**Sensitivity:** {insights['sensitivity']}")
            st.write(f"**Duration Gap:** {insights['duration_gap']:.2f} years")
            
            # Asset composition
            asset_metrics = insights['asset_metrics']
            if asset_metrics['total'] > 0:
                st.write(f"**Assets:** {asset_metrics['fixed_pct']:.0f}% fixed-rate (avg duration: {asset_metrics['avg_duration']:.1f}y)")
            
            # Liability composition  
            liability_metrics = insights['liability_metrics']
            if liability_metrics['total'] > 0:
                st.write(f"**Liabilities:** {liability_metrics['fixed_pct']:.0f}% fixed-rate (avg duration: {liability_metrics['avg_duration']:.1f}y)")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Educational explanation
            st.markdown('<div class="alm-advisor">', unsafe_allow_html=True)
            st.write("**📚 Why This Matters:**")
            
            if insights['sensitivity'] == "Asset Sensitive":
                st.write("Your institution **benefits from rising rates** because:")
                st.write("• Assets reprice/mature faster than liabilities")
                st.write("• Interest income increases more than funding costs")
                st.write("• Duration gap is positive (assets shorter duration)")
                
                st.write("**Risk in falling rates:**")
                st.write("• Asset yields drop faster than funding costs")
                st.write("• Net interest margin compression")
                
            elif insights['sensitivity'] == "Liability Sensitive":
                st.write("Your institution **benefits from falling rates** because:")
                st.write("• Liabilities reprice/mature faster than assets")
                st.write("• Funding costs decrease more than asset yields")
                st.write("• Duration gap is negative (liabilities shorter duration)")
                
                st.write("**Risk in rising rates:**")
                st.write("• Funding costs increase faster than asset yields")
                st.write("• Net interest margin compression")
            
            else:
                st.write("Your portfolio is **well-balanced** with minimal rate sensitivity")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            recommendations = generate_alm_recommendations(insights)
            if recommendations:
                st.markdown('<div class="recommendation">', unsafe_allow_html=True)
                st.write("**💡 Strategic Recommendations:**")
                for rec in recommendations:
                    st.write(rec)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.info("Add assets and liabilities to see real-time ALM analysis and recommendations!")
    
    # Display current portfolio and shock analysis
    if st.session_state.get('assets') or st.session_state.get('liabilities'):
        st.markdown('<div class="section-header">Current Portfolio & Shock Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('assets'):
                st.subheader("Assets")
                assets_df = pd.DataFrame(st.session_state.assets)
                st.dataframe(assets_df, use_container_width=True)
                
                total_assets = sum([a['amount'] for a in st.session_state.assets])
                st.metric("Total Assets", f"${total_assets:,.0f}")
        
        with col2:
            if st.session_state.get('liabilities'):
                st.subheader("Liabilities")
                liabilities_df = pd.DataFrame(st.session_state.liabilities)
                st.dataframe(liabilities_df, use_container_width=True)
                
                total_liabilities = sum([l['amount'] for l in st.session_state.liabilities])
                st.metric("Total Liabilities", f"${total_liabilities:,.0f}")
        
        # Interest Rate Shock Analysis
        st.subheader("Interest Rate Shock Analysis")
        
        rate_change = st.slider("Interest Rate Change (basis points)", 
                               min_value=-500, max_value=500, value=100, step=25)
        rate_change_decimal = rate_change / 10000
        
        if st.button("Run Shock Analysis"):
            portfolio = st.session_state.get('assets', []) + st.session_state.get('liabilities', [])
            results = simulate_rate_shock(portfolio, rate_change_decimal)
            
            asset_impact = sum([r['change'] for r in results if 'Asset' in r['type']])
            liability_impact = sum([r['change'] for r in results if 'Liability' in r['type']])
            net_impact = asset_impact - liability_impact
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Asset Value Change", f"${asset_impact:,.0f}")
            with col2:
                st.metric("Liability Value Change", f"${liability_impact:,.0f}")
            with col3:
                st.metric("Net Economic Value Change", f"${net_impact:,.0f}")
            
            # Enhanced sensitivity analysis with explanations
            if net_impact > 0 and rate_change > 0:
                sensitivity = "Asset Sensitive"
                explanation = "✅ Your institution **benefits** from rising rates because asset values increase more than liability values."
                css_class = "sensitivity-positive"
            elif net_impact < 0 and rate_change > 0:
                sensitivity = "Liability Sensitive"
                explanation = "❌ Your institution **suffers** from rising rates because liability costs increase more than asset income."
                css_class = "sensitivity-negative"
            elif net_impact > 0 and rate_change < 0:
                sensitivity = "Liability Sensitive"
                explanation = "✅ Your institution **benefits** from falling rates because liability costs decrease more than asset income."
                css_class = "sensitivity-positive"
            else:
                sensitivity = "Asset Sensitive"
                explanation = "❌ Your institution **suffers** from falling rates because asset income decreases more than liability costs."
                css_class = "sensitivity-negative"
            
            st.markdown(f'<div class="{css_class}">Interest Rate Sensitivity: {sensitivity}</div>', unsafe_allow_html=True)
            st.write(explanation)
            
            # Detailed results
            st.subheader("Detailed Impact Analysis")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
    
    # Clear portfolio buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Assets"):
            st.session_state.assets = []
            st.rerun()
    with col2:
        if st.button("Clear Liabilities"):
            st.session_state.liabilities = []
            st.rerun()

def swap_analytics_page():
    st.markdown('<div class="section-header">Swap Analytics Tool</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Learning Objective:** Build, price, and analyze interest rate swaps for ALM hedging and risk management.
    Learn how swaps work, their valuation, and risk characteristics.
    """)
    
    # Initialize swap calculator
    if 'swap_calc' not in st.session_state:
        st.session_state.swap_calc = SwapCalculator()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Swap Builder")
        
        # Basic swap parameters
        with st.expander("📋 Basic Terms", expanded=True):
            notional = st.number_input("Notional Amount ($)", value=10000000, step=1000000, format="%d")
            tenor_years = st.number_input("Tenor (Years)", min_value=1, max_value=30, value=5, step=1)
            
            trade_date = st.date_input("Trade Date", value=date.today())
            
            direction = st.selectbox("Direction", ["Pay Fixed / Receive Float", "Receive Fixed / Pay Float"])
            pay_fixed = direction == "Pay Fixed / Receive Float"
            
            fixed_rate = st.number_input("Fixed Rate (%)", min_value=0.0, value=4.5, step=0.01, format="%.3f")
        
        # Fixed leg specifications
        with st.expander("🔒 Fixed Leg Details"):
            fixed_freq = st.selectbox("Fixed Payment Frequency", 
                                    ["Monthly", "Quarterly", "Semi-Annual", "Annual"], 
                                    index=2)
            fixed_daycount = st.selectbox("Fixed Day Count", 
                                        ["30/360", "ACT/360", "ACT/365"], 
                                        index=0)
        
        # Floating leg specifications
        with st.expander("📈 Floating Leg Details"):
            floating_index = st.selectbox("Floating Rate Index", 
                                        ["SOFR", "3M LIBOR", "Fed Funds"], 
                                        index=0)
            float_freq = st.selectbox("Float Payment Frequency", 
                                    ["Monthly", "Quarterly", "Semi-Annual", "Annual"], 
                                    index=1)
            float_daycount = st.selectbox("Float Day Count", 
                                        ["ACT/360", "ACT/365", "30/360"], 
                                        index=0)
        
        # Market data section
        st.subheader("📊 Market Data")
        
        if st.button("🔄 Refresh Market Data"):
            st.session_state.swap_calc.market_rates = get_current_market_rates()
            st.success("Market data refreshed!")
        
        # Display current rates
        st.write("**Current Yield Curve:**")
        for tenor, rate in st.session_state.swap_calc.market_rates.items():
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"{tenor}:")
            with col_b:
                new_rate = st.number_input(f"", value=rate, step=0.01, format="%.2f", 
                                         key=f"rate_{tenor}", label_visibility="collapsed")
                st.session_state.swap_calc.market_rates[tenor] = new_rate
    
    with col2:
        st.subheader("📈 Swap Valuation & Analytics")
        
        # Calculate swap metrics
        swap_metrics = st.session_state.swap_calc.calculate_swap_npv(
            notional, tenor_years, fixed_rate, floating_index, pay_fixed,
            fixed_freq, float_freq, fixed_daycount, float_daycount
        )
        
        # Display key metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("NPV", f"${swap_metrics['npv']:,.0f}")
        with col_b:
            st.metric("DV01", f"${swap_metrics['dv01']:,.0f}")
        with col_c:
            st.metric("Current Float Rate", f"{swap_metrics['current_float_rate']:.2f}%")
        
        # Swap pricing breakdown
        st.markdown('<div class="swap-pricing">', unsafe_allow_html=True)
        st.write("**🔍 Valuation Breakdown:**")
        st.write(f"**Fixed Leg PV:** ${swap_metrics['fixed_leg_pv']:,.0f}")
        st.write(f"**Floating Leg PV:** ${swap_metrics['float_leg_pv']:,.0f}")
        
        direction_text = "Pay Fixed" if pay_fixed else "Receive Fixed"
        st.write(f"**Direction:** {direction_text}")
        
        if pay_fixed:
            st.write(f"**NPV Calculation:** Float PV - Fixed PV = ${swap_metrics['npv']:,.0f}")
        else:
            st.write(f"**NPV Calculation:** Fixed PV - Float PV = ${swap_metrics['npv']:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # DV01 explanation
        st.markdown('<div class="risk-attribution">', unsafe_allow_html=True)
        st.write("**📚 Understanding DV01:**")
        st.write("**Formula:** DV01 = |NPV(r) - NPV(r + 0.01%)|")
        st.write("**Meaning:** Dollar change in swap value for 1 basis point rate move")
        st.write(f"**Your Swap:** ${swap_metrics['dv01']:,.0f} DV01 means a 1bp rate increase changes value by ~${swap_metrics['dv01']:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Rate shock analysis
        st.subheader("⚡ Rate Shock Analysis")
        
        shock_scenarios = [-200, -100, -50, -25, 0, 25, 50, 100, 200]
        scenario_results = []
        
        for shock_bp in shock_scenarios:
            shock_decimal = shock_bp / 10000
            
            # Calculate new floating rate
            base_float_rate = swap_metrics['current_float_rate'] / 100
            shocked_float_rate = base_float_rate + shock_decimal
            
            # Simplified NPV calculation with shocked rate
            shocked_npv = st.session_state.swap_calc._calculate_shocked_npv(
                notional, tenor_years, fixed_rate, shocked_float_rate, pay_fixed,
                fixed_freq, float_freq
            )
            
            pnl = shocked_npv - swap_metrics['npv']
            
            scenario_results.append({
                'Rate Shock (bp)': shock_bp,
                'New NPV ($)': shocked_npv,
                'P&L ($)': pnl,
                'P&L (%)': (pnl / notional) * 100
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        st.dataframe(scenario_df, use_container_width=True)
        
        # P&L visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scenario_df['Rate Shock (bp)'],
            y=scenario_df['P&L ($)'],
            mode='lines+markers',
            name='P&L',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, color='#2563eb')
        ))
        
        fig.update_layout(
            title='Swap P&L vs Rate Shocks',
            xaxis_title='Rate Shock (basis points)',
            yaxis_title='P&L ($)',
            height=400,
            hovermode='x',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cash flow analysis section
    st.markdown('<div class="section-header">Cash Flow Analysis</div>', unsafe_allow_html=True)
    
    if st.button("📊 Generate Cash Flow Schedule"):
        with st.spinner("Generating detailed cash flow schedule..."):
            # Calculate maturity date
            maturity_date = trade_date + timedelta(days=int(tenor_years * 365))
            
            # Generate cash flow schedule
            cash_flows = st.session_state.swap_calc.generate_cashflow_schedule(
                notional, tenor_years, fixed_rate, floating_index, pay_fixed,
                fixed_freq, float_freq, trade_date
            )
            
            # Display cash flows
            st.subheader("💰 Detailed Cash Flow Schedule")
            
            # Format the dataframe for better display
            display_df = cash_flows.copy()
            display_df['Payment_Date'] = display_df['Payment_Date'].dt.strftime('%Y-%m-%d')
            display_df['Fixed_Payment'] = display_df['Fixed_Payment'].apply(lambda x: f"${x:,.0f}")
            display_df['Floating_Payment'] = display_df['Floating_Payment'].apply(lambda x: f"${x:,.0f}")
            display_df['Net_Payment'] = display_df['Net_Payment'].apply(lambda x: f"${x:,.0f}")
            display_df['Present_Value'] = display_df['Present_Value'].apply(lambda x: f"${x:,.0f}")
            display_df['Running_NPV'] = display_df['Running_NPV'].apply(lambda x: f"${x:,.0f}")
            display_df['Discount_Factor'] = display_df['Discount_Factor'].apply(lambda x: f"{x:.6f}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_fixed = cash_flows['Fixed_Payment'].sum()
                st.metric("Total Fixed Payments", f"${total_fixed:,.0f}")
            
            with col2:
                total_floating = cash_flows['Floating_Payment'].sum()
                st.metric("Total Floating Payments", f"${total_floating:,.0f}")
            
            with col3:
                total_net = cash_flows['Net_Payment'].sum()
                st.metric("Total Net Payments", f"${total_net:,.0f}")
            
            with col4:
                final_npv = cash_flows['Running_NPV'].iloc[-1]
                st.metric("Final NPV", f"${final_npv:,.0f}")
            
            # Cash flow timeline visualization
            st.subheader("📅 Cash Flow Timeline")
            
            fig = go.Figure()
            
            # Fixed payments
            fig.add_trace(go.Bar(
                x=cash_flows['Payment_Date'],
                y=cash_flows['Fixed_Payment'],
                name='Fixed Payments',
                marker_color='#dc3545' if pay_fixed else '#28a745',
                opacity=0.7
            ))
            
            # Floating payments
            fig.add_trace(go.Bar(
                x=cash_flows['Payment_Date'],
                y=cash_flows['Floating_Payment'],
                name='Floating Payments',
                marker_color='#28a745' if pay_fixed else '#dc3545',
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Cash Flow Timeline',
                xaxis_title='Payment Date',
                yaxis_title='Cash Flow ($)',
                barmode='relative',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Excel export
                if st.button("📊 Export to Excel"):
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        cash_flows.to_excel(writer, sheet_name='Cash_Flows', index=False)
                        
                        # Add swap summary sheet
                        swap_summary = pd.DataFrame([
                            ['Notional', f"${notional:,}"],
                            ['Tenor', f"{tenor_years} years"],
                            ['Direction', direction],
                            ['Fixed Rate', f"{fixed_rate:.3f}%"],
                            ['Floating Index', floating_index],
                            ['NPV', f"${swap_metrics['npv']:,.0f}"],
                            ['DV01', f"${swap_metrics['dv01']:,.0f}"]
                        ], columns=['Parameter', 'Value'])
                        swap_summary.to_excel(writer, sheet_name='Swap_Summary', index=False)
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="💾 Download Excel File",
                        data=output.getvalue(),
                        file_name=f"swap_analysis_{trade_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                # CSV export
                csv_data = cash_flows.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"swap_cashflows_{trade_date}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # PDF export
                if st.button("📋 Generate PDF Report"):
                    swap_details = {
                        'notional': notional,
                        'start_date': trade_date.strftime('%Y-%m-%d'),
                        'maturity_date': maturity_date.strftime('%Y-%m-%d'),
                        'tenor': tenor_years,
                        'pay_fixed': pay_fixed,
                        'fixed_rate': fixed_rate,
                        'fixed_freq': fixed_freq,
                        'fixed_daycount': fixed_daycount,
                        'floating_index': floating_index,
                        'float_freq': float_freq,
                        'float_daycount': float_daycount
                    }
                    
                    pdf_buffer = create_pdf_report(swap_details, cash_flows, swap_metrics)
                    
                    st.download_button(
                        label="📋 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"swap_confirmation_{trade_date}.pdf",
                        mime="application/pdf"
                    )

def duration_calculator_page():
    st.markdown('<div class="section-header">Bond Duration Calculator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Learning Objective:** Understand different duration measures and how they relate to interest rate sensitivity.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Bond Parameters")
        
        face_value = st.number_input("Face Value ($)", min_value=100, value=1000, step=100)
        coupon_rate = st.number_input("Annual Coupon Rate (%)", min_value=0.0, value=5.0, step=0.25) / 100
        ytm = st.number_input("Yield to Maturity (%)", min_value=0.1, value=6.0, step=0.25) / 100
        years = st.number_input("Years to Maturity", min_value=0.5, value=10.0, step=0.5)
        
        # Calculate bond metrics
        bond_price = calculate_bond_price(face_value, coupon_rate, ytm, years)
        macaulay_duration = calculate_macaulay_duration(face_value, coupon_rate, ytm, years)
        modified_duration = calculate_modified_duration(macaulay_duration, ytm)
        convexity = calculate_convexity(face_value, coupon_rate, ytm, years)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Bond Price", f"${bond_price:.2f}")
        st.metric("Macaulay Duration", f"{macaulay_duration:.3f} years")
        st.metric("Modified Duration", f"{modified_duration:.3f}")
        st.metric("Convexity", f"{convexity:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Duration Sensitivity Analysis")
        
        # Create yield change scenarios
        yield_changes = np.arange(-0.03, 0.035, 0.005)
        
        prices_duration = []
        prices_convexity = []
        actual_prices = []
        
        for dy in yield_changes:
            # Duration approximation
            price_duration = bond_price * (1 - modified_duration * dy)
            prices_duration.append(price_duration)
            
            # Duration + Convexity approximation
            price_convexity = bond_price * (1 - modified_duration * dy + 0.5 * convexity * dy**2)
            prices_convexity.append(price_convexity)
            
            # Actual price
            new_ytm = ytm + dy
            if new_ytm <= 0:
                new_ytm = 0.001
            actual_price = calculate_bond_price(face_value, coupon_rate, new_ytm, years)
            actual_prices.append(actual_price)
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=yield_changes * 100,
            y=actual_prices,
            mode='lines',
            name='Actual Price',
            line=dict(color='#1e3a8a', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=yield_changes * 100,
            y=prices_duration,
            mode='lines',
            name='Duration Approximation',
            line=dict(color='#dc3545', dash='dash', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=yield_changes * 100,
            y=prices_convexity,
            mode='lines',
            name='Duration + Convexity',
            line=dict(color='#28a745', dash='dot', width=2)
        ))
        
        fig.update_layout(
            title='Bond Price Sensitivity to Yield Changes',
            xaxis_title='Yield Change (%)',
            yaxis_title='Bond Price ($)',
            hovermode='x unified',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.markdown("""
        **Key Insights:**
        - **Duration** provides a linear approximation of price sensitivity
        - **Convexity** captures the curvature, improving accuracy for large yield changes
        - Longer duration = higher interest rate sensitivity
        - Higher convexity = more favorable price performance (especially when rates fall)
        """)

def historical_analysis_page():
    st.markdown('<div class="section-header">Historical Interest Rate Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Learning Objective:** Understand how interest rate environments have changed over time and 
    how different ALM strategies would have performed historically.
    """)
    
    # Simple historical rate simulation
    st.info("📊 **Note**: This demo uses simulated historical data. In production, this would connect to FRED API for real historical rates.")
    
    # Create simulated historical data
    dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='M')
    
    # Simulate different rate cycles
    fed_funds = []
    treasury_10y = []
    
    for i, date in enumerate(dates):
        # Create realistic rate cycles
        cycle = i / 50  # Slower cycle
        noise = np.random.normal(0, 0.1)
        
        # Fed funds simulation
        base_fed = 3 + 2 * np.sin(cycle) + noise
        if date.year >= 2008 and date.year <= 2015:  # Financial crisis period
            base_fed = max(0.1, base_fed - 3)
        elif date.year >= 2020 and date.year <= 2021:  # COVID period
            base_fed = max(0.1, base_fed - 2)
        
        fed_funds.append(max(0.1, base_fed))
        
        # 10Y treasury (usually higher than fed funds)
        treasury_10y.append(base_fed + 1 + 0.5 * np.sin(cycle + 1) + noise * 0.5)
    
    # Create DataFrame
    hist_data = pd.DataFrame({
        'date': dates,
        'fed_funds': fed_funds,
        'treasury_10y': treasury_10y
    })
    
    # Display historical chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_data['date'],
        y=hist_data['fed_funds'],
        mode='lines',
        name='Fed Funds Rate',
        line=dict(color='#3b82f6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=hist_data['date'],
        y=hist_data['treasury_10y'],
        mode='lines',
        name='10-Year Treasury',
        line=dict(color='#dc3545', width=2)
    ))
    
    fig.update_layout(
        title='Historical Interest Rates (Simulated)',
        xaxis_title='Date',
        yaxis_title='Interest Rate (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical scenario testing
    st.subheader("🕰️ Historical Scenario Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenarios = {
            "2008 Financial Crisis": (2008, 2009),
            "Post-Crisis Recovery": (2010, 2015),
            "Rate Normalization": (2015, 2019),
            "COVID Response": (2020, 2021),
            "Recent Tightening": (2022, 2024)
        }
        
        selected_scenario = st.selectbox("Choose Historical Period:", list(scenarios.keys()))
        start_year, end_year = scenarios[selected_scenario]
        
        # Filter data for selected period
        period_data = hist_data[
            (hist_data['date'].dt.year >= start_year) & 
            (hist_data['date'].dt.year <= end_year)
        ]
        
        if not period_data.empty:
            rate_change = period_data['fed_funds'].iloc[-1] - period_data['fed_funds'].iloc[0]
            
            st.write(f"**Period:** {start_year} - {end_year}")
            st.write(f"**Fed Funds Change:** {rate_change:+.2f}%")
            
            # Portfolio for testing
            asset_duration = st.slider("Asset Duration", 1.0, 15.0, 7.0, 0.5)
            liability_duration = st.slider("Liability Duration", 0.1, 10.0, 2.0, 0.5)
            
    with col2:
        if st.button("Analyze Historical Impact"):
            # Calculate approximate impact
            duration_gap = asset_duration - liability_duration
            impact_pct = -duration_gap * (rate_change / 100)
            
            st.markdown('<div class="alm-advisor">', unsafe_allow_html=True)
            st.write(f"**Historical Analysis for {selected_scenario}:**")
            st.write(f"**Rate Change:** {rate_change:+.2f}%")
            st.write(f"**Duration Gap:** {duration_gap:.1f} years")
            st.write(f"**Estimated Portfolio Impact:** {impact_pct*100:+.2f}%")
            
            if impact_pct > 0:
                st.success("✅ Portfolio would have BENEFITED during this period")
            else:
                st.error("❌ Portfolio would have SUFFERED during this period")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show period chart
            fig_period = go.Figure()
            
            fig_period.add_trace(go.Scatter(
                x=period_data['date'],
                y=period_data['fed_funds'],
                mode='lines',
                name='Fed Funds Rate',
                line=dict(color='#3b82f6', width=2)
            ))
            
            fig_period.update_layout(
                title=f'Interest Rates During {selected_scenario}',
                xaxis_title='Date',
                yaxis_title='Rate (%)',
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_period, use_container_width=True)

def monte_carlo_page():
    st.markdown('<div class="section-header">Monte Carlo Interest Rate Simulations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Learning Objective:** Use Monte Carlo simulations to understand the range of possible 
    outcomes for your ALM strategy under different interest rate scenarios.
    """)
    
    def vasicek_simulation(r0, kappa, theta, sigma, T, n_paths, n_steps):
        """Vasicek model simulation"""
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0
        
        for i in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i + 1] = (paths[:, i] + 
                              kappa * (theta - paths[:, i]) * dt + 
                              sigma * dW)
        
        return paths
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        current_rate = st.number_input("Current Rate (%)", value=5.0, step=0.25) / 100
        long_term_mean = st.number_input("Long-term Mean (%)", value=4.0, step=0.25) / 100
        mean_reversion_speed = st.number_input("Mean Reversion Speed", value=0.3, step=0.1)
        volatility = st.number_input("Volatility", value=0.02, step=0.01)
        
        time_horizon = st.number_input("Time Horizon (years)", value=5, step=1)
        n_simulations = st.selectbox("Number of Simulations", [100, 500, 1000], index=1)
        
        # Portfolio parameters
        st.subheader("Portfolio Parameters")
        portfolio_value = st.number_input("Portfolio Value ($M)", value=100, step=10) * 1e6
        portfolio_duration = st.number_input("Portfolio Duration", value=5.0, step=0.5)
        
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner(f"Running {n_simulations} simulations..."):
                # Generate rate paths
                n_steps = time_horizon * 12
                rate_paths = vasicek_simulation(
                    current_rate, mean_reversion_speed, long_term_mean,
                    volatility, time_horizon, n_simulations, n_steps
                )
                
                # Calculate portfolio values
                portfolio_values = np.zeros((n_simulations, n_steps + 1))
                portfolio_values[:, 0] = portfolio_value
                
                for path in range(n_simulations):
                    for step in range(1, n_steps + 1):
                        rate_change = rate_paths[path, step] - rate_paths[path, step - 1]
                        value_change = -portfolio_duration * rate_change * portfolio_values[path, step - 1]
                        portfolio_values[path, step] = portfolio_values[path, step - 1] + value_change
                
                # Store results
                st.session_state.mc_results = {
                    'rate_paths': rate_paths,
                    'portfolio_values': portfolio_values,
                    'time_steps': np.linspace(0, time_horizon, n_steps + 1)
                }
    
    with col2:
        if 'mc_results' in st.session_state:
            results = st.session_state.mc_results
            
            # Summary statistics
            final_values = results['portfolio_values'][:, -1]
            value_changes = final_values - portfolio_value
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Mean Change", f"${np.mean(value_changes):,.0f}")
                st.metric("Std Dev", f"${np.std(value_changes):,.0f}")
            
            with col_b:
                st.metric("95% VaR", f"${np.percentile(value_changes, 5):,.0f}")
                st.metric("99% VaR", f"${np.percentile(value_changes, 1):,.0f}")
            
            with col_c:
                prob_loss = np.mean(value_changes < 0) * 100
                st.metric("Prob of Loss", f"{prob_loss:.1f}%")
                
            # Distribution chart
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=value_changes,
                nbinsx=50,
                name='Value Changes',
                opacity=0.7,
                marker_color='#3b82f6'
            ))
            
            fig.add_vline(x=np.percentile(value_changes, 5), line_dash="dash", 
                         line_color="#dc3545", annotation_text="95% VaR")
            
            fig.update_layout(
                title='Distribution of Portfolio Value Changes',
                xaxis_title='Value Change ($)',
                yaxis_title='Frequency',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a Monte Carlo simulation to see results")

def advanced_scenarios_page():
    st.markdown('<div class="section-header">Advanced Scenario Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Learning Objective:** Explore complex interest rate scenarios including yield curve changes, 
    stress tests, and regulatory scenarios.
    """)
    
    scenario_type = st.selectbox("Select Scenario Type:", [
        "Yield Curve Scenarios",
        "Regulatory Stress Tests", 
        "Economic Shock Scenarios"
    ])
    
    if scenario_type == "Yield Curve Scenarios":
        st.subheader("📈 Yield Curve Scenario Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario = st.selectbox("Select Scenario:", [
                "Parallel Shift",
                "Steepening", 
                "Flattening",
                "Twist (2s10s)"
            ])
            
            magnitude = st.slider("Magnitude (basis points)", -300, 300, 100, 25)
            
            # Initial yield curve
            maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
            initial_rates = np.array([4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4])
            
            # Apply scenario
            if scenario == "Parallel Shift":
                new_rates = initial_rates + magnitude / 100
            elif scenario == "Steepening":
                adjustment = np.linspace(-magnitude/200, magnitude/100, len(maturities))
                new_rates = initial_rates + adjustment
            elif scenario == "Flattening":
                adjustment = np.linspace(magnitude/100, -magnitude/200, len(maturities))
                new_rates = initial_rates + adjustment
            else:  # Twist
                adjustment = np.zeros(len(maturities))
                idx_2y = np.argmin(np.abs(maturities - 2))
                idx_10y = np.argmin(np.abs(maturities - 10))
                adjustment[idx_2y] = -magnitude / 100
                adjustment[idx_10y] = magnitude / 100
                new_rates = initial_rates + adjustment
        
        with col2:
            # Visualize curves
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=maturities,
                y=initial_rates,
                mode='lines+markers',
                name='Initial Curve',
                line=dict(color='#3b82f6', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=maturities,
                y=new_rates,
                mode='lines+markers',
                name=f'{scenario} ({magnitude:+}bp)',
                line=dict(color='#dc3545', width=2)
            ))
            
            fig.update_layout(
                title='Yield Curve Scenario',
                xaxis_title='Maturity (Years)',
                yaxis_title='Yield (%)',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif scenario_type == "Regulatory Stress Tests":
        st.subheader("🏛️ Regulatory Stress Test Scenarios")
        
        stress_test = st.selectbox("Select Stress Test:", [
            "CCAR Severely Adverse",
            "Basel III Rate Shock"
        ])
        
        if stress_test == "CCAR Severely Adverse":
            st.write("**CCAR Severely Adverse Scenario:**")
            st.write("- Severe global recession")
            st.write("- Fed Funds Rate drops to 0.25%")
            st.write("- 10Y Treasury falls 100bp")
            st.write("- Credit spreads widen significantly")
            
            # Sample portfolio impact
            if st.button("Calculate CCAR Impact"):
                st.info("🔍 **Analysis**: Asset-heavy institutions would see duration losses offset by lower funding costs")
        
        else:  # Basel III
            st.write("**Basel III Interest Rate Risk in Banking Book:**")
            st.write("- ±200bp parallel shock scenarios")
            st.write("- Steepener and flattener scenarios")
            st.write("- Assessment of EVE and NII impact")
            
            if st.button("Run Basel III Scenarios"):
                st.info("🔍 **Analysis**: Institutions must maintain adequate capital for 200bp rate shocks")
    
    else:  # Economic Shock Scenarios
        st.subheader("💥 Economic Shock Scenarios")
        
        shock_scenarios = {
            "1980s Volcker Shock": "Fed Funds raised to 20% to combat inflation",
            "2008 Financial Crisis": "Credit markets freeze, emergency rate cuts",
            "COVID-19 Pandemic": "Emergency policies, supply chain disruption",
            "Hypothetical Cyber Crisis": "Financial infrastructure attack"
        }
        
        selected_shock = st.selectbox("Select Shock:", list(shock_scenarios.keys()))
        
        st.write(f"**{selected_shock}:**")
        st.write(shock_scenarios[selected_shock])
        
        if st.button("Analyze Impact"):
            if selected_shock == "1980s Volcker Shock":
                st.error("🔴 **High Impact**: S&Ls with duration mismatch faced severe losses")
            elif selected_shock == "2008 Financial Crisis":
                st.warning("🟡 **Credit Focus**: Interest rate risk secondary to credit losses")
            elif selected_shock == "COVID-19 Pandemic":
                st.info("🔵 **Policy Response**: Massive stimulus, rates to zero")
            else:
                st.error("🔴 **Unknown Territory**: Flight to quality, operational disruption")

def educational_resources_page():
    st.markdown('<div class="section-header">Educational Resources</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["ALM Concepts", "Swap Fundamentals", "Risk Management", "Formulas"])
    
    with tab1:
        st.markdown("""
        ## 🎯 Asset-Liability Management Concepts
        
        ### Interest Rate Sensitivity
        - **Asset Sensitive**: Benefits when rates rise (short duration liabilities)
        - **Liability Sensitive**: Benefits when rates fall (short duration assets)
        - **Duration Gap**: Key measure of rate sensitivity
        
        ### Portfolio Analysis
        - **Repricing Analysis**: When assets/liabilities reset rates
        - **Duration Analysis**: Price sensitivity to rate changes
        - **Simulation Analysis**: Multiple scenario testing
        
        ### ALM Strategies
        - **Natural Hedging**: Match asset/liability characteristics
        - **Derivative Hedging**: Use swaps, caps, floors
        - **Product Mix Management**: Adjust new business mix
        """)
    
    with tab2:
        st.markdown("""
        ## ⚡ Interest Rate Swap Fundamentals
        
        ### What is a Swap?
        An agreement to exchange cash flows based on different interest rate calculations
        
        ### Key Components
        - **Notional Amount**: Reference amount for calculations
        - **Fixed Leg**: Pays/receives fixed rate
        - **Floating Leg**: Pays/receives variable rate (SOFR, LIBOR, etc.)
        - **Tenor**: Length of the swap agreement
        
        ### Valuation Concepts
        - **NPV**: Net present value of all future cash flows
        - **DV01**: Dollar value change per 1bp rate move
        - **Curve Risk**: Sensitivity to different parts of yield curve
        
        ### ALM Applications
        - **Asset Conversion**: Convert fixed assets to floating exposure
        - **Liability Management**: Extend liability duration
        - **Gap Management**: Reduce duration gap
        """)
    
    with tab3:
        st.markdown("""
        ## 🛡️ Risk Management Framework
        
        ### Interest Rate Risk Types
        - **Repricing Risk**: Timing mismatches in rate resets
        - **Yield Curve Risk**: Non-parallel curve movements
        - **Basis Risk**: Spread changes between indices
        - **Optionality Risk**: Embedded options in products
        
        ### Risk Measurement
        - **Gap Analysis**: Maturity/repricing buckets
        - **Duration Analysis**: Price sensitivity measures
        - **Value-at-Risk**: Potential losses at confidence levels
        - **Stress Testing**: Extreme scenario analysis
        
        ### Risk Management Tools
        - **Interest Rate Swaps**: Most common hedge
        - **Interest Rate Caps/Floors**: Protect against rate moves
        - **Swaptions**: Options on swaps
        - **Asset/Liability Mix**: Natural hedging
        """)
    
    with tab4:
        st.markdown("""
        ## 📐 Key Formulas
        
        ### Duration Calculations
        ```
        Macaulay Duration = Σ[t × PV(CFt)] / Bond Price
        Modified Duration = Macaulay Duration / (1 + y)
        DV01 = Modified Duration × Bond Price × 0.0001
        ```
        
        ### Swap Valuation
        ```
        Swap NPV = PV(Fixed Leg) - PV(Floating Leg)
        Fixed Leg PV = Σ[Fixed Payment / (1 + r)^t]
        Floating Leg PV = Notional × (1 - DF(maturity))
        ```
        
        ### Risk Metrics
        ```
        Duration Gap = Asset Duration - (Liabilities/Assets) × Liability Duration
        Value Change ≈ -Duration × ΔRate × Portfolio Value
        Convexity Adjustment = 0.5 × Convexity × (ΔRate)²
        ```
        
        ### Monte Carlo
        ```
        Vasicek Model: dr = κ(θ - r)dt + σdW
        VaR = μ + σ × Φ⁻¹(α)
        Expected Shortfall = E[X | X ≤ VaR]
        ```
        """)

if __name__ == "__main__":
    main()