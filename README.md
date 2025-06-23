# 🏦 ALM Educator

A comprehensive Asset-Liability Management (ALM) educational platform built with Streamlit, featuring interactive analytics, swap pricing, and risk management tools for financial education.

## 🌟 Features

### 🔐 Secure Access
- Password-protected educational platform
- Professional financial industry styling
- Educational disclaimer and compliance

### 📊 Core Modules

1. **ALM Sensitivity Analysis**
   - Interactive portfolio building
   - Real-time sensitivity analysis
   - Intelligent ALM advisor with recommendations
   - Interest rate shock testing

2. **Duration Calculator**
   - Bond price sensitivity analysis
   - Macaulay and Modified Duration calculations
   - Convexity analysis with visualizations
   - Duration vs. actual price comparisons

3. **Swap Analytics Tool**
   - Complete interest rate swap builder
   - NPV and DV01 calculations
   - Cash flow schedule generation
   - Export to Excel, CSV, and PDF

4. **Historical Rate Analysis**
   - Simulated historical rate scenarios
   - Portfolio backtesting capabilities
   - Multiple economic cycle analysis

5. **Monte Carlo Simulations**
   - Vasicek interest rate model
   - Portfolio value distributions
   - Value-at-Risk calculations
   - Risk scenario analysis

6. **Advanced Scenarios**
   - Yield curve scenario testing
   - Regulatory stress tests (CCAR, Basel III)
   - Economic shock simulations

7. **Educational Resources**
   - Comprehensive learning materials
   - Formula references
   - Risk management frameworks

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/alm-educator.git
cd alm-educator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Optional - Install QuantLib for advanced swap analytics:**
```bash
# Windows
pip install QuantLib-Python

# macOS
brew install quantlib
pip install QuantLib-Python

# Linux (Ubuntu/Debian)
sudo apt-get install libquantlib0-dev
pip install QuantLib-Python
```

4. **Run the application:**
```bash
streamlit run irr_educate.py
```

5. **Access the platform:**
   - Open your browser to `http://localhost:8501`
   - Enter password: `almeducate2025!`

## 🎯 Educational Objectives

This tool is designed for:
- **Finance Students** learning ALM concepts
- **Banking Professionals** training on interest rate risk
- **Risk Managers** exploring scenario analysis
- **Educators** teaching quantitative finance

## 📚 Learning Outcomes

After using this platform, users will understand:
- Asset-liability sensitivity analysis
- Interest rate risk measurement and management
- Swap valuation and risk characteristics
- Duration and convexity concepts
- Monte Carlo simulation applications
- Regulatory stress testing frameworks

## 🛠️ Technical Architecture

### Built With
- **Frontend**: Streamlit with custom CSS styling
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Financial Calculations**: Custom algorithms + QuantLib (optional)
- **Export Capabilities**: ReportLab (PDF), XlsxWriter (Excel)

### Key Technical Features
- Professional Bloomberg-style UI design
- Real-time calculations and visualizations
- Export functionality for analysis results
- Modular architecture for easy extension
- Educational tooltips and explanations

## 📁 Project Structure

```
alm-educator/
├── irr_educate.py          # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── docs/                  # Additional documentation (optional)
```

## 🔧 Configuration

### Password Configuration
To change the access password, modify line in `irr_educate.py`:
```python
if st.session_state["password"] == "your_new_password":
```

### Market Data Sources
The tool uses simulated market data for educational purposes. For production use with real data, you can:
- Connect to FRED API for historical rates
- Integrate with Bloomberg/Refinitiv APIs
- Use other financial data providers

## 📖 Usage Examples

### Building a Portfolio
1. Navigate to "ALM Sensitivity Analysis"
2. Add assets (Fixed/Variable Rate Loans, Securities)
3. Add liabilities (Deposits, Borrowings)
4. View real-time sensitivity analysis and recommendations

### Pricing a Swap
1. Go to "Swap Analytics Tool"
2. Configure swap parameters (notional, tenor, rates)
3. View NPV, DV01, and cash flows
4. Export detailed analysis reports

### Running Scenarios
1. Access "Monte Carlo Simulations"
2. Set rate model parameters
3. Run simulations to see portfolio distributions
4. Analyze Value-at-Risk metrics

## 🤝 Contributing

We welcome contributions to improve the educational value of this tool:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This tool is designed for educational purposes only.** All market data, rates, and scenarios used in this application are either simulated or based on publicly available historical information. The calculations and analytics provided should not be used for actual trading, investment decisions, or risk management purposes. This platform is intended to help students and professionals learn about Asset-Liability Management concepts in a controlled educational environment.



## 
- Built with Streamlit for rapid development
- Inspired by Bloomberg Terminal design principles
- Educational content based on industry best practices
- Thanks to the open-source financial Python community

---
