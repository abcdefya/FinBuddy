# FinBuddy - Your Personal Financial AI Companion

![FinBuddy Logo](https://via.placeholder.com/150) <!-- Replace with your logo URL if you have one -->

Welcome to **FinBuddy**, a cutting-edge open-source AI platform built to transform how you interact with finance. Created as my graduation project, FinBuddy leverages advanced AI technologies to deliver powerful, user-friendly financial analysis and insights—making finance accessible to everyone, from students to seasoned investors.

Think of FinBuddy as your go-to financial sidekick, designed to simplify complex decisions, uncover market trends, and guide you with personalized advice—all through an innovative multi-layer AI system.

---

## 🌟 Features

- **Smart Financial Agents**: Tackle tough financial questions with a step-by-step reasoning approach for clear, actionable results.
- **Multi-Layer Architecture**:
  1. **AI Agents Layer**: Dedicated agents for stock analysis, budgeting, and report creation.
  2. **Core Algorithms Layer**: Optimized AI strategies for financial tasks.
  3. **Operations Layer**: Efficient data processing and model training.
  4. **Flexible AI Foundation**: Seamlessly integrates with various AI models for top performance.
- **Real-Time Data**: Connects to financial data sources like Yahoo Finance, Finnhub, and SEC filings.
- **Beginner-Friendly**: Includes tutorials and workflows for all experience levels.
- **Open-Source**: Free to explore, adapt, and enhance—finance belongs to everyone!

---

## 🚀 Why FinBuddy?

FinBuddy isn’t just a tool—it’s your partner in mastering finance. Built from scratch as my graduation project, it blends academic innovation with real-world utility to:
- Deliver clear, reliable financial insights.
- Streamline equity research and market analysis.
- Provide a foundation for future AI-driven financial solutions.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- Git
- API keys for financial data (e.g., Finnhub, Yahoo Finance) and AI services (e.g., OpenAI)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[YourUsername]/FinBuddy.git
   cd FinBuddy
   ```
2. **Set Up a Virtual Environment**:
   ```bash
   conda create -n finbuddy python=3.8
   conda activate finbuddy
   ```
3. **Install Dependencies**:
   ```bash
   pip install -U finbuddy
   ```
4. **Configure API Keys**:
   - Rename `OAI_CONFIG_LIST_sample` to `OAI_CONFIG_LIST` and add your AI service API keys.
   - Update `config_api_keys.py` with your financial data API keys.

### Quick Start
Run a sample task:
```bash
python -m finbuddy.agents.workflow --task "analyze AAPL stock"
```

Explore the `/tutorials` folder for more examples!

---

## 📂 Project Structure

```
FinBuddy/
├── finbuddy/
│   ├── agents/           # AI agent workflows and tools
│   ├── data_source/      # Utilities for financial data access
│   └── functional/       # Analysis, visualization, and utility scripts
├── tutorials/            # Guides for all skill levels
├── README.md             # You’re here!
└── requirements.txt      # Dependencies
```

---

## 📚 Tutorials

- **Beginner**: Predict stock trends with `tutorials/buddy_forecaster.ipynb`.
- **Advanced**: Create a detailed equity report with `tutorials/buddy_research_report.ipynb`.

---

## 🌍 Contributing

FinBuddy is my graduation project, but it’s also a collaborative effort! Want to join in?
1. Fork the repo.
2. Create a branch (`git checkout -b feature/cool-addition`).
3. Commit your changes (`git commit -m "Added cool addition"`).
4. Push to your branch (`git push origin feature/cool-addition`).
5. Open a Pull Request.

More details in `CONTRIBUTING.md` (coming soon!).

---

## 🎓 About This Project

FinBuddy is my graduation project—a labor of love at the intersection of AI and finance. I set out to build a tool that empowers users with smart, accessible financial insights, proving that advanced analysis doesn’t have to be out of reach. From concept to code, it’s a showcase of my skills and vision for the future of financial technology.

**Created by**: DO THE ANH
**Institution**: Hanoi uni of Sci and Tech 
**Date**: March 2025

---

## 📬 Contact

Questions, suggestions, or just want to chat? Reach me at anh.dothe47@gmail.com or open an issue on GitHub!

---

## 🙌 Acknowledgments

- Gratitude to my professors and peers for their guidance.
- Shoutout to the open-source community for inspiring innovation.

---

**Ready to make finance your buddy? Let’s dive in with FinBuddy!**

---
