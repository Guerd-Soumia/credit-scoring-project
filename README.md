# Credit Scoring Multi-Agent System

**Project 13 — Octroi Décentralisé de Crédit & Analyse de Solvabilité**

An AI-powered multi-agent system that evaluates creditworthiness by combining transaction fraud detection, borrower risk assessment, and a final rule-based decision model with LLM justification.

## Team

- **Soumia GUERD**
- **Soumeya TCHOULAK**
- **Meriem BACHESAIS**

## Project Overview

The objective of this project is to develop a multi-agent AI system capable of analyzing financial transactions and borrower profiles to estimate the risk of loan default.

The system combines three agents coordinated through a unified decision pipeline.

## System Architecture

### Agent 1 — Transaction Fraud Detection (RNN / LSTM)
- Analyzes credit card transaction features.
- Uses a Recurrent Neural Network (LSTM).
- Detects potentially fraudulent transactions.
- Produces a fraud probability score.

### Agent 2 — Loan Default Prediction (Random Forest)
- Analyzes borrower financial and demographic information.
- Uses a Random Forest classifier.
- Estimates the probability of loan default.
- Applies a 50% risk threshold for its risk classification.

### Agent 3 — Final Decision (Rules + Local LLM)
- Reads the scores from Agent 1 and Agent 2.
- Applies deterministic rules: if fraud > 50%, reject; if default > 50%, reject; otherwise approve.
- Uses a local LLM (Llama 3.2 1B via Ollama) to generate a natural-language justification sentence.
- If the LLM is offline, falls back to a safe template.

## User Interface

The project includes an interactive web application developed with **Streamlit**.

The interface provides three main sections:

**1. Transaction Fraud Analysis**
- Load real normal transactions.
- Load real fraudulent transactions.
- Generate random transactions.
- Edit the 30 input features.
- Run the LSTM fraud prediction.

**2. Loan Solvency Assessment**
- Load a safe applicant profile.
- Load a risky applicant profile.
- Generate a random applicant profile.
- Edit the 79 borrower features.
- Run the Random Forest prediction.

**3. Final Decision (Agent 3)**
- Combines the results of Tabs 1 and 2.
- Applies the strict decision rules.
- Generates the final approval/rejection and justification.

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Streamlit
- Ollama (Local LLM)
- Jupyter Notebook

## Installation and Execution

### 1. Clone the repository
```bash
git clone https://github.com/Guerd-Soumia/credit-scoring-project.git
cd credit-scoring-project