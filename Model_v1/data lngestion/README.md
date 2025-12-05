# 📈 End-to-End Stock Market AI Prediction System

An automated Machine Learning pipeline that collects real-time stock data, trains prediction models, and serves forecasts via an interactive dashboard. This project demonstrates full-stack data science capabilities, moving beyond static notebooks to a production-ready system.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Scikit-learn](https://img.shields.io/badge/ML-Scikit_learn-orange) ![Plotly](https://img.shields.io/badge/Visualization-Plotly-green)

## 🏗 System Architecture

The system is designed as an automated pipeline consisting of three main layers: **Data Ingestion**, **Model Processing**, and **Serving Layer**.

```mermaid
graph TD
    subgraph Data_Layer [Zone 1: Data Ingestion]
        API[Yahoo Finance API] -->|Fetch Data| Script_Get[Data Collector Script]
        Script_Get -->|Save CSV| Storage[(Raw Data Storage)]
    end

    subgraph ML_Layer [Zone 2: Machine Learning Engine]
        Storage -->|Load Data| Trainer[Model Trainer]
        Trainer -->|Train & Serialize| Model_File{{Saved Model .pkl}}
    end

    subgraph Serving_Layer [Zone 3: User Interface]
        Model_File -->|Load Model| Dashboard[Streamlit Dashboard]
        Storage -->|Load History| Dashboard
        User((User)) -->|Interactive View| Dashboard
    end

sdfsdfsd
