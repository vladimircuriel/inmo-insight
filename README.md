<h1 align="center">
   InmoInsight - Housing Market Data-Driven Analysis
</h1>

<p align="center"> 
  <img src="https://github.com/user-attachments/assets/inmoinsight-logo" alt="InmoInsight" width="400"/> 
</p>

<div align="center">  
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />  
  <img src="https://img.shields.io/badge/Apache_Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white" />  
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" />  
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" />  
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />  
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />  
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />  
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />  
  <img src="https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white" />  
</div>

---

**InmoInsight** is an end-to-end data engineering and machine learning project for analyzing the rental housing market in the Dominican Republic. It combines web scraping, ETL pipelines orchestrated with Apache Airflow, AI-powered data enrichment via OpenAI Batch API, a predictive XGBoost model, and interactive Tableau dashboards—all designed to extract actionable insights from real estate listings.

## Table of Contents

- [Features](#features)
- [Application](#application)
- [Tools Used](#tools-used)
- [Installation](#installation)
- [Areas for Improvement](#areas-for-improvement)

---

## Features

- **Automated Web Scraping**: Extracts apartment listings from supercasas.com using BeautifulSoup and requests, parsing prices, rooms, bathrooms, locations, amenities, and free-text descriptions.

- **ETL Pipeline with Airflow**: A scheduled DAG runs every two weeks, orchestrating Extract → Transform → Enrich → Load stages with built-in retries, logging, and data validation.

- **AI-Powered Enrichment**: Leverages OpenAI Batch API to extract unstructured features from listing descriptions (service room, walk-in closet, conflicts in data), normalize locations, and validate cross-referenced fields.

- **Rent Price Prediction**: XGBoost regressor trained on 32 engineered features achieves R² = 0.757 with MAE of ~11,773 DOP. Model artifacts (weights, encoders, metadata) are persisted for inference.

- **Interactive Dashboard**: Tableau workbook visualizes price distributions, amenity correlations, location heatmaps, and filtering by sector, rooms, and price range.

- **Tech Stack & Infrastructure**  
  - **Orchestration:** Apache Airflow (TaskFlow API), Docker Compose  
  - **Extraction:** BeautifulSoup4, Requests  
  - **Processing:** Pandas, NumPy, Scikit-Learn  
  - **Enrichment:** OpenAI API (Batch)  
  - **Storage:** PostgreSQL, SQLAlchemy  
  - **Modeling:** XGBoost, Joblib  
  - **Visualization:** Matplotlib, Tableau Desktop  

---

## Application

A showcase is available in [my portfolio](https://vladimircuriel.com/posts/20_inmo-insight/)!

---

## Tools Used

- ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat-square) **Python 3.13**: Core language for scraping, data processing, and machine learning pipelines.

- ![Apache Airflow](https://img.shields.io/badge/Apache_Airflow-017CEE?logo=apache-airflow&logoColor=white&style=flat-square) **Apache Airflow**: Workflow orchestration for scheduling and monitoring ETL pipelines with TaskFlow API.

- ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white&style=flat-square) **PostgreSQL**: Relational database storing enriched apartment listings for querying and analysis.

- ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?logo=xgboost&logoColor=white&style=flat-square) **XGBoost**: Gradient boosting regressor for predicting rental prices with high accuracy.

- ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square) **Scikit-Learn**: Preprocessing, feature encoding, train-test splitting, and evaluation metrics.

- ![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white&style=flat-square) **OpenAI API**: Batch processing for extracting unstructured features and validating data via GPT.

- ![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-3776AB?logo=python&logoColor=white&style=flat-square) **BeautifulSoup4**: HTML parsing library for web scraping apartment listings.

- ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=flat-square) **Pandas**: Data manipulation and cleaning throughout the pipeline.

- ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white&style=flat-square) **Docker**: Containerization of Airflow and PostgreSQL for reproducible environments.

- ![Tableau](https://img.shields.io/badge/Tableau-E97627?logo=tableau&logoColor=white&style=flat-square) **Tableau Desktop**: Interactive dashboards for exploratory analysis and stakeholder presentations.

- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white&style=flat-square) **Matplotlib**: Model evaluation plots (residuals, feature importance, learning curves).

- ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?logo=sqlalchemy&logoColor=white&style=flat-square) **SQLAlchemy**: ORM for database connections and data loading.

---

## Installation

### Prerequisites

- **Docker**
- **Python 3.13+**

### Steps

1. **Clone the repository**:

```bash
git clone https://github.com/vladimircuriel/inmo-insight
```

2. **Navigate to the project directory**:

```bash
cd inmo-insight
```

3. **Start Airflow services**:

```bash
cd airflow
docker compose up -d
```

4. **Access Airflow UI**:

Open your browser and visit `http://localhost:8080` to access the Airflow dashboard. Default credentials are `airflow` / `airflow`.

5. **Train the model** (optional):

```bash
cd ..
uv run python -m model.train_rent_predictor --input data/supercasas_apartments_v2.csv
```

6. **ENV Variables**:

Create a `.env` file in the `airflow/` directory:

```dotenv
OPENAI_API_KEY=sk-your-openai-api-key    # API key for OpenAI Batch enrichment
POSTGRES_HOST=localhost                   # PostgreSQL host
POSTGRES_PORT=5432                        # PostgreSQL port
POSTGRES_DB=inmo_insight                  # Database name
POSTGRES_USER=postgres                    # Database user
POSTGRES_PASSWORD=your-password           # Database password
```

---

## Areas for Improvement

- The scraper only extracts data from supercasas.com; additional sources would improve market coverage.  
- The model does not incorporate geolocation features (distance to metro, schools, hospitals) which could improve predictions.  
- OpenAI enrichment costs can accumulate; caching or local LLM alternatives are not yet implemented.  
- No real-time prediction API is exposed; the model is trained offline and requires manual inference.  
- There is no automated model retraining when new data is loaded; drift detection is not implemented.  
- Data versioning (e.g., DVC) is not set up, making it harder to track changes across pipeline runs.  
- Test coverage for ETL stages and model training is minimal; Great Expectations or similar validation frameworks could be added.  
- The project lacks a web interface for end users to query predictions or explore data interactively.