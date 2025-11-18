# 🛡️ Fraud Detection with Apache Spark MLlib on AWS EMR

<div align="center">

![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white)
![AWS](https://img.shields.io/badge/AWS%20EMR-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLlib](https://img.shields.io/badge/MLlib-3.5.0-FF6F00?style=for-the-badge&logo=apache-spark&logoColor=white)

**A comprehensive fraud detection system leveraging distributed machine learning, adversarial robustness, and real-time streaming capabilities**

[📋 Project Overview](#-project-overview) • [👥 Team](#-team) • [🏗️ Architecture](#️-architecture) • [📚 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [📊 Features](#-features)

</div>

---

## 📋 Project Overview

This project implements a production-grade fraud detection system using **Apache Spark MLlib** on **AWS EMR**, designed to detect fraudulent credit card transactions with high accuracy and robustness. The system includes:

- 🔍 **Data Profiling & Quality Assurance**: Comprehensive data analysis and validation
- 🤖 **Machine Learning Pipelines**: Multiple ML models (Logistic Regression, Random Forest, GBT)
- 🎯 **Hyperparameter Tuning**: Automated optimization with cross-validation
- 🛡️ **Adversarial Robustness**: Defense mechanisms against adversarial attacks
- 📊 **Real-time Streaming**: Structured streaming for live fraud detection
- 📈 **Comprehensive Evaluation**: Advanced metrics and visualization

---

## 👥 Team

| Role                                          | Team Member            | Responsibilities                                                                        |
| --------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------- |
| 🏗️ **Technical Lead / ML Pipeline Architect** | **Rohit Lahori**       | EMR setup, MLlib pipelines, optimization, streaming integration                         |
| 📊 **Evaluation & Research Lead**             | **Ansh Kaushik**       | Evaluation metrics, hyperparameter tuning, adversarial robustness, streaming evaluation |
| ✅ **Quality Assurance & Documentation Lead** | **Hussain Bin Yousuf** | Data profiling, testing framework, adversarial defenses, technical report               |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kaggle Dataset                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS S3 (Raw Data)                            │
│              s3://bucket/raw-data/creditcard.csv                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Data Profiling         │  │   Feature Engineering    │
│   (Hussain)              │  │   (Rohit)                │
└────────────┬─────────────┘  └────────────┬─────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              ML Model Training (Rohit)                          │
│    • Logistic Regression • Random Forest • GBT Classifier      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            Model Evaluation & Tuning (Ansh)                     │
│    • Cross-Validation • Hyperparameter Tuning • Metrics         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         Adversarial Robustness Testing (Ansh + Hussain)         │
│    • FGSM Attacks • Defense Mechanisms • Robustness Metrics    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Streaming Integration (Rohit + Ansh)               │
│    • Structured Streaming • Real-time Predictions               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final Deliverables                           │
│    • Technical Report • Presentation • Code Repository         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 Documentation

### 📘 Master Integration Plan

**Start here!** The central orchestration document covering:

- Project architecture and integration points
- Git branching strategy and collaboration workflow
- S3 bucket structure and EMR Studio organization
- Cost management and budget tracking
- Emergency scenarios and rollback procedures
- Final integration and testing protocol

👉 **[Read MASTER_INTEGRATION_PLAN.md](./MASTER_INTEGRATION_PLAN.md)**

### 👨‍💻 Individual Execution Plans

#### 🏗️ Rohit Lahori - Technical Lead

Complete guide for EMR setup, MLlib pipelines, optimization, and streaming integration.

👉 **[Read ROHIT_LAHORI_PLAN.md](./ROHIT_LAHORI_PLAN.md)**

#### 📊 Ansh Kaushik - Evaluation Lead

Detailed instructions for evaluation metrics, hyperparameter tuning, adversarial robustness, and streaming evaluation.

👉 **[Read ANSH_KAUSHIK_PLAN.md](./ANSH_KAUSHIK_PLAN.md)**

#### ✅ Hussain Bin Yousuf - QA Lead

Comprehensive guide for data profiling, testing framework, adversarial defenses, and documentation.

👉 **[Read HUSSAIN_BIN_YOUSUF_PLAN.md](./HUSSAIN_BIN_YOUSUF_PLAN.md)**

---

## 🚀 Quick Start

### Prerequisites

- AWS Account with EMR access
- AWS CLI configured (`aws configure`)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)
- Python 3.9+ with pip
- Git

### Initial Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Rohit-Lahori02/FraudDetection.git
   cd FraudDetection
   ```

2. **Set up AWS credentials**

   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, region, and output format
   ```

3. **Configure Kaggle API**

   ```bash
   # Download kaggle.json from your Kaggle account settings
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Follow Phase 0: Dataset Preparation**

   - See [MASTER_INTEGRATION_PLAN.md](./MASTER_INTEGRATION_PLAN.md#phase-0-dataset-preparation)

5. **Set up EMR Cluster**
   - Follow instructions in [ROHIT_LAHORI_PLAN.md](./ROHIT_LAHORI_PLAN.md#phase-1-emr-setup-and-configuration)

---

## 📊 Features

### 🔍 Data Profiling

- Comprehensive statistical analysis
- Missing value detection
- Data quality validation
- Automated HTML report generation

### 🤖 Machine Learning Models

- **Logistic Regression**: Fast, interpretable baseline model
- **Random Forest**: Robust ensemble method
- **Gradient Boosted Trees**: High-performance boosting algorithm

### 🎯 Model Optimization

- Stratified cross-validation for imbalanced data
- Hyperparameter tuning with MLflow tracking
- Model comparison and statistical significance testing

### 🛡️ Adversarial Robustness

- FGSM (Fast Gradient Sign Method) attack implementation
- Feature clamping defense
- Ensemble voting defense
- Robustness metrics and evaluation

### 📡 Real-time Streaming

- Structured streaming with Spark
- Real-time fraud detection
- Batch evaluation and metrics aggregation

### 📈 Evaluation Metrics

- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Precision, Recall, F1-Score
- Confusion Matrix
- Specificity and Balanced Accuracy

---

## 📁 Project Structure

```
FraudDetection/
├── README.md                          # This file
├── MASTER_INTEGRATION_PLAN.md         # Central orchestration document
├── ANSH_KAUSHIK_PLAN.md               # Ansh's execution plan
├── ROHIT_LAHORI_PLAN.md               # Rohit's execution plan
├── HUSSAIN_BIN_YOUSUF_PLAN.md         # Hussain's execution plan
├── src/
│   ├── common/                        # Shared utilities
│   │   ├── spark_session.py
│   │   ├── s3_utils.py
│   │   ├── config.py
│   │   └── schema_validator.py
│   ├── rohit/                         # Rohit's code
│   │   ├── infrastructure/
│   │   ├── pipelines/
│   │   ├── optimization/
│   │   ├── streaming/
│   │   └── deployment/
│   ├── ansh/                          # Ansh's code
│   │   ├── evaluation/
│   │   ├── adversarial/
│   │   ├── streaming/
│   │   └── visualization/
│   └── hussain/                       # Hussain's code
│       ├── profiling/
│       ├── testing/
│       ├── defenses/
│       └── documentation/
├── tests/                             # Test suite
├── notebooks/                         # EMR Studio notebooks
│   ├── rohit/
│   ├── ansh/
│   └── hussain/
├── scripts/                           # Utility scripts
└── docs/                              # Additional documentation
```

---

## 🎯 Key Deliverables

- ✅ **Technical Report**: Comprehensive analysis of methodology, results, and findings
- ✅ **Presentation**: 15-minute presentation with slides and demo
- ✅ **Code Repository**: Well-documented, tested, and production-ready code
- ✅ **Working Demo**: End-to-end demonstration of the fraud detection system

---

## 💰 Cost Management

This project includes comprehensive cost tracking and optimization:

- **Estimated AWS Costs**: ~$50-100 for development and testing
- **Cost Alerts**: Automated budget alerts via AWS Budgets
- **Optimization Checklist**: Strategies to minimize costs
- **Plan B**: Local Spark setup for cost-sensitive scenarios

See [MASTER_INTEGRATION_PLAN.md](./MASTER_INTEGRATION_PLAN.md#cost-management) for details.

---

## 🔧 Technologies & Tools

| Category                | Technology                                            |
| ----------------------- | ----------------------------------------------------- |
| **Big Data**            | Apache Spark 3.5.0, Spark MLlib                       |
| **Cloud Platform**      | AWS EMR, AWS S3, AWS IAM, AWS CloudWatch              |
| **ML Framework**        | Spark MLlib (Logistic Regression, Random Forest, GBT) |
| **Experiment Tracking** | MLflow                                                |
| **Testing**             | pytest, pytest-spark, moto                            |
| **Data Profiling**      | ydata-profiling                                       |
| **Visualization**       | matplotlib, seaborn                                   |
| **CI/CD**               | GitHub Actions                                        |
| **Version Control**     | Git, GitHub                                           |

---

## 📞 Support & Escalation

For issues or questions:

1. Check the relevant execution plan document
2. Review the [MASTER_INTEGRATION_PLAN.md](./MASTER_INTEGRATION_PLAN.md#emergency-scenarios--rollback-procedures) for emergency procedures
3. Consult team members based on their expertise areas
4. Escalate to AWS Support if needed (see emergency contacts in master plan)

---

## 📝 License

This project is part of a Master's program coursework. All rights reserved.

---

## 🙏 Acknowledgments

- **Dataset**: Credit Card Fraud Detection from Kaggle
- **Platform**: AWS EMR for distributed computing
- **Framework**: Apache Spark MLlib for machine learning

---

<div align="center">

**Built with ❤️ by Ansh Kaushik, Rohit Lahori, and Hussain Bin Yousuf**

⭐ Star this repo if you find it helpful!

</div>
