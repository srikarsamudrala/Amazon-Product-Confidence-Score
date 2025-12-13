# Amazon Product Confidence Score 🚀

**Team M15:** Srikar Samudrala, Diksha Bagade, Nirman Taterh, Tenzin Tayang  
**Date:** December 2025

## 📌 Project Overview
Fake and manipulated reviews have become a major challenge for Amazon’s marketplace, directly influencing search rankings, product visibility, and customer confidence. When reviews are artificially inflated, customers struggle to judge true quality, often leading to unmet expectations.

The **Amazon Product Confidence Score** is a machine learning-driven system designed to reflect how trustworthy a product's reviews are, not just how positive they seem. By analyzing reviewer sentiment and filtering spam, this system aims to maintain marketplace integrity and provide reliable information for customers and internal teams.

## 🧠 Methodology

### 1. Data Understanding & Preparation
* **Dataset:** Utilized the Amazon Product Review Spam Dataset from Kaggle, containing 26.7 million reviews from 15.4 million unique reviewers.
* **Sampling:** Implemented stratified product sampling of 50,000 unique ASINs per category (Cellphones, Home & Kitchen, Sports, and Toys) to preserve "review burst" patterns critical for spam detection.
* **Class Imbalance:** Maintained the natural ~80/20 spam-to-non-spam split to ensure realistic modeling.
* **Preprocessing:** Cleaning involved noise removal (dropping internal IDs), text normalization (lowercasing, artifact removal), and tokenization.

### 2. Spam Detection Models
We evaluated three models to handle the extreme class imbalance and detect fraudulent reviews:
* **Logistic Regression (Baseline):** Established a lower bound on performance using handcrafted features.
* **Random Forest:** Served as a production-grade interpretable model, achieving **84.57% accuracy** and robust performance against noise.
* **Bi-LSTM RNN with Attention (Best Performer):** A deep learning model designed to capture sequential patterns and linguistic nuances.
    * **Accuracy:** 96.13%
    * **Spam Recall:** 97.65% (Prioritized to minimize risk)
    * **ROC-AUC:** 98.17%

### 3. Sentiment Analysis
Spam reviews distort sentiment signals, so we analyzed sentiment only after filtering spam.
* **VADER (Baseline):** A lexicon-based model that achieved a Mean Absolute Error (MAE) of ~1.47 stars due to a lack of contextual depth.
* **DeBERTa (Fine-Tuned):** A transformer-based model that significantly outperformed VADER. After fine-tuning on the Amazon dataset, MAE dropped to **0.46**, capturing over 80% of the rating variance ($R^2 > 0.81$).

## 📊 The Confidence Score
The **Amazon Product Confidence Score** replaces the standard star rating with a weighted metric derived from genuine feedback. It combines the sentiment of written reviews with the average rating of non-spam reviews.

**Formula:**

$$\text{Confidence Score} = (0.35 \times \text{Spam Rating}) + (0.65 \times \text{Sentiment Rating})$$

* **Weighting:** 65% Sentiment / 35% Non-Spam Rating.
* **Rationale:** Written reviews often capture the customer experience more accurately than simple star ratings.

## 📉 Business Impact
* **Financial Savings:** By reducing "buyer's remorse," the system mitigates return costs, estimated at **$3 to $10 per item**.
* **Fraud Detection:** During testing, the system successfully caught **97.7%** of fraudulent reviews.
* **Marketplace Fairness:** Levels the playing field for honest merchants disadvantaged by competitors using artificial inflation.
* **Internal Empowerment:** Acts as an early-warning signal for Product Trust and Compliance teams to intervene rapidly.

## 🧪 A/B Testing & Future Work
* **Deployment Strategy:** We proposed an A/B testing framework to validate weighting hypotheses (e.g., 50/50 vs. 65/35 vs. 80/20) and optimize for conversion and engagement metrics.
* **Future Scope:** Work will focus on integrating with internal pipelines, developing merchant risk profiles, and continuous retraining to handle model drift.

## Dataset Link
https://drive.google.com/file/d/1mMK9qTjzMFLeFZ32KWw07i8_V_7U9DVY/view?usp=sharing
