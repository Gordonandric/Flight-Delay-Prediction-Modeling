# ✈️ Flight Delay Prediction Modeling
Flight delay prediction using Random Forest on synthetic and real data, exploring accuracy, key features, and class imbalance.
## Background

When my family was stranded in London during a transit system meltdown on a trip in 2024, it wasn’t just the chaos that struck me—it was how disconnected and brittle the decision-making seemed. That experience pushed me to explore predictive modeling through a Kaggle project on flight delays.

Flight delays affect millions of travelers every year and pose significant challenges to airline operations. Predicting delays can help optimize scheduling, improve customer satisfaction, and reduce operational costs. This project explores two models — one built on synthetic data and the other on real-world data — to predict whether a flight will be delayed.

- **Synthetic Dataset**: 50,000 generated samples based on airline attributes (e.g., origin, destination, departure time, gate wait time, and previous flight delays)
- **Real-World Dataset**: 1.6M+ flight records sourced from [Kaggle.com](https://www.kaggle.com)

---

## Project Narrative

### 🧠 AI Model 1 – Synthetic Data

The first part of the project involved creating and modeling synthetic data to simulate realistic flight scenarios:

**Steps:**
1. Generated 50,000 samples using Python (`pandas`, `numpy`, etc.), simulating features like departure time, previous late flights, and gate wait time.
2. Converted object-type columns to integers; applied one-hot encoding to categorical features.
3. Split the data into 80% training and 20% testing sets; ensured aligned feature dimensions.
4. Handled class imbalance using `SMOTETomek` to balance delayed vs. non-delayed flights.
5. Trained a `RandomForestClassifier` achieving **99.98% accuracy** with only 2 misclassified samples.
6. Saved the model using `joblib`.

**Top Features:**
- `Number of previous flights late` — 73%
- `Average gate wait time` — 20%

> ⚠️ Note: The high accuracy is likely due to the simplified nature of the synthetic data, which may not reflect real-world complexity.

---

### 🌍 AI Model 2 – Real-World Data

The second part of the project used a real dataset with over 1.6 million flight records.

**Steps:**
1. Imported and cleaned the dataset. Filled missing values (mode for categorical, median for numerical).
2. Encoded categorical variables; created a binary target label (`Delayed`: 1 or 0).
3. Split the data into 70% training and 30% testing sets.
4. Addressed class imbalance by resampling minority (delayed) cases.
5. Trained a `RandomForestClassifier` achieving **92% accuracy**.

**Performance:**
- `F1 score (Not Delayed)` — 0.9602  
- `F1 score (Delayed)` — 0.0295

**Top Features:**
- `Scheduled Arrival Time` — 36%
- `Scheduled Departure Time` — 34%
- `Actual Departure Time` — 7%

> ⚠️ The low F1 score for delayed flights shows the challenge of handling real-world imbalance, even with good overall accuracy.

---

## 🔗 Links

- 📄 [Writeup on Kaggle](https://www.kaggle.com/writeups/gordonandric/flight-delay-prediction-modeling)  
- 🧪 [Synthetic Data Notebook](https://www.kaggle.com/code/gordonandric/airline-synthetic)  
- 📊 [Real-World Data Notebook](https://www.kaggle.com/code/gordonandric/airline-real)

