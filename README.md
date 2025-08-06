# Flight-Delay-Prediction-Modeling
Flight delay prediction using Random Forest on synthetic and real data, exploring accuracy, key features, and class imbalance.
Background
When my family was stranded in London during a transit system meltdown on a trip in 2024, it wasn’t just the chaos that struck me—it was how disconnected and brittle the decision-making seemed. That experience pushed me to explore predictive modeling through a Kaggle project on flight delays.
Flight delays affect millions of travelers every year and pose significant challenges to airline operations. Predicting delays can help optimize scheduling, improve customer satisfaction, and reduce operational costs. This project explores two models — one built on synthetic data and the other on real-world data — to predict whether a flight will be delayed.
The synthetic dataset consisted of 50,000 generated samples based on common airline attributes (e.g., origin, destination, departure time, gate wait time, and previous flight delays). The real-world dataset, sourced from Kaggle.com, included over 1.6 million records with actual flight statistics and delay statuses.

Project Narrative
AI Model 1 – Synthetic Data 
The first part of the project involved creating and modeling synthetic data to simulate realistic flight scenarios:
Step 1–2: Generated 50,000 samples using Python (pandas, numpy, etc.), simulating key flight features such as departure time, number of previous late flights, and gate wait time.
Step 3: Addressed preprocessing issues by converting object-type columns into integers and applying one-hot encoding to categorical variables.
Step 4: Split the data into training (80%) and test (20%) sets and aligned features to avoid shape mismatches.
Step 5: Tackled class imbalance using SMOTETomek to balance the delayed vs. non-delayed samples.
Step 6–7: Trained a RandomForestClassifier model which achieved a striking 99.98% accuracy with only two misclassified test samples.
Step 8: Saved the model using joblib.

Top features:
Number of previous flights late (73%)
Average gate wait time (20%)

⚠️ This exceptionally high accuracy is likely due to the model learning from a simplified, synthetic dataset where feature significance was strongly weighted. It may not generalize well to real-world data.

AI Model 2 – Real-World Data 
The second part of the project used a real-world dataset containing over 1.6 million flight records:
Step 1–2: Imported and cleaned the dataset, filling missing values (mode for categorical, median for numerical).
Step 3: Encoded categorical variables and created a binary target label (Delayed: 1 or 0).
Step 4: Split the dataset into training (70%) and testing (30%) sets.
Step 5: Dealt with class imbalance by separating majority (not delayed) and minority (delayed) classes and applying resampling techniques.
Step 6–7: Trained the RandomForestClassifier, achieving 92% accuracy.

F1 score for “Not Delayed” class: 0.9602
F1 score for “Delayed” class: 0.0295
Step 8: Saved the model using joblib.

Top features:
Scheduled Arrival Time (36%)
Scheduled Departure Time (34%)
Actual Departure Time (7%)
⚠️ Although the accuracy was high, the low F1 score for the delayed class highlights the challenge of dealing with imbalanced real-world data.
Links:

 https://www.kaggle.com/writeups/gordonandric/flight-delay-prediction-modeling
 https://www.kaggle.com/code/gordonandric/airline-synthetic
 https://www.kaggle.com/code/gordonandric/airline-real
