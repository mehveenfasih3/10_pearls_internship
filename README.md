# AQI_prediction_system

**Submitted to:** 10Pearls  
**Submitted by:** Mehveen Fasih  
**Date:** November 8, 2025  

---

## **1. Introduction**
This project focuses on developing an **automated air quality prediction system** that fetches, processes, analyzes, and models real-time environmental data.  
The system predicts **Air Quality Index (AQI)** values using weather and pollution data sourced from the **OpenWeather API**.

The workflow integrates **GitHub Actions** for automation and **Hopsworks Feature Store** for centralized data management and model storage.  
This end-to-end MLOps pipeline operates on a scheduled basis, ensuring continuous data updates and model retraining with minimal manual effort.

---

## **2. Project Objectives**
- Automate real-time collection of environmental and weather data.  
- Perform **exploratory data analysis (EDA)** and **feature engineering**.  
- Develop predictive machine learning models for **AQI forecasting**.  
- Maintain and register models and datasets in **Hopsworks Feature Store**.  
- Enable continuous retraining and improvement of model accuracy.  

---

## **3. Workflow Overview**
The project consists of three **automated workflows** implemented through **GitHub Actions**, each scheduled to run at fixed intervals.

### **a. Fetch Real-Time Data**
- Runs every **3 hours**.  
- Collects air quality and weather data via the **OpenWeather API**.  
- Saves data to `data/karachi_2_years.csv`.  
- Automatically commits and pushes updates to the repository.

### **b. Process EDA and Upload to Hopsworks**
- Runs every **3 hours and 15 minutes**.  
- Cleans and merges datasets for modeling.  
- Performs **feature engineering** and **AQI computation**.  
- Uploads validated data to **Hopsworks Feature Store** as `air_quality_features`.  

### **c. Model Training for AQI Prediction**
- Runs every **6 hours and 45 minutes**.  
- Retrieves the latest data from **Hopsworks**.  
- Trains multiple regression models: **Random Forest**, **Gradient Boosting**, and **XGBoost**.  
- Evaluates models using **RMSE, MAE, R², and MAPE** metrics.  
- Registers models and metadata in the **Hopsworks Model Registry**.

---

## **4. Key Technologies Used**

| **Component**          | **Technology**                  |
|------------------------|----------------------------------|
| Programming Language   | Python 3.10                      |
| Data Storage           | Hopsworks Feature Store          |
| Data Collection        | OpenWeather API                  |
| Automation             | GitHub Actions (CI/CD)           |

---

## **5. Core Functions and Features**
- **Data Cleaning & Validation:** Ensures consistent and complete datasets free of missing or invalid values.  
- **Feature Engineering:** Derives meaningful predictors from pollution and weather attributes.  
- **Model Comparison:** Automatically selects the best model based on **R² scores**.  
- **Continuous Integration:** All workflows are triggered and managed automatically.  
- **Centralized Model Management:** Stores datasets, feature versions, and trained models in **Hopsworks**.  

---

## **6. Model Performance Analysis**
The performance comparison evaluates three machine learning algorithms — **Gradient Boosting**, **Random Forest**, and **XGBoost** — for AQI prediction.  
Model accuracy was measured using the **R² Score**, indicating how effectively each model explains variance in AQI data.

- **Gradient Boosting (R² = 0.9211)** — Highest performance, demonstrating superior accuracy and stability.  
- **XGBoost (R² = 0.8253)** — Strong and reliable, but slightly less sensitive to subtle variations.  
- **Random Forest (R² = 0.7511)** — Solid baseline model, though less precise for AQI fluctuations.

**Conclusion:**  
Gradient Boosting outperformed other models, making it the optimal choice for **real-time AQI forecasting**.

![Model Performance Comparison](6c39632f-8aa9-4320-a810-b7f0261ae858.png)

---

## **7. Project Outcomes**
- Fully automated end-to-end pipeline: data fetching → cleaning → training → storage.  
- Continuous model updates with every scheduled workflow.  
- Centralized and versioned **Feature Store** for scalable future enhancements.  
- Modular, reusable codebase suitable for **multi-city AQI prediction**.  

---

## **8. Conclusion**
This project showcases a **complete MLOps pipeline** for air quality monitoring and prediction.  
By combining **GitHub Actions automation** with **Hopsworks Feature Store**, the system ensures real-time data flow, automated retraining, and production-grade scalability.

It demonstrates practical use of **cloud integration**, **CI/CD workflows**, and **machine learning best practices** for environmental analytics and predictive maintenance applications.

  
**Mehveen Fasih**  
*Data Science Intern – 10Pearls*  

