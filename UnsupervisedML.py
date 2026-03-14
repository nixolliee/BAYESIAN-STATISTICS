# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:20:12 2026

@author: ACER
"""
import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_excel("C:Users/ACER/OneDrive/Documents/MATHEMATICAL STATISTICS\BAYESIAN\Machine Learning.xlsx")

data_model = data.drop("Applicant ID", axis=1)


encoder = LabelEncoder()
for col in data_model.columns:
    data_model[col] = encoder.fit_transform(data_model[col])


kmeans = KMeans(n_clusters=2, random_state=42)
data["Cluster"] = kmeans.fit_predict(data_model)

data["Loan Decision"] = data["Cluster"].map({
    1: "Eligible for Loan",
    0: "Not Eligible for Loan"
})

print(data)