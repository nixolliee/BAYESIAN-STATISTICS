# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:31:46 2026

@author: ACER
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

train_texts = [
    # Health & Lifestyle News
    "The Department of Health reported today that dengue cases in Metro Manila have increased by 30% compared to last year.",
    "Hospitals across Metro Manila report rising respiratory infections, urging residents to wear masks and get flu vaccinations.",
    "Nutritionists warn that teenagers consuming too much fast food are at higher risk for obesity and diabetes.",

    # Hard News
    "Local authorities confirmed that a magnitude 5.6 earthquake struck the northern region early this morning.",
    "A cargo ship collided with a pier in Manila Bay this morning, causing minor injuries but no fatalities.",
    "Authorities reported flooding in Pampanga after heavy overnight rains, prompting evacuations in low-lying areas.",

    # Sports News
    "Filipino basketball sensation Kiefer Ravena led his team to victory in the PBA finals last night.",
    "The Philippine national football team defeated Vietnam 2-1 in the SEA Games, scoring two late goals.",
    "Manny Pacquiao’s former protege won a regional boxing championship last night, earning praise from fans.",

    # Entertainment & Celebrity News
    "The upcoming Metro Manila Film Festival will feature a special tribute to classic Filipino cinema.",
    "Sarah Geronimo announced her upcoming world tour, including Manila, Tokyo, and Los Angeles stops.",
    "A popular actor won the Best Actor award at the Metro Manila Film Festival for his latest drama film.",

    # Science & Technology News
    "Tech startup Innovatek announced the launch of a new AI-powered translation app that supports multiple Philippine languages.",
    "UP Diliman researchers developed a low-cost water filter capable of removing heavy metals in rural areas.",
    "A Filipino engineer invented a solar-powered drone to assist in disaster relief operations.",

    # Political News
    "The Senate passed a bill that increases the minimum wage for workers in the private sector.",
    "The House approved a bill to provide free internet access in public schools nationwide.",
    "Local government units passed a resolution banning single-use plastics in public markets.",

    # Soft News
    "Tourists flocked to Palawan over the long weekend, drawn by its pristine beaches and scenic lagoons.",
    "A weekend food festival in Cebu City featured local delicacies and street performances.",
    "A small art gallery in Makati is gaining popularity for showcasing emerging Filipino artists.",

    # Environmental News
    "A study revealed that air pollution in urban areas has worsened due to increased vehicle emissions.",
    "Deforestation in Mindoro is threatening native wildlife, prompting calls for stricter logging regulations.",
    "Volunteers planted over 5,000 trees along the Pasig River to combat urban pollution.",

    # Business & Financial News
    "The stock market experienced a slight decline today, with the Philippine Stock Exchange Index dropping by 0.8%.",
    "The Philippine peso strengthened slightly against the US dollar amid investor optimism.",
    "A local startup launched an e-commerce platform connecting small farmers to urban consumers.",

    # Opinion & Editorial
    "The current traffic situation in Metro Manila highlights the urgent need for efficient public transport solutions.",
    "Experts argue that promoting cycling and walking in cities can reduce traffic congestion and improve health.",
    "Social media spreads news quickly but requires critical thinking to avoid misinformation."
]

train_labels = [
    "Health & Lifestyle News", "Health & Lifestyle News", "Health & Lifestyle News",
    "Hard News", "Hard News", "Hard News",
    "Sports News", "Sports News", "Sports News",
    "Entertainment & Celebrity News", "Entertainment & Celebrity News", "Entertainment & Celebrity News",
    "Science & Technology News", "Science & Technology News", "Science & Technology News",
    "Political News", "Political News", "Political News",
    "Soft News", "Soft News", "Soft News",
    "Environmental News", "Environmental News", "Environmental News",
    "Business & Financial News", "Business & Financial News", "Business & Financial News",
    "Opinion & Editorial", "Opinion & Editorial", "Opinion & Editorial"
]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_texts, train_labels)

while True:
    user_input = input("\nEnter news text to classify. Enter 'exit' to end: ").strip()
    if user_input.lower() == "exit":
        print("Exiting News Classifier. Goodbye!")
        break
    if user_input:
        prediction = model.predict([user_input])[0]
        print(f"Predicted Category: {prediction}")
    else:
        print("No input provided. Please try again.")