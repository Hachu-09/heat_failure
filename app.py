import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure the Google Generative AI API
genai.configure(api_key="AIzaSyBg18tLTIeKEWL9mH2k-XH20apKiKoJWSk")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the heart failure dataset to get feature names and scaler
heart_data = pd.read_csv("HEART_MODEL/Heart Failure/heart_failure.csv")
x = heart_data.drop(columns="HeartDisease", axis=1)

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(x)

# Load the trained heart failure prediction model
with open('HEART_MODEL/Heart Failure/heart_failure_prediction.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

print("Heart failure model loaded successfully.")

# Initialize Flask app
app = Flask(__name__)

# Function to predict heart failure risk
def predict_heart_failure(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    input_data_df = pd.DataFrame(input_data_as_numpy_array, columns=x.columns)
    input_data_scaled = scaler.transform(input_data_df)

    prediction = ensemble_model.predict(input_data_scaled)[0]
    probability = ensemble_model.predict_proba(input_data_scaled)[0][1]
    risk_percentage = probability * 100

    if prediction == 0:
        risk = "The person is not at risk of heart failure"
        disease_type = "No Heart Failure"
    else:
        risk = "The person is at risk of heart failure"
        disease_type = "Heart Failure"

    return risk, risk_percentage, disease_type

# Function to generate a prevention report using Google Generative AI
def generate_prevention_report(risk, risk_percentage, disease_type, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        -Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        -Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        -Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        -In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        -Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        -Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        -Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        -Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        -Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        -Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        -Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    **Details:**
    Risk: {risk}
    Disease: {disease_type}
    Age: {age}
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

# Endpoint to handle predictions and return the risk and report
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the POST request (JSON)
        data = request.json
        
        # Extract features from JSON
        input_data = [
            data['age'], data['sex'], data['cp'], data['trtbps'], data['chol'], 
            data['fbs'], data['restecg'], data['thalachh'], data['exng'], 
            data['oldpeak'], data['slp']
        ]
        
        # Predict the disease
        risk, risk_percentage, disease_type = predict_heart_failure(input_data)
        
        # Generate a report if heart failure is predicted
        if disease_type == "Heart Failure":
            report = generate_prevention_report(risk, risk_percentage, disease_type, data['age'])
            return jsonify({
                'risk': risk,
                'risk_percentage': f"{risk_percentage:.2f}%",
                'disease_type': disease_type,
                'report': report
            })
        else:
            return jsonify({
                'risk': risk,
                'risk_percentage': "0%",
                'disease_type': disease_type
            })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
