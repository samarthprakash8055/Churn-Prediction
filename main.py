"""
Bank Churn Prediction - Gradio Deployment
Production-ready inference with LangChain + Groq integration
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import joblib
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pathlib import Path
import plotly.graph_objects as go

# =========================
# ENVIRONMENT SETUP
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize LangChain with Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=GROQ_API_KEY
)

# =========================
# LOAD MODELS & DATA
# =========================
def load_resources():
    """Load scaler, models, and customer data"""
    models_dir = Path("models")
    
    # Load scaler
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    # Load all SMOTE models
    model_files = {
        "Logistic Regression": "logistic_regression_smote_churn.pkl",
        "Decision Tree": "decision_tree_smote_churn.pkl",
        "Random Forest": "random_forest_smote_churn.pkl",
        "KNN": "knn_smote_churn.pkl",
        "SVM": "svm_smote_churn.pkl",
        "XGBoost": "xgboost_smote_churn.pkl",
    }
    
    models = {}
    for name, filename in model_files.items():
        path = models_dir / filename
        if path.exists():
            models[name] = joblib.load(path)
    
    # Load customer data
    df = pd.read_csv("data/churn.csv")
    
    return scaler, models, df

scaler, models, customer_df = load_resources()
print(f"Loaded {len(models)} models")

# =========================
# FEATURE ENGINEERING
# =========================
def prepare_features(
    country_france, country_germany, country_spain,
    credit_score, is_male, age, tenure, balance,
    num_products, has_cr_card, is_active_member, estimated_salary
):
    """
    Prepare features matching preprocessed_bank_churn.csv exactly
    Order must match training data columns
    """
    features = {
        'Country_France': country_france,
        'Country_Germany': country_germany,
        'Country_Spain': country_spain,
        'CreditScore': credit_score,
        'IsMale': is_male,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
    }
    
    return pd.DataFrame([features])

# =========================
# PREDICTION ENGINE
# =========================
def get_model_probability(model, input_df):
    """Extract probability with fallback logic"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_df)[0][1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(input_df)[0]
        return 1 / (1 + np.exp(-decision))  # Sigmoid
    else:
        return float(model.predict(input_df)[0])

def make_predictions(input_df):
    """Generate predictions from all models"""
    probabilities = {}
    
    for name, model in models.items():
        try:
            prob = get_model_probability(model, input_df)
            probabilities[name] = prob
        except Exception as e:
            print(f"⚠️ Error with {name}: {str(e)}")
            continue
    
    avg_probability = np.mean(list(probabilities.values()))
    return avg_probability, probabilities

# =========================
# VISUALIZATION
# =========================
def create_gauge_chart(probability):
    """Create probability gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFD700'},
                {'range': [60, 100], 'color': '#FF6347'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_model_comparison(probabilities):
    """Create bar chart comparing model predictions"""
    models_list = list(probabilities.keys())
    probs_list = [probabilities[m] * 100 for m in models_list]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs_list,
            y=models_list,
            orientation='h',
            marker=dict(
                color=probs_list,
                colorscale='RdYlGn_r',
                showscale=False
            ),
            text=[f"{p:.1f}%" for p in probs_list],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Model-wise Predictions",
        xaxis_title="Churn Probability (%)",
        yaxis_title="Model",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# =========================
# LANGCHAIN PROMPT TEMPLATES
# =========================

# Risk Analysis Prompt Template
risk_analysis_template = PromptTemplate(
    input_variables=["customer_name", "credit_score", "age", "tenure", "balance", 
                     "num_products", "has_cr_card", "is_active_member", "estimated_salary", 
                     "country", "probability"],
    template="""You are a senior data scientist at a bank.

Customer: {customer_name}

Profile:
- Credit Score: {credit_score}
- Age: {age}
- Tenure: {tenure} years
- Balance: ${balance:,.2f}
- Products: {num_products}
- Credit Card: {has_cr_card}
- Active Member: {is_active_member}
- Salary: ${estimated_salary:,.2f}
- Country: {country}

Churn Risk: {probability:.1%}

Write EXACTLY 3 sentences explaining this risk level.
- If risk > 40%: explain risk factors
- If risk ≤ 40%: explain stability factors
- DO NOT mention percentages or models
"""
)

# Retention Email Prompt Template
retention_email_template = PromptTemplate(
    input_variables=["customer_name", "tenure", "balance", "num_products", 
                     "is_active_member", "explanation"],
    template="""You are a relationship manager at a premium bank.

Customer: {customer_name}

Profile Summary:
- Tenure: {tenure} years
- Balance: ${balance:,.2f}
- Products: {num_products}
- Active: {is_active_member}

Analysis: {explanation}

Write a professional retention email (under 150 words):
- Warm, personalized tone
- Specific incentives for their profile
- Use bullet points
- DO NOT mention "churn", "risk", "prediction"
- Focus on value and exclusive benefits
"""
)

# Create LangChain chains
risk_analysis_chain = LLMChain(llm=llm, prompt=risk_analysis_template)
retention_email_chain = LLMChain(llm=llm, prompt=retention_email_template)

# =========================
# LANGCHAIN GENERATION FUNCTIONS
# =========================
def explain_prediction(probability, features_dict, customer_name):
    """Generate natural language explanation using LangChain + Groq"""
    try:
        country = 'France' if features_dict['Country_France'] else 'Germany' if features_dict['Country_Germany'] else 'Spain'
        
        response = risk_analysis_chain.invoke({
            "customer_name": customer_name,
            "credit_score": f"{features_dict['CreditScore']:.0f}",
            "age": f"{features_dict['Age']:.0f}",
            "tenure": f"{features_dict['Tenure']:.0f}",
            "balance": features_dict['Balance'],
            "num_products": features_dict['NumOfProducts'],
            "has_cr_card": 'Yes' if features_dict['HasCrCard'] else 'No',
            "is_active_member": 'Yes' if features_dict['IsActiveMember'] else 'No',
            "estimated_salary": features_dict['EstimatedSalary'],
            "country": country,
            "probability": probability
        })
        return response['text'].strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def generate_email(probability, features_dict, explanation, customer_name):
    """Generate retention email using LangChain + Groq"""
    try:
        response = retention_email_chain.invoke({
            "customer_name": customer_name,
            "tenure": f"{features_dict['Tenure']:.0f}",
            "balance": features_dict['Balance'],
            "num_products": features_dict['NumOfProducts'],
            "is_active_member": 'Yes' if features_dict['IsActiveMember'] else 'No',
            "explanation": explanation
        })
        return response['text'].strip()
    except Exception as e:
        return f"Error generating email: {str(e)}"

# =========================
# GRADIO INTERFACE
# =========================
def predict_churn(
    customer_id,
    credit_score, geography, gender, age, tenure,
    balance, num_products, has_credit_card, is_active_member, estimated_salary
):
    """Main prediction function"""
    
    # Convert inputs to features
    country_france = 1 if geography == "France" else 0
    country_germany = 1 if geography == "Germany" else 0
    country_spain = 1 if geography == "Spain" else 0
    is_male = 1 if gender == "Male" else 0
    has_cr_card = 1 if has_credit_card else 0
    is_active = 1 if is_active_member else 0
    
    # Prepare features
    input_df = prepare_features(
        country_france, country_germany, country_spain,
        credit_score, is_male, age, tenure, balance,
        num_products, has_cr_card, is_active, estimated_salary
    )
    
    # Make predictions
    avg_prob, model_probs = make_predictions(input_df)
    
    # Create visualizations
    gauge_fig = create_gauge_chart(avg_prob)
    comparison_fig = create_model_comparison(model_probs)
    
    # Risk assessment
    if avg_prob > 0.6:
        risk_level = "HIGH RISK"
        risk_color = "#FF6347"
    elif avg_prob > 0.3:
        risk_level = "MEDIUM RISK"
        risk_color = "#FFD700"
    else:
        risk_level = "LOW RISK"
        risk_color = "#90EE90"
    
    risk_html = f"<h2 style='color: {risk_color};'>{risk_level}</h2>"
    
    # Get customer name
    customer_row = customer_df[customer_df['CustomerId'] == int(customer_id)]
    customer_name = customer_row['Surname'].values[0] if not customer_row.empty else "Customer"
    
    # Features dict for LLM
    features_dict = {
        'Country_France': country_france,
        'Country_Germany': country_germany,
        'Country_Spain': country_spain,
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
    }
    
    # Generate explanations using LangChain + Groq
    explanation = explain_prediction(avg_prob, features_dict, customer_name)
    email = generate_email(avg_prob, features_dict, explanation, customer_name)
    
    return gauge_fig, comparison_fig, risk_html, explanation, email

def load_customer_data(customer_id):
    """Load customer data when ID is selected"""
    customer = customer_df[customer_df['CustomerId'] == int(customer_id)].iloc[0]
    
    return (
        int(customer['CreditScore']),
        customer['Geography'],
        customer['Gender'],
        int(customer['Age']),
        int(customer['Tenure']),
        float(customer['Balance']),
        int(customer['NumOfProducts']),
        bool(customer['HasCrCard']),
        bool(customer['IsActiveMember']),
        float(customer['EstimatedSalary'])
    )

# =========================
# BUILD GRADIO UI
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Bank Churn Prediction") as demo:
    gr.Markdown(
        """
        #  Bank Churn Prediction System
        ### AI-Powered Customer Retention Analysis with LangChain + Groq (Llama 3.3 70B)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Customer Information")
            
            customer_id = gr.Dropdown(
                choices=customer_df['CustomerId'].astype(str).tolist(),
                label="Select Customer ID",
                value=str(customer_df['CustomerId'].iloc[0])
            )
            
            credit_score = gr.Slider(300, 850, value=650, label="Credit Score")
            geography = gr.Radio(["France", "Germany", "Spain"], label="Country", value="France")
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
            age = gr.Slider(18, 100, value=35, label="Age")
            tenure = gr.Slider(0, 10, value=5, label="Tenure (years)")
            balance = gr.Number(value=50000, label="Account Balance ($)")
            num_products = gr.Slider(1, 4, value=2, step=1, label="Number of Products")
            has_credit_card = gr.Checkbox(value=True, label="Has Credit Card")
            is_active_member = gr.Checkbox(value=True, label="Is Active Member")
            estimated_salary = gr.Number(value=60000, label="Estimated Salary ($)")
            
            predict_btn = gr.Button("Analyze Churn Risk", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### Prediction Results")
            
            with gr.Row():
                gauge_plot = gr.Plot(label="Churn Probability")
                comparison_plot = gr.Plot(label="Model Comparison")
            
            risk_level_display = gr.HTML()
            
            gr.Markdown("### AI-Generated Insights (Groq)")
            explanation_output = gr.Textbox(label="Risk Analysis", lines=4, interactive=False)
            
            gr.Markdown("### Personalized Retention Strategy")
            email_output = gr.Textbox(label="Retention Email", lines=8, interactive=False)
    
    # Event handlers
    customer_id.change(
        fn=load_customer_data,
        inputs=[customer_id],
        outputs=[
            credit_score, geography, gender, age, tenure,
            balance, num_products, has_credit_card, is_active_member, estimated_salary
        ]
    )
    
    predict_btn.click(
        fn=predict_churn,
        inputs=[
            customer_id, credit_score, geography, gender, age, tenure,
            balance, num_products, has_credit_card, is_active_member, estimated_salary
        ],
        outputs=[gauge_plot, comparison_plot, risk_level_display, explanation_output, email_output]
    )

# =========================
# LAUNCH
# =========================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )