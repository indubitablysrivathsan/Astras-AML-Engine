"""
Investigation Chatbot Service
Provides conversational AI capabilities to ask questions about specific AML alerts
using the local Mistral 7B LLM.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

from services.sar.sar_generator import (
    load_alert_data,
    format_customer_info,
    format_transaction_summary,
    format_behavioral_context,
    get_shap_drivers
)

def build_investigation_prompt(alert_id, user_message, chat_history, features_df, explainer, feature_cols, bsi_df, db_path=DB_PATH):
    """Builds the comprehensive prompt for investigating an alert."""
    alert, customer, transactions = load_alert_data(alert_id, db_path)
    
    customer_info = format_customer_info(customer)
    txn_summary = format_transaction_summary(transactions, top_n=30)
    behavioral_context = format_behavioral_context(alert['customer_id'], features_df, bsi_df)
    top_drivers, risk_score = get_shap_drivers(alert, features_df, explainer, feature_cols)
    
    shap_text = "Top Risk Drivers (SHAP Analysis):"
    for _, d in top_drivers.head(5).iterrows():
        feat_name = d['feature'].replace('_', ' ').title()
        shap_text += f"\n- {feat_name}: SHAP impact {d['shap_value']:+.3f}"

    system_context = f"""You are an expert Anti-Money Laundering (AML) AI Assistant. 
You are currently helping an analyst investigate Alert #{alert['alert_id']} ({alert['alert_type'].replace('_', ' ').title()}).

Use the provided context to answer the user's questions about this alert concisely and accurately.
If the information is not present in the context below, state that you don't know based on the provided data. Do not hallucinate transactions.

--- ALERT CONTEXT ---
Customer Information:
{customer_info}

Transaction Summary:
{txn_summary}

Alert Details:
Severity: {alert['severity'].title()}
Risk Score: {risk_score:.2%}
Total Alert Volume: ${alert['total_amount']:,.2f}

{shap_text}

{behavioral_context}
---------------------
"""
    
    # Append history
    conversation_text = ""
    for msg in chat_history:
         role = "System/AI Agent" if msg["role"] == "assistant" else "User/Analyst"
         conversation_text += f"{role}: {msg['content']}\n"
        
    conversation_text += f"User/Analyst: {user_message}\nSystem/AI Agent:"
    
    return system_context + "\nConversation History:\n" + conversation_text


def stream_investigation_response(llm, alert_id, user_message, chat_history, features_df, explainer, feature_cols, bsi_df, db_path=DB_PATH):
    """Streams the response from the LLM."""
    prompt = build_investigation_prompt(alert_id, user_message, chat_history, features_df, explainer, feature_cols, bsi_df, db_path)
    
    for chunk in llm.stream(prompt):
        yield chunk

