"""
Investigator Alert Dossier Synthesizer.
Transforms quantitative risk metrics, anomaly scores, and TreeSHAP attributions 
into structured, natural-language investigative briefings for fraud analysts.
"""

import os
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.InvestigatorLLM")


def generate_local_dossier(
    transaction_id: str,
    amount: float,
    time_hour: float,
    risk_score: float,
    calibrated_prob: float,
    anomaly_risk: float,
    risk_tier: str,
    recommendation: str,
    top_risk_drivers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generates a high-fidelity investigative brief using deterministic, rule-based 
    domain synthesis (zero external API latency, works completely offline).
    """
    # 1. Executive Summary
    if risk_tier in ["CRITICAL", "HIGH"]:
        urgency = "URGENT ACTION REQUIRED"
        summary = (
            f"Transaction [{transaction_id}] flagged as {risk_tier} RISK ({risk_score:.1f}/100) "
            f"with a calibrated fraud likelihood of {calibrated_prob * 100:.1f}%. "
            f"The transaction demonstrates severe deviation from normal spending topology "
            f"(Anomaly Risk: {anomaly_risk * 100:.1f}%)."
        )
    elif risk_tier == "MEDIUM":
        urgency = "ELEVATED VIGILANCE"
        summary = (
            f"Transaction [{transaction_id}] flagged with MEDIUM RISK ({risk_score:.1f}/100). "
            f"Moderate risk signals detected with calibrated fraud probability of {calibrated_prob * 100:.1f}%."
        )
    else:
        urgency = "NORMAL"
        summary = (
            f"Transaction [{transaction_id}] evaluated as LOW RISK ({risk_score:.1f}/100). "
            f"Behavior aligns consistently with legitimate historical cardholder baselines."
        )

    # 2. Key Suspicious Indicators from SHAP drivers
    bullet_points = []
    for driver in top_risk_drivers:
        f_name = driver["feature"]
        f_val = driver["value"]
        shap_val = driver["shap_value"]
        desc = driver.get("description", f_name)
        
        bullet_points.append(
            f"• {desc} [{f_name}={f_val:+.3f}]: Contributed +{shap_val:.3f} to the model log-odds risk score."
        )

    if amount > 1000.0:
        bullet_points.append(f"• High Monetary Velocity: Raw transaction amount (${amount:,.2f}) significantly exceeds median baseline.")

    if 1.0 <= time_hour <= 5.0:
        bullet_points.append(f"• Off-Hours Activity: Transaction occurred at {time_hour:.1f}h (early morning window frequently associated with automated bot attacks).")

    # 3. Action Checklist for Investigator
    if risk_tier == "CRITICAL":
        checklist = [
            "1. Temporarily place a security hold on card token.",
            "2. Send immediate push notification / SMS verification to verified device.",
            "3. Inspect subsequent authorization attempts for velocity attacks within the last 60 minutes."
        ]
    elif risk_tier == "HIGH":
        checklist = [
            "1. Cross-reference shipping address against cardholder billing address.",
            "2. Inspect recent IP geolocation velocity (impossible travel check).",
            "3. Contact cardholder if secondary verification fails."
        ]
    elif risk_tier == "MEDIUM":
        checklist = [
            "1. Challenge via 3D Secure 2.0 / OTP on next purchase attempt.",
            "2. Monitor account for repeated small test charges (card testing pattern)."
        ]
    else:
        checklist = [
            "1. No intervention necessary. Transaction auto-approved."
        ]

    dossier = {
        "transaction_id": transaction_id,
        "urgency_level": urgency,
        "risk_tier": risk_tier,
        "risk_score": risk_score,
        "calibrated_fraud_probability": f"{calibrated_prob * 100:.2f}%",
        "anomaly_score": f"{anomaly_risk * 100:.2f}%",
        "executive_summary": summary,
        "key_risk_drivers": bullet_points,
        "investigator_action_checklist": checklist,
        "recommended_action": recommendation
    }

    return dossier


def generate_llm_dossier(
    transaction_id: str,
    amount: float,
    time_hour: float,
    risk_score: float,
    calibrated_prob: float,
    anomaly_risk: float,
    risk_tier: str,
    recommendation: str,
    top_risk_drivers: List[Dict[str, Any]],
    gemini_api_key: str = None
) -> Dict[str, Any]:
    """
    Synthesizes the dossier using an LLM (Gemini API) if API key is provided,
    otherwise gracefully falls back to the deterministic synthesizer.
    """
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return generate_local_dossier(
            transaction_id=transaction_id,
            amount=amount,
            time_hour=time_hour,
            risk_score=risk_score,
            calibrated_prob=calibrated_prob,
            anomaly_risk=anomaly_risk,
            risk_tier=risk_tier,
            recommendation=recommendation,
            top_risk_drivers=top_risk_drivers
        )

    # If API key is available, call Gemini API
    try:
        import urllib.request
        prompt = f"""
You are an expert FinTech Fraud Investigator AI. Generate a professional, concise fraud alert dossier.
Transaction ID: {transaction_id}
Amount: ${amount:,.2f}
Time (Hour): {time_hour:.1f}
Risk Score: {risk_score}/100 ({risk_tier})
Calibrated Fraud Probability: {calibrated_prob*100:.2f}%
Anomaly Risk: {anomaly_risk*100:.2f}%
Recommendation: {recommendation}

Top SHAP Risk Drivers:
{json.dumps(top_risk_drivers, indent=2)}

Return a JSON object with:
- "executive_summary": 2-3 sentences explaining the alert
- "key_risk_drivers": list of string explanations
- "investigator_action_checklist": list of 3 concrete next steps
"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            # Clean possible markdown fencing
            clean_text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_text)
            parsed["transaction_id"] = transaction_id
            parsed["risk_score"] = risk_score
            parsed["risk_tier"] = risk_tier
            parsed["recommended_action"] = recommendation
            return parsed
    except Exception as e:
        logger.warning(f"LLM API call failed ({e}), falling back to deterministic dossier generator.")
        return generate_local_dossier(
            transaction_id=transaction_id,
            amount=amount,
            time_hour=time_hour,
            risk_score=risk_score,
            calibrated_prob=calibrated_prob,
            anomaly_risk=anomaly_risk,
            risk_tier=risk_tier,
            recommendation=recommendation,
            top_risk_drivers=top_risk_drivers
        )
