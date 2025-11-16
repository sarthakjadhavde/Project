# File: hybrid_logic.py
import numpy as np
import config

# Get label-to-integer mapping from config
KSS_ALERT = config.KSS_LABELS["ALERT"]
KSS_DROWSY = config.KSS_LABELS["DROWSY"]
KSS_VERY_DROWSY = config.KSS_LABELS["VERY_DROWSY"]

def make_final_prediction(physio_probs, context_risk):
    """
    Combines outputs from the two models to make a final decision.
    """
    prob_alert = physio_probs[KSS_ALERT]
    prob_drowsy = physio_probs[KSS_DROWSY]
    prob_very_drowsy = physio_probs[KSS_VERY_DROWSY]
    is_risky = (context_risk == 1)

    # Rule 1: High Confidence "Very Drowsy"
    if prob_very_drowsy > 0.85:
        return "Very Drowsy"

    # Rule 2: High Confidence "Alert"
    if prob_alert > 0.8 and not is_risky:
        return "Alert"

    # Rule 3: Tie-Breaker (Ambiguous Case)
    is_ambiguous = (prob_drowsy > prob_alert) and (prob_drowsy < 0.7)
    if is_ambiguous and is_risky:
        return "Drowsy"

    # Rule 4: Default Fallback (trust Model 1's best guess)
    default_class = np.argmax(physio_probs)
    if default_class == KSS_ALERT:
        return "Alert"
    elif default_class == KSS_DROWSY:
        return "Drowsy"
    else:
        return "Very Drowsy"
        