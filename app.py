
"""
SmartDesk - Banking Intent Classifier and AI Chatbot

Features:
1. TinyBERT Banking Intent Classification
2. DistilBERT Sentiment Analysis
3. Complaint Priority Detection
4. Rule-Based Banking Issue Detection
5. Banking FAQ Recommendations
6. Prediction History
7. CSV Export
8. Ollama Qwen2.5 AI Chatbot
9. English, Hindi and Marathi Support
10. Safe predefined responses for suspected banking fraud
"""

import re
import streamlit as st
import torch
import pandas as pd
import ollama

from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# CONFIGURATION
MODEL_DIR = "./smartdesk_tinybert_model"

SENTIMENT_MODEL = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

OLLAMA_MODEL = "qwen2.5:3b"

MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# PAGE SETUP

st.set_page_config(
    page_title="ArthMitra - Banking AI",
    page_icon="🏦",
    layout="wide"
)

st.title("ArthMitra - Banking AI Assistant")

st.caption(
    "TinyBERT Banking Intent Classification "
    "+ Multilingual AI Banking Chatbot"
)


# LOAD TINYBERT
@st.cache_resource(show_spinner="Loading TinyBERT model...")
def load_model(model_dir):

    path = Path(model_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"Model folder not found: {model_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    loaded_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    )

    loaded_model.to(DEVICE)
    loaded_model.eval()

    id2label = loaded_model.config.id2label

    labels = [
        id2label.get(i, id2label.get(str(i), str(i)))
        for i in range(loaded_model.config.num_labels)
    ]

    return tokenizer, loaded_model, labels


try:
    tokenizer, model, label_names = load_model(MODEL_DIR)

except Exception as e:
    st.error(f"Unable to load TinyBERT model: {e}")
    st.stop()


# TEXT NORMALIZATION
def normalize_message(message):

    text = str(message).lower().strip()

    text = re.sub(r"[!?.,।;:]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# HINDI AND MARATHI BANKING PHRASES
BANKING_PHRASES = {

    # Roman Hindi
    "mera debit card block ho gaya hai": "my debit card is blocked",
    "mera debit card block ho gaya": "my debit card is blocked",
    "mera card block ho gaya hai": "my card is blocked",
    "mera atm card block ho gaya hai": "my atm card is blocked",
    "mera account block ho gaya hai": "my bank account is blocked",
    "mera bank account block ho gaya hai": "my bank account is blocked",
    "mera account freeze ho gaya hai": "my bank account is frozen",
    "mere account se paise kat gaye": "money has been deducted from my account",
    "mere account se paise kat gaye hain": "money has been deducted from my account",
    "mere account se paise kat gaye lekin cash nahi mila": "money was deducted but cash was not received",
    "atm se paise nahi nikle": "cash was not dispensed by the atm",
    "atm se paise nahi nikle lekin paise kat gaye": "money was deducted but the atm did not dispense cash",
    "mera upi kaam nahi kar raha": "my upi is not working",
    "mera upi kaam nahi kar raha hai": "my upi is not working",
    "upi payment fail ho gaya": "my upi payment failed",
    "payment fail ho gaya": "my payment failed",
    "main password bhool gaya": "i forgot my password",
    "main pin bhool gaya": "i forgot my pin",
    "mujhe paise receive nahi hue": "i have not received the money",
    "mere account mein paise nahi aaye": "money has not been credited to my account",
    "mujhe refund nahi mila": "i have not received my refund",
    "mera account hack ho gaya": "my bank account has been hacked",
    "mere account se kisi ne paise nikal liye": "someone withdrew money from my account without authorization",
    "kisi ne mere account se paise nikal liye": "someone withdrew money from my account without authorization",

    # Devanagari Hindi
    "मेरा डेबिट कार्ड ब्लॉक हो गया है": "my debit card is blocked",
    "मेरा कार्ड ब्लॉक हो गया है": "my card is blocked",
    "मेरा एटीएम कार्ड ब्लॉक हो गया है": "my atm card is blocked",
    "मेरा बैंक अकाउंट ब्लॉक हो गया है": "my bank account is blocked",
    "मेरा खाता फ्रीज हो गया है": "my bank account is frozen",
    "मेरे खाते से पैसे कट गए": "money has been deducted from my account",
    "मेरे अकाउंट से पैसे कट गए": "money has been deducted from my account",
    "एटीएम से पैसे नहीं निकले": "cash was not dispensed by the atm",
    "मेरा यूपीआई काम नहीं कर रहा है": "my upi is not working",
    "मेरा पेमेंट फेल हो गया": "my payment failed",
    "मेरा यूपीआई पेमेंट फेल हो गया": "my upi payment failed",
    "मैं अपना पासवर्ड भूल गया": "i forgot my password",
    "मुझे पैसे नहीं मिले": "i have not received the money",
    "मेरे खाते से किसी ने पैसे निकाल लिए": "someone withdrew money from my account without authorization",
    "मुझे रिफंड नहीं मिला": "i have not received my refund",
    "मेरा अकाउंट हैक हो गया है": "my bank account has been hacked",
    "मेरे खाते से अनजान व्यक्ति ने पैसे निकाल लिए": "an unknown person withdrew money from my account",

    # Roman Marathi
    "majha debit card block jhala ahe": "my debit card is blocked",
    "maza debit card block zala ahe": "my debit card is blocked",
    "majhe debit card block jhale ahe": "my debit card is blocked",
    "majha debit card block zala aahe": "my debit card is blocked",
    "majha debit card block zala": "my debit card is blocked",
    "majha card block jhala ahe": "my card is blocked",
    "majha account block jhala": "my bank account is blocked",
    "majha account block jhala ahe": "my bank account is blocked",
    "majhe paise kat zale": "money has been deducted from my account",
    "majhe paise kaple gele": "money has been deducted from my account",
    "paise account madhun gele": "money has been debited from my account",
    "atm madhun paise ale nahi": "cash was not dispensed by the atm",
    "majha upi chalat nahi": "my upi is not working",
    "majha upi kaam karat nahi": "my upi is not working",
    "payment fail jhala": "my payment failed",
    "mala paise milale nahi": "i have not received the money",
    "majha account hack jhala": "my bank account has been hacked",
    "mala refund milala nahi": "i have not received my refund",
    "majhya account madhun konitari paise kadhle": "someone withdrew money from my account without authorization",

    # Devanagari Marathi
    "माझे डेबिट कार्ड ब्लॉक झाले आहे": "my debit card is blocked",
    "माझं डेबिट कार्ड ब्लॉक झालं आहे": "my debit card is blocked",
    "माझा डेबिट कार्ड ब्लॉक झाला आहे": "my debit card is blocked",
    "माझे कार्ड ब्लॉक झाले आहे": "my card is blocked",
    "माझे खाते ब्लॉक झाले आहे": "my bank account is blocked",
    "माझ्या खात्यातून पैसे कट झाले": "money has been deducted from my account",
    "माझ्या खात्यातून पैसे कापले गेले": "money has been deducted from my account",
    "एटीएममधून पैसे आले नाहीत": "cash was not dispensed by the atm",
    "माझे यूपीआय चालत नाही": "my upi is not working",
    "मला पैसे मिळाले नाहीत": "i have not received the money",
    "माझे खाते हॅक झाले आहे": "my bank account has been hacked",
    "मला रिफंड मिळाला नाही": "i have not received my refund",
    "माझ्या खात्यातून कोणीतरी पैसे काढले": "someone withdrew money from my account without authorization",
    "माझ्या खात्यातून पैसे काढले गेले": "someone withdrew money from my account without authorization",
    "कुणीतरी माझ्या खात्यातून पैसे काढले": "someone withdrew money from my account without authorization"
}


def preprocess_banking_query(message):

    normalized = normalize_message(message)

    return BANKING_PHRASES.get(normalized)


# LANGUAGE DETECTION
def detect_customer_language(message):

    text = message.lower()

    marathi_words = [
        "माझा", "माझे", "माझं", "माझ्या",
        "झाले", "झालं", "झाला", "आहे",
        "मधून", "मला", "नाहीत", "खात्यातून",
        "कुणीतरी", "काढले", "काढले गेले"
    ]

    if any(word in text for word in marathi_words):
        return "Marathi"

    hindi_words = [
        "मेरा", "मेरी", "मेरे", "मुझे",
        "है", "हैं", "गया", "गई", "हुआ",
        "खाते", "खाता", "निकाल"
    ]

    if any(word in text for word in hindi_words):
        return "Hindi"

    words = set(re.findall(r"[a-z]+", text))

    roman_marathi = {
        "majha", "maza", "majhi", "majhe",
        "jhala", "zala", "aahe", "ahe",
        "madhun", "mala", "tumcha", "kay",
        "nahi", "zale", "jhale", "kadhle",
        "konitari"
    }

    roman_hindi = {
        "mera", "meri", "mere", "hai",
        "hain", "gaya", "gayi", "hua",
        "mujhe", "paise", "khata", "nahi",
        "kya", "kaise", "karna", "chahiye"
    }

    marathi_count = len(words & roman_marathi)
    hindi_count = len(words & roman_hindi)

    if marathi_count > hindi_count:
        return "Marathi"

    if hindi_count > 0:
        return "Hindi"

    if any("\u0900" <= char <= "\u097F" for char in message):
        return "Hindi"

    return "English"


# FRAUD DETECTION - IMPORTANT SAFETY CHECK

def is_suspected_fraud(message):

    text = normalize_message(message)

    fraud_keywords = [

        # English
        "fraud",
        "scam",
        "stolen",
        "unauthorized",
        "unauthorised",
        "identity theft",
        "account hacked",
        "account has been hacked",
        "money stolen",
        "suspicious transaction",
        "unknown transaction",
        "without my permission",
        "without authorization",
        "without authorisation",
        "someone withdrew money",
        "someone took money",
        "unknown person withdrew",
        "i did not make this transaction",
        "i didn't make this transaction",
        "i did not withdraw",
        "money withdrawn without",

        # Hindi
        "मेरे खाते से किसी ने",
        "मेरे अकाउंट से किसी ने",
        "अनजान व्यक्ति ने पैसे",
        "मेरे खाते से पैसे निकाल",
        "मेरे खाते से पैसे निकाले",
        "बिना अनुमति पैसे निकाले",
        "मेरे पैसे चोरी",
        "खाते से धोखाधड़ी",

        # Marathi
        "माझ्या खात्यातून कोणीतरी",
        "कुणीतरी माझ्या खात्यातून",
        "माझ्या खात्यातून पैसे काढले",
        "माझ्या खात्यातून पैसे काढले गेले",
        "माझ्या खात्यातून पैसे चोरले",
        "माझ्या परवानगीशिवाय पैसे",
        "माझे पैसे चोरी",
        "खात्यातून अनधिकृत पैसे",

        # Roman Hindi
        "mere account se kisi ne paise",
        "kisi ne mere account se paise",
        "mere account se paise nikal liye",
        "mere paise chori",

        # Roman Marathi
        "majhya khatyatun konitari",
        "majhya account madhun konitari",
        "konitari majhya account madhun",
        "majhya account madhun paise kadhle",
        "majhe paise chori"
    ]

    if any(keyword in text for keyword in fraud_keywords):
        return True

    return False


# OLLAMA CONNECTION

def check_ollama():

    try:

        result = ollama.list()

        if isinstance(result, dict):
            models = result.get("models", [])
        else:
            models = getattr(result, "models", [])

        available_models = []

        for item in models:

            if isinstance(item, dict):
                name = item.get("model") or item.get("name", "")
            else:
                name = (
                    getattr(item, "model", None)
                    or getattr(item, "name", "")
                )

            if name:
                available_models.append(str(name))

        return any(
            name == OLLAMA_MODEL
            or name.startswith(OLLAMA_MODEL + ":")
            for name in available_models
        )

    except Exception as e:

        print("Ollama connection error:", str(e))

        return False


# OLLAMA CHAT HELPER

def ask_ollama(messages, temperature=0.2, num_predict=350):

    try:

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": num_predict
            }
        )

        answer = response.message.content.strip()

        if not answer:
            raise RuntimeError("Qwen returned an empty response.")

        return answer

    except Exception as e:

        error_message = str(e)

        print("Qwen chatbot error:", error_message)

        raise RuntimeError(
            f"Unable to get a response from Qwen: {error_message}"
        ) from e


# TRANSLATION TO ENGLISH

def translate_to_english(message):

    direct_translation = preprocess_banking_query(message)

    if direct_translation:
        return direct_translation

    # Do not use an LLM to translate suspected fraud messages.
    # Use a neutral English description for downstream analysis.
    if is_suspected_fraud(message):
        return (
            "Someone may have withdrawn money from the customer's "
            "bank account without authorization."
        )

    language = detect_customer_language(message)

    if language == "English":
        return message

    translation_prompt = """
You are a professional Hindi and Marathi banking translator.

Translate the customer's entire message into natural English.

You understand:
- Hindi written in Devanagari
- Hindi written in Roman letters
- Marathi written in Devanagari
- Marathi written in Roman letters
- Mixed Hindi-English and Marathi-English

Rules:
1. Translate the actual meaning.
2. Preserve banking terminology.
3. Do not answer the customer.
4. Do not add information.
5. Do not change the meaning.
6. Return ONLY the English translation.
"""

    return ask_ollama(
        [
            {
                "role": "system",
                "content": translation_prompt
            },
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=0,
        num_predict=150
    )


# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():

    sent_tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL
    )

    sent_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL
    )

    sent_model.to(DEVICE)
    sent_model.eval()

    return sent_tokenizer, sent_model


def analyze_sentiment(message):

    sent_tokenizer, sent_model = load_sentiment_model()

    inputs = sent_tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        logits = sent_model(**inputs).logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

    pred_id = int(torch.argmax(probabilities).item())

    confidence = float(probabilities[pred_id].item())

    label = sent_model.config.id2label[pred_id].upper()

    if confidence < 0.65:
        label = "UNCERTAIN"

    return label, confidence


# ============================================================
# TINYBERT INTENT CLASSIFICATION
# ============================================================

def predict_intent(message):

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        logits = model(**inputs).logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

    pred_id = int(torch.argmax(probabilities).item())

    confidence = float(probabilities[pred_id].item())

    intent = label_names[pred_id]

    top_probs, top_ids = torch.topk(
        probabilities,
        k=min(5, len(label_names))
    )

    top_k = []

    for probability, index in zip(top_probs, top_ids):

        top_k.append({
            "Intent": label_names[int(index)],
            "Confidence": float(probability)
        })

    return intent, confidence, top_k


# ============================================================
# RULE-BASED BANKING ISSUE DETECTION
# ============================================================

def detect_banking_issue(original_message, english_message):

    # Fraud detection has the highest priority.
    if (
        is_suspected_fraud(original_message)
        or is_suspected_fraud(english_message)
    ):
        return "Possible fraud or unauthorized transaction"

    text = normalize_message(
        original_message + " " + english_message
    )

    card_keywords = [
        "debit card is blocked",
        "my card is blocked",
        "card blocked",
        "card block",
        "debit card block",
        "मेरा डेबिट कार्ड ब्लॉक",
        "मेरा कार्ड ब्लॉक",
        "माझे डेबिट कार्ड ब्लॉक",
        "माझा डेबिट कार्ड ब्लॉक",
        "माझे कार्ड ब्लॉक"
    ]

    if any(word in text for word in card_keywords):
        return "Possible blocked card issue"

    account_keywords = [
        "account is blocked",
        "account blocked",
        "account is frozen",
        "account freeze",
        "bank account is blocked",
        "मेरा बैंक अकाउंट ब्लॉक",
        "मेरा खाता फ्रीज",
        "माझे खाते ब्लॉक"
    ]

    if any(word in text for word in account_keywords):
        return "Possible blocked or frozen account"

    atm_keywords = [
        "cash was not dispensed",
        "cash not received",
        "atm withdrawal failed",
        "money was deducted but",
        "atm did not dispense",
        "एटीएम से पैसे नहीं निकले",
        "एटीएममधून पैसे आले नाहीत"
    ]

    if any(word in text for word in atm_keywords):
        return "Possible ATM cash withdrawal issue"

    debit_keywords = [
        "money has been deducted",
        "money has been debited",
        "money was deducted",
        "पैसे कट गए",
        "पैसे कट झाले"
    ]

    if any(word in text for word in debit_keywords):
        return "Possible account debit issue"

    payment_keywords = [
        "payment failed",
        "payment failure",
        "transaction failed",
        "upi payment failed"
    ]

    if any(word in text for word in payment_keywords):
        return "Possible payment failure"

    return "No specific rule-based issue identified"


# ============================================================
# COMPLAINT PRIORITY
# ============================================================

def detect_priority(message, intent, banking_issue):

    if (
        is_suspected_fraud(message)
        or "Possible fraud" in banking_issue
    ):
        return "Critical"

    text = normalize_message(
        message + " " + intent + " " + banking_issue
    )

    high_keywords = [
        "card blocked",
        "card is blocked",
        "debit card is blocked",
        "account blocked",
        "account is blocked",
        "account frozen",
        "cash not received",
        "cash was not dispensed",
        "payment failed",
        "transfer pending",
        "transfer failed",
        "transaction failed",
        "money has been deducted",
        "money has been debited"
    ]

    if any(word in text for word in high_keywords):
        return "High"

    if banking_issue in [
        "Possible blocked card issue",
        "Possible blocked or frozen account",
        "Possible ATM cash withdrawal issue",
        "Possible account debit issue",
        "Possible payment failure"
    ]:
        return "High"

    return "Normal"


# ============================================================
# BANKING FAQ DATABASE
# ============================================================

FAQ_DATABASE = {

    "cash_withdrawal": (
        "How can I withdraw cash?",
        "Use your debit card at an ATM and follow "
        "the on-screen instructions."
    ),

    "cash_withdrawal_pending": (
        "What should I do if cash is not received?",
        "If your account was debited but cash was not "
        "dispensed, report the ATM transaction to your bank."
    ),

    "cash_withdrawal_failed": (
        "What happens when an ATM withdrawal fails?",
        "Check your account statement. If money was debited, "
        "contact your bank to raise a complaint."
    ),

    "cash_withdrawal_not_recognised": (
        "What should I do about an unknown withdrawal?",
        "Immediately contact your bank through its official "
        "support channel and report the unauthorized transaction."
    ),

    "cash_withdrawal_not_recognized": (
        "What should I do about an unknown withdrawal?",
        "Immediately contact your bank through its official "
        "support channel and report the unauthorized transaction."
    )
}


def recommend_faq(intent):

    intent = intent.lower()

    if intent in FAQ_DATABASE:
        return FAQ_DATABASE[intent]

    if "card" in intent:

        return (
            "How do I manage my bank card?",
            "Use your bank's official app or contact "
            "customer support for card-related assistance."
        )

    if "transfer" in intent or "beneficiary" in intent:

        return (
            "How can I track a bank transfer?",
            "Check your transaction history and verify "
            "the recipient details."
        )

    if "cash" in intent or "atm" in intent:

        return (
            "What should I do about an ATM issue?",
            "Check your transaction history and contact "
            "your bank if the issue remains unresolved."
        )

    if "pin" in intent:

        return (
            "What should I do if my PIN is blocked?",
            "Contact your bank through its official support "
            "channel to learn how to reset your PIN."
        )

    return (
        "How can I get banking assistance?",
        "Please contact your bank's official customer support "
        "or visit its official help centre."
    )


def get_safe_faq(intent, confidence, threshold):

    if confidence < threshold:

        return (
            "Your banking issue needs further verification.",
            "TinyBERT could not confidently identify your issue. "
            "Please describe the problem in more detail or "
            "contact your bank through its official support channel."
        )

    return recommend_faq(intent)


# ============================================================
# FIXED FRAUD RESPONSES
# ============================================================

def get_fraud_response(language):

    responses = {

        "Marathi": """
तुमच्या खात्यातून अनधिकृतपणे पैसे काढले गेले असतील, तर कृपया त्वरित खालील पावले उचला.

1. तुमच्या बँकेच्या अधिकृत हेल्पलाइनवर त्वरित संपर्क साधा आणि अनधिकृत व्यवहाराची तक्रार नोंदवा.
2. बँकेच्या अधिकृत मोबाइल ॲप किंवा हेल्पलाइनद्वारे संबंधित डेबिट कार्ड किंवा खाते सुरक्षित करा. आवश्यक असल्यास कार्ड ब्लॉक करा.
3. व्यवहाराची तारीख, रक्कम आणि Transaction ID उपलब्ध असल्यास नोंदवून ठेवा.
4. भारताच्या अधिकृत सायबर क्राइम पोर्टलवर https://cybercrime.gov.in तक्रार नोंदवा. आर्थिक सायबर फसवणुकीसाठी 1930 या हेल्पलाइनवरही त्वरित संपर्क साधा.
5. बँकेकडून तक्रार क्रमांक (Complaint Reference Number) घ्या.

महत्त्वाचे: कोणालाही OTP, ATM PIN, पासवर्ड किंवा पूर्ण कार्ड नंबर सांगू नका. आम्हाला तुमचे गोपनीय तपशील देण्याची आवश्यकता नाही.
""",

        "Hindi": """
यदि आपके बैंक खाते से बिना अनुमति पैसे निकाले गए हैं, तो तुरंत ये कदम उठाएं:

1. अपने बैंक की आधिकारिक हेल्पलाइन पर संपर्क करके अनधिकृत लेनदेन की शिकायत दर्ज करें।
2. बैंक के आधिकारिक ऐप या हेल्पलाइन से संबंधित कार्ड या खाते को सुरक्षित करें। जरूरत पड़ने पर कार्ड ब्लॉक करें।
3. लेनदेन की तारीख, राशि और Transaction ID नोट करें।
4. भारत के आधिकारिक साइबर क्राइम पोर्टल https://cybercrime.gov.in पर शिकायत दर्ज करें। वित्तीय साइबर धोखाधड़ी के लिए 1930 पर भी तुरंत संपर्क करें।
5. बैंक से शिकायत संदर्भ संख्या प्राप्त करें।

सुरक्षा: किसी को भी OTP, ATM PIN, पासवर्ड या पूरा कार्ड नंबर न बताएं। हमें आपके गोपनीय विवरण की आवश्यकता नहीं है।
""",

        "English": """
If money was withdrawn from your account without your permission, take these steps immediately:

1. Contact your bank through its official helpline and report the unauthorized transaction.
2. Secure the affected account or card through the bank's official app or helpline. Block the card if necessary.
3. Keep the transaction date, amount and transaction ID, if available.
4. Report the incident at India's official cybercrime portal: https://cybercrime.gov.in. You can also call 1930 immediately for financial cyber fraud.
5. Obtain a complaint reference number from your bank.

Security: Never share your OTP, ATM PIN, password or complete card number with anyone. SmartDesk does not need these details.
"""
    }

    return responses.get(language, responses["English"])


# ============================================================
# CHATBOT LANGUAGE INSTRUCTIONS
# ============================================================

def get_language_instruction(language):

    instructions = {

        "English": """
Respond only in natural, simple English.
Do not mix Hindi or Marathi into the answer.
""",

        "Hindi": """
Respond ONLY in natural Hindi using Devanagari script.
Use proper Hindi grammar and simple words.
Do not respond in Marathi or Roman Hindi.
""",

        "Marathi": """
Respond ONLY in natural Marathi using Devanagari script.
Use correct Marathi grammar and simple, polite words.
Do not respond in Hindi or Roman Marathi.
"""
    }

    return instructions[language]


# ============================================================
# AI CHATBOT RESPONSE GENERATION
# ============================================================

def generate_chatbot_response(
    user_message,
    english_message,
    intent,
    confidence,
    banking_issue,
    priority,
    history,
    preferred_language,
    review_threshold
):

    detected_language = detect_customer_language(user_message)

    if preferred_language == "Automatic - Detect from message":
        response_language = detected_language
    else:
        response_language = preferred_language

    # CRITICAL SAFETY RULE:
    # Never send suspected fraud messages to Qwen for advice.
    # Use a fixed response to avoid unsafe or misleading instructions.
    if (
        is_suspected_fraud(user_message)
        or is_suspected_fraud(english_message)
        or "Possible fraud" in banking_issue
    ):
        return get_fraud_response(response_language)

    language_instruction = get_language_instruction(
        response_language
    )

    if confidence < review_threshold:

        faq_information = """
The intent prediction is uncertain.
Do not use an unrelated FAQ.
Understand the customer's actual message.
"""

    else:

        faq_question, faq_answer = recommend_faq(intent)

        faq_information = f"""
FAQ question: {faq_question}
FAQ information: {faq_answer}
Use only if relevant to the customer's actual problem.
"""

    system_prompt = f"""
You are SmartDesk, a professional banking customer support assistant.

Your responsibility is to provide practical solutions,
not merely repeat the customer's problem.

LANGUAGE:
{language_instruction}

ORIGINAL CUSTOMER MESSAGE:
{user_message}

ENGLISH TRANSLATION:
{english_message}

BANKING ISSUE:
{banking_issue}

PREDICTED TINYBERT INTENT:
{intent}

TINYBERT CONFIDENCE:
{confidence:.2%}

HUMAN REVIEW THRESHOLD:
{review_threshold:.2%}

COMPLAINT PRIORITY:
{priority}

FAQ INFORMATION:
{faq_information}

IMPORTANT RESPONSE RULES:

1. Never simply repeat the customer's question.
2. Give a useful solution immediately.
3. Use numbered steps wherever appropriate.
4. Do not invent bank policies, phone numbers or transaction details.
5. Ask only one follow-up question when necessary.
6. Keep the response concise and understandable.

DEBIT CARD BLOCKED:
- Check card status in the official banking app.
- Use an official unblock option only if the bank provides one.
- Otherwise contact official customer care or visit the branch.
- If misuse is suspected, report it immediately.
- Never promise that the card can definitely be unblocked.

BANK ACCOUNT BLOCKED:
- Check notifications from the bank.
- Contact the bank through official channels.
- Ask about the reason and required verification.
- Visit the branch if required.

ATM CASH NOT RECEIVED:
- Check the account statement.
- Keep the ATM receipt and transaction reference.
- Report the failed withdrawal to the bank.
- Request a complaint reference number.
- Do not promise a specific refund time.

UPI PAYMENT FAILED:
- Check the UPI transaction status.
- Check whether the amount was debited.
- Keep the transaction reference number.
- Raise a complaint through the official UPI app or bank.

FORGOTTEN PIN OR PASSWORD:
- Explain the official reset procedure through the bank app,
  website or branch.

SECURITY:
- Never ask for OTP, PIN, password or full card number.
- Never claim to have accessed the customer's account.
- Never claim to have unblocked a card.
- Never claim a transaction was completed.
- Never promise a refund.
- Never ask the customer to share confidential banking details.

LOW CONFIDENCE:
If TinyBERT confidence is below the review threshold,
treat the predicted intent as uncertain.
Understand the actual customer message instead of forcing an FAQ.

RESPONSE FORMAT:
1. Short helpful opening.
2. Practical solution steps.
3. Relevant safety reminder.
4. One short follow-up question only if useful.

Be natural, professional, empathetic and solution-oriented.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Include recent conversation history.
    messages.extend(history[-8:])

    messages.append({
        "role": "user",
        "content": (
            f"Customer's original message:\n{user_message}\n\n"
            f"English translation:\n{english_message}\n\n"
            "Provide a practical solution. Do not just repeat "
            "the customer's problem."
        )
    })

    return ask_ollama(
        messages,
        temperature=0.2,
        num_predict=350
    )


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Settings")

REVIEW_THRESHOLD = st.sidebar.slider(
    "Human-review threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.60,
    step=0.05
)

st.sidebar.divider()

st.sidebar.markdown(f"**Device:** `{DEVICE}`")

st.sidebar.markdown(
    f"**Total intents:** {len(label_names)}"
)

st.sidebar.markdown("**Intent model:** TinyBERT")

st.sidebar.markdown(
    f"**Chatbot model:** {OLLAMA_MODEL}"
)

ollama_available = check_ollama()

if ollama_available:

    st.sidebar.success(
        f"Connected: {OLLAMA_MODEL}"
    )

else:

    st.sidebar.error(
        "Qwen model unavailable"
    )

    st.sidebar.caption(
        "Check Ollama server and installed model."
    )


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🏦 Intent Classifier",
    "📜 Prediction History",
    "💬 AI Banking Chatbot"
])


# ============================================================
# TAB 1: INTENT CLASSIFICATION
# ============================================================

with tab1:

    st.subheader("Customer Query Analysis")

    st.caption(
        "Enter a banking query in English, Hindi or Marathi."
    )

    examples = [

        "My card was charged twice for the same purchase.",
        "I forgot my banking app password and cannot log in.",
        "I transferred money but the recipient has not received it.",
        "How do I activate my new debit card?",
        "I need help with a card payment that failed.",
        "Someone made an unauthorized transaction from my account.",
        "My PIN is blocked after too many wrong attempts.",
        "Mera debit card block ho gaya hai",
        "मेरे खाते से पैसे कट गए",
        "Majha debit card block jhala ahe",
        "माझ्या खात्यातून पैसे कट झाले",
        "माझ्या खात्यातून कोणीतरी पैसे काढले"
    ]

    message = st.text_area(
        "Enter customer message",
        height=130,
        placeholder="Type a banking query..."
    )

    selected = st.selectbox(
        "Or select an example",
        [""] + examples
    )

    if selected:
        message = selected

    if st.button(
        "🚀 Classify Intent",
        type="primary",
        use_container_width=True
    ):

        if not message.strip():

            st.warning("Please enter a customer message.")

        else:

            try:

                with st.spinner("Analyzing customer message..."):

                    original_message = message.strip()

                    english_message = translate_to_english(
                        original_message
                    )

                    intent, confidence, top_k = predict_intent(
                        english_message
                    )

                    status = (
                        "Needs human review"
                        if confidence < REVIEW_THRESHOLD
                        else "Automatically classified"
                    )

                    sentiment, sentiment_confidence = (
                        analyze_sentiment(english_message)
                    )

                    banking_issue = detect_banking_issue(
                        original_message,
                        english_message
                    )

                    priority = detect_priority(
                        original_message + " " + english_message,
                        intent,
                        banking_issue
                    )

                    faq_question, faq_answer = get_safe_faq(
                        intent,
                        confidence,
                        REVIEW_THRESHOLD
                    )

                    # Do not show a generic FAQ for suspected fraud.
                    if "Possible fraud" in banking_issue:
                        faq_question = "What should I do about an unauthorized withdrawal?"
                        faq_answer = (
                            "Contact your bank immediately through its official "
                            "helpline. Report the transaction and call 1930 for "
                            "financial cyber fraud in India."
                        )

                    record = {

                        "Time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                        "Customer Message": original_message,

                        "English Translation": english_message,

                        "Predicted Intent": intent,

                        "Confidence": round(confidence * 100, 2),

                        "Detected Banking Issue": banking_issue,

                        "Sentiment": sentiment,

                        "Priority": priority,

                        "Status": status
                    }

                    st.session_state.history.append(record)

                st.divider()

                st.subheader("📌 Classification Results")

                col1, col2, col3 = st.columns(3)

                col1.metric("Predicted Intent", intent)

                col2.metric("Confidence", f"{confidence:.1%}")

                col3.metric("Sentiment", sentiment)

                if status == "Automatically classified":
                    st.success(f"✅ {status}")
                else:
                    st.warning(f"⚠️ {status}")

                st.markdown("**Original customer message:**")
                st.info(original_message)

                st.markdown("**English translation:**")
                st.info(english_message)

                st.subheader("🔎 Rule-Based Banking Issue Detection")

                st.write(banking_issue)

                st.caption(
                    "This is a separate rule-based indication, "
                    "not a TinyBERT-trained intent."
                )

                st.subheader("🚨 Complaint Priority")

                if priority == "Critical":
                    st.error("🔴 CRITICAL PRIORITY")

                elif priority == "High":
                    st.warning("🟠 HIGH PRIORITY")

                else:
                    st.success("🟢 NORMAL PRIORITY")

                st.subheader("😊 Sentiment Analysis")

                st.write(f"Sentiment: **{sentiment}**")

                st.progress(sentiment_confidence)

                st.caption(
                    f"DistilBERT confidence: {sentiment_confidence:.1%}. "
                    "This English sentiment model gives an estimate, "
                    "not a definitive measure of customer emotion."
                )

                st.subheader("🎯 Top-5 Intent Predictions")

                for i, item in enumerate(top_k, 1):

                    st.write(
                        f"{i}. **{item['Intent']}** - "
                        f"{item['Confidence']:.1%}"
                    )

                    st.progress(item["Confidence"])

                st.divider()

                st.subheader("💡 Recommended Banking FAQ")

                st.markdown(f"**{faq_question}**")

                st.info(faq_answer)

                with st.expander("View complete prediction details"):

                    st.json({

                        "original_message": original_message,
                        "english_translation": english_message,
                        "predicted_intent": intent,
                        "confidence": confidence,
                        "banking_issue": banking_issue,
                        "sentiment": sentiment,
                        "priority": priority,
                        "status": status,
                        "top_5": top_k,
                        "faq": faq_question
                    })

            except Exception as e:

                st.error(f"Classification error: {e}")


# ============================================================
# TAB 2: PREDICTION HISTORY AND CSV EXPORT
# ============================================================

with tab2:

    st.subheader("📜 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="📥 Download Prediction History (CSV)",
            data=csv_data,
            file_name="smartdesk_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Prediction History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Classify a customer query to see results here."
        )


# ============================================================
# TAB 3: AI BANKING CHATBOT
# ============================================================

with tab3:

    st.subheader("💬 SmartDesk AI Banking Assistant")

    st.caption(
        "Ask banking questions in English, Hindi or Marathi."
    )

    language = st.selectbox(
        "Preferred response language",
        [
            "Automatic - Detect from message",
            "English",
            "Hindi",
            "Marathi"
        ],
        key="chat_language"
    )

    st.divider()

    # Display previous messages.
    for chat in st.session_state.chat_messages:

        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_message = st.chat_input(
        "Ask your banking question..."
    )

    if user_message:

        try:

            with st.spinner(
                "SmartDesk is preparing a solution..."
            ):

                english_message = translate_to_english(
                    user_message
                )

                intent, confidence, top_k = predict_intent(
                    english_message
                )

                sentiment, sentiment_confidence = (
                    analyze_sentiment(english_message)
                )

                banking_issue = detect_banking_issue(
                    user_message,
                    english_message
                )

                priority = detect_priority(
                    user_message + " " + english_message,
                    intent,
                    banking_issue
                )

                answer = generate_chatbot_response(

                    user_message=user_message,

                    english_message=english_message,

                    intent=intent,

                    confidence=confidence,

                    banking_issue=banking_issue,

                    priority=priority,

                    history=st.session_state.chat_messages,

                    preferred_language=language,

                    review_threshold=REVIEW_THRESHOLD
                )

            # Save conversation after successful response.
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_message
            })

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("user"):
                st.write(user_message)

            with st.chat_message("assistant"):
                st.write(answer)

            with st.expander("🔍 View Banking AI Analysis"):

                st.write("**English translation:**")
                st.write(english_message)

                st.write("**Predicted intent:**")
                st.write(intent)

                st.write(
                    "**Confidence:**",
                    f"{confidence:.2%}"
                )

                if confidence < REVIEW_THRESHOLD:

                    st.warning(
                        "Low confidence: human review recommended."
                    )

                st.write("**Rule-based banking issue:**")
                st.write(banking_issue)

                st.write("**Sentiment:**")
                st.write(sentiment)

                st.write("**Complaint priority:**")
                st.write(priority)

                st.write("**Top predictions:**")

                st.dataframe(
                    pd.DataFrame(top_k),
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                "Unable to generate a response."
            )

            st.exception(e)

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_messages = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartDesk | TinyBERT + Banking77 + Ollama Qwen2.5"
)

st.caption(
    "Sentiment analysis uses DistilBERT. "
    "Priority and additional banking issue detection are rule-based. "
    "Chatbot responses are generated locally through Ollama, "
    "except for predefined fraud-safety responses."
)

st.caption(
    "Educational demonstration only. "
    "Never enter OTPs, PINs, passwords or full card numbers."
)



