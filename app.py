import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Phishing Email Detector", layout="wide")
st.title("Phishing Email Detection System")

# 1. Load dataset & combine subject + body
df = pd.read_csv("emails_dataset.csv")
df["full_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# 2. Convert text into numbers using TF-IDF (ignoring common English stop words)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["full_text"])
y = df["label"]

# 3. Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Top KPI Row
c1, c2, c3 = st.columns(3)
c1.metric("Training Emails Loaded", len(df))
c2.metric("Phishing Samples", int((df["label"] == "phishing").sum()))
c3.metric("Legitimate Samples", int((df["label"] == "legitimate").sum()))

st.divider()

# 4. Interactive Email Scanner Interface
st.subheader("Live Email Security Analyzer")
col_left, col_right = st.columns([1.2, 1])

with col_left:
    sender = st.text_input("Sender Email Address", "support@paypa1-security.com")
    subject = st.text_input("Email Subject", "URGENT: Your account has been limited")
    body = st.text_area(
        "Email Message Body",
        "Dear user, we noticed suspicious activity. Click http://198.51.100.12/verify immediately to confirm your password.",
        height=130
    )
    scan_btn = st.button("Scan Email for Phishing", type="primary")

with col_right:
    if scan_btn:
        combined_text = f"{subject} {body}"
        email_vector = vectorizer.transform([combined_text])
        
        prediction = model.predict(email_vector)[0]
        confidence = model.predict_proba(email_vector).max() * 100

        # Heuristic / Rule-based checks
        urls = re.findall(r"https?://[^\s]+", body)
        ip_urls = [u for u in urls if re.search(r"https?://\d+\.\d+\.\d+\.\d+", u)]
        urgent_words = [
            w for w in ["urgent", "verify", "suspended", "immediately", "password", "blocked", "kyc", "click"]
            if w in combined_text.lower()
        ]
        domain = sender.split("@")[-1] if "@" in sender else ""
        spoofed_domain = any(char.isdigit() for char in domain)

        # Final Decision
        if prediction == "phishing" or ip_urls or spoofed_domain:
            st.error(f"VERDICT: POTENTIAL PHISHING DETECTED (AI Confidence: {confidence:.1f}%)")
        else:
            st.success(f"VERDICT: LEGITIMATE EMAIL (AI Confidence: {confidence:.1f}%)")

        st.markdown("**Technical Indicators Breakdown:**")
        st.write(f"- **Extracted URLs:** `{urls if urls else 'None'}`")
        
        red_flags = []
        if ip_urls:
            red_flags.append("Raw IP address used instead of a domain name in URL")
        if spoofed_domain:
            red_flags.append(f"Suspicious numeric character in sender domain (`{domain}`)")
        if urgent_words:
            red_flags.append(f"Social engineering urgency keywords: `{', '.join(urgent_words)}`")

        if red_flags:
            for flag in red_flags:
                st.warning(flag)
        else:
            st.info("No heuristic red flags detected in sender or links.")