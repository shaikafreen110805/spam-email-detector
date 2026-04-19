import streamlit as st
import pickle

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# UI config
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Email Detector")
st.markdown("### Smart ML-based Spam Classifier")

st.write("Enter a message below to check whether it is spam or not.")

user_input = st.text_area("✍️ Your Message")

if st.button("🔍 Analyze"):
    if user_input.strip() != "":
        input_data = vectorizer.transform([user_input])
        prob = model.predict_proba(input_data)[0]

        spam_prob = prob[1]
        ham_prob = prob[0]

        st.subheader("📊 Confidence Scores")
        st.progress(float(spam_prob))

        col1, col2 = st.columns(2)
        col1.metric("Spam Probability", f"{spam_prob:.2f}")
        col2.metric("Ham Probability", f"{ham_prob:.2f}")

        # Decision threshold
        if spam_prob > 0.4:
            st.error("🚨 This message is likely SPAM!")
        else:
            st.success("✅ This message seems safe")

    else:
        st.warning("⚠️ Please enter a message")