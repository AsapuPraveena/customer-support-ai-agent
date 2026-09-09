
import joblib

model = joblib.load("customer_support_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_intent(message):
    message_vector = vectorizer.transform([message])
    return model.predict(message_vector)[0]

def generate_reply(intent):
    replies = {
        "technical_issue": "Sorry you're facing a technical issue. Please describe the problem or share your account details so we can assist you.",
        "refund_billing": "Sorry about the refund issue. Please share your order or transaction details so we can check the refund status.",
        "delivery_order": "We understand your concern. Please share your order number so we can check the delivery status.",
        "account_access": "Sorry you're having trouble accessing your account. Please try resetting your password.",
        "payment_billing": "We can help with your payment issue. Please share your transaction details so we can check it.",
        "cancellation": "We can help with your cancellation request. Please share your order details so we can check the cancellation status.",
        "other": "Thanks for contacting us. Please provide more details about your issue so we can assist you."
    }
    return replies.get(intent, replies["other"])

def customer_support_ai(message):
    intent = predict_intent(message)
    reply = generate_reply(intent)
    return intent, reply
