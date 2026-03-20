from utils.news_fetcher import fetch_news
from utils.relevance_filter import rule_based_filter, strong_keyword_check
from models.relevance_model import train_relevance_model, clean_text
from models.event_model import train_event_model
from utils.risk_engine import calculate_risk

# 🔹 Step 1: Train model
model = train_relevance_model()

# 🔹 Step 2: Fetch news
query = "war OR flood OR strike OR port closure OR supply chain disruption"
df = fetch_news(query)

# 🔹 Step 3: Combine text
df["text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["content"].fillna("")
)

# 🔹 Step 4: Clean text
df["clean_text"] = df["text"].apply(clean_text)

# 🔹 Step 5: Rule-based filtering
df["rule_pass"] = df["clean_text"].apply(rule_based_filter)

df_filtered = df[df["rule_pass"] == True].copy()

# 🔹 Step 6: ML prediction
df_filtered.loc[:, "is_relevant"] = model.predict(df_filtered["clean_text"])

# 🔹 Step 7: Confidence score
probs = model.predict_proba(df_filtered["clean_text"])
df_filtered.loc[:, "confidence"] = probs[:, 1]

# 🔹 Step 8: Strong keyword check
df_filtered.loc[:, "strong_signal"] = df_filtered["clean_text"].apply(strong_keyword_check)

# 🔹 Step 9: FINAL BALANCED FILTER 🔥
final_df = df_filtered[
    (
        (df_filtered["confidence"] > 0.7)  # strong ML signal
    ) |
    (
        (df_filtered["confidence"] > 0.65) & 
        (df_filtered["strong_signal"] == True)  # needs strong keyword support
    )
]

# Remove duplicates
final_df = final_df.drop_duplicates(subset="title")

# 🔹 Output
print("\nFinal Relevant News:\n")
print(final_df[["title", "source", "confidence"]])

# 🔹 Step 10: Train event model
event_model = train_event_model()

# 🔹 Step 11: Predict event type
final_df.loc[:, "event_type"] = event_model.predict(final_df["clean_text"])

def correct_event(text, predicted):
    text = text.lower()

    geo_keywords = ["war", "conflict", "attack", "sanction", "military"]

    if any(word in text for word in geo_keywords):
        return "Geopolitical"

    return predicted

final_df["event_type"] = final_df.apply(
    lambda row: correct_event(row["clean_text"], row["event_type"]),
    axis=1
)

# 🔹 Output
print("\nFinal News with Event Type:\n")
print(final_df[["title", "event_type", "confidence"]])

# 🔹 Step 14: Apply Risk Scoring

risk_results = final_df["event_type"].apply(calculate_risk)

final_df["risk_score"] = risk_results.apply(lambda x: x[0])
final_df["risk_level"] = risk_results.apply(lambda x: x[1])

# 🔹 Final Output
print("\nFinal News with Risk Analysis:\n")
print(final_df[["title", "event_type", "risk_level", "risk_score"]])