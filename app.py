import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import get_comments_from_post, clean_text, analyze_sentiment, get_word_frequency

st.set_page_config(page_title="Instagram Comment Sentiment", layout="wide")

st.title("📊 Instagram Post Comment Sentiment Analyzer")

# Input fields
post_url = st.text_input("🔗 Enter Instagram Post URL:")
username = st.text_input("👤 Instagram Username:")
password = st.text_input("🔑 Instagram Password:", type="password")

if st.button("Analyze"):
    if post_url and username and password:
        with st.spinner("Fetching comments and analyzing..."):
            try:
                comments = get_comments_from_post(post_url, username, password)
                cleaned = [clean_text(c) for c in comments]
                results = [analyze_sentiment(c) for c in cleaned]

                df = pd.DataFrame({
                    "Original Comment": comments,
                    "Cleaned": cleaned,
                    "Sentiment": [r[0] for r in results],
                    "Score": [r[1] for r in results]
                })

                st.success(f"✅ Fetched {len(comments)} comments!")

                # Plot 1: Sentiment Distribution
                st.subheader("📈 Sentiment Distribution")
                fig1, ax1 = plt.subplots()
                sns.countplot(data=df, x="Sentiment", ax=ax1)
                st.pyplot(fig1)

                # Plot 2: Frequent Words
                st.subheader("📝 Most Frequent Words")
                word_counts = get_word_frequency(comments)
                word_df = pd.DataFrame(word_counts, columns=["Word", "Frequency"])

                fig2, ax2 = plt.subplots()
                sns.barplot(data=word_df, x="Frequency", y="Word", ax=ax2)
                st.pyplot(fig2)

                # Optional WordCloud
                st.subheader("☁️ Word Cloud")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df["Cleaned"]))
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis("off")
                st.pyplot(fig_wc)

                # Display data
                with st.expander("🔍 See Comment Analysis Data"):
                    st.dataframe(df)

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please fill in all fields.")
