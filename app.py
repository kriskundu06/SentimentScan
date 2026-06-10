import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

st.set_page_config(
    page_title="Sentiment Analysis Tool based on AI",
    layout = 'wide',
    initial_sidebar_state = "expanded"
)
#sidebar

st.title("🎯 SentimentScan")
st.subheader("Transformer-Powered Customer Sentiment Analysis")

st.markdown("""
Analyze customer reviews using a DistilBERT transformer model.
Get instant sentiment predictions and confidence scores.
""")

#load model
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_model():
    #DistilBERT
    model = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    return model

#load the model and show the spinner while sentiment analysis is being performed
with st.spinner("Loading the sentiment analysis model..."):
    sentiment_pipeline = load_sentiment_model()

#main interface
tab1, tab2 = st.tabs(["Analyze Text", "Analyze CSV"])   

#tab1
with tab1:
    st.header("Analyze single review")
    st.info("""
    Examples:
    • The product quality is amazing and delivery was fast.

    • Worst experience ever, customer support was terrible.
            
    • The product is okay, nothing special.
    """)
    user_input = st.text_area("Please enter the customer review:","The service and the product was just amazing, I just love it")

    if st.button("Analyze Sentiment"):
        if user_input:
            #run the model
            result = sentiment_pipeline(user_input)
            #extract the result and the score
            label = result[0]["label"]
            score = result[0]["score"]
            #display the results
            if label.lower() == "positive":
                st.success("😊 Positive Review")
            elif label.lower() == "negative":
                st.error("😠 Negative Review")
            else:
                st.warning("😐 Neutral Review")

            st.metric(
                label="Confidence Score",
                value=f"{score*100:.2f}%"
            )
        else:
            st.warning("Please enter some text value to analyze the sentiment.")

#tab2
with tab2:
    st.header("Analyze the Dataset(CSV)") 
    uploaded_file = st.file_uploader("Upload your CSV file with customer sentiment", type = ["csv"])
    if uploaded_file is not None:
        #reading the csv file
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset:")
        st.dataframe(df.head())

        #check if 'review' column exists in the dataset upload
        text_column = "review_text"
        if text_column in df.columns:
            if st.button("Analyze the entire dataset"):
                st.write("Analyzing .. this might take few moments, please be patient")    

                #apply the model to the text column to analyze
                df["AI_sentiment"] = df[text_column].apply(lambda x: sentiment_pipeline(str(x))[0]["label"])
                #display the results
                st.write(" ### Analysis Completed!")
                st.dataframe(df[[text_column, "AI_sentiment"]].head(10))

                #show a simple ar graph to display sentiment
                st.write(" ### Sentiment Distribution:")
                sentiment_counts = df["AI_sentiment"].value_counts()
                fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

                #download the result
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label = "Download the sentiment analysis result as CSV",
                    data = csv,
                    file_name = "Sentiment Analysis file.csv",
                    mime = "text/csv"
                )

            else:
                st.error(f"Could not find the column '{text_column}' in the uploaded dataset. Please make sure your CSV file contains a column named '{text_column}' with the text data to analyze.")  

with st.expander("About DistilBERT"):
    st.write("""
    DistilBERT is a lightweight transformer model created
    through knowledge distillation from BERT.

    It retains most of BERT's accuracy while being
    significantly faster and smaller.
    """)
st.markdown("---")
st.markdown(
    """
    <div>
        Made with ❤️ by Krishanu Kundu
    </div>
    """,
    unsafe_allow_html=True
)
