import streamlit as st
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import TextPreprocessor

# Page config
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="😊",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        preprocessor = TextPreprocessor()
        return model, vectorizer, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, vectorizer, preprocessor = load_model()

# Title and description
st.title("😊 Sentiment Analysis System")
st.markdown("### Analyze the sentiment of your text in real-time!")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This system uses Machine Learning to classify text into:\n\n"
        "- 😊 **Positive**\n"
        "- 😐 **Neutral**\n"
        "- 😞 **Negative**\n\n"
        "Built with scikit-learn and NLTK"
    )
    
    st.header("Model Info")
    st.success("Model: Logistic Regression\nAccuracy: ~95%")
    
    st.header("Examples")
    st.markdown(
        "**Positive**: *This is amazing!*\n\n"
        "**Negative**: *Terrible experience*\n\n"
        "**Neutral**: *It's okay*"
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Your Text")
    
    # Text input
    text_input = st.text_area(
        "Type or paste your text here:",
        height=150,
        placeholder="Enter a review, comment, or any text to analyze..."
    )
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.subheader("Quick Examples")
    
    example1 = st.button("😊 Positive Example", use_container_width=True)
    example2 = st.button("😞 Negative Example", use_container_width=True)
    example3 = st.button("😐 Neutral Example", use_container_width=True)
    
    if example1:
        text_input = "This product is absolutely amazing! I love it so much. Best purchase ever!"
        analyze_button = True
    elif example2:
        text_input = "Terrible product. Very disappointed. Complete waste of money."
        analyze_button = True
    elif example3:
        text_input = "It's okay. Nothing special but does the job. Average quality."
        analyze_button = True

# Analysis
if analyze_button and text_input:
    if model is None:
        st.error("Model not loaded. Please train the model first by running: python main.py")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess
            cleaned_text = preprocessor.clean_text(text_input)
            
            if len(cleaned_text) == 0:
                st.warning("⚠️ Text is too short or contains no meaningful words after preprocessing.")
            else:
                # Predict
                text_vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vectorized)[0]
                
                # Get confidence
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(text_vectorized)[0]
                    confidence = max(probabilities) * 100
                else:
                    confidence = None
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Sentiment display
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if prediction == 'positive':
                        st.success("### 😊 POSITIVE")
                    else:
                        st.info("### 😊 Positive")
                
                with col_b:
                    if prediction == 'neutral':
                        st.warning("### 😐 NEUTRAL")
                    else:
                        st.info("### 😐 Neutral")
                
                with col_c:
                    if prediction == 'negative':
                        st.error("### 😞 NEGATIVE")
                    else:
                        st.info("### 😞 Negative")
                
                # Confidence
                if confidence:
                    st.markdown("---")
                    st.subheader("Confidence Score")
                    st.progress(confidence / 100)
                    st.write(f"**{confidence:.2f}%** confident in this prediction")
                
                # Details
                with st.expander("📝 View Details"):
                    st.write("**Original Text:**")
                    st.write(text_input)
                    st.write("**Cleaned Text:**")
                    st.write(cleaned_text)
                    st.write("**Prediction:**")
                    st.write(prediction.upper())

elif analyze_button:
    st.warning("⚠️ Please enter some text to analyze.")

# Batch analysis
st.markdown("---")
st.subheader("📦 Batch Analysis")

with st.expander("Analyze Multiple Texts"):
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        height=150,
        placeholder="Text 1\nText 2\nText 3"
    )
    
    batch_button = st.button("Analyze All", type="secondary")
    
    if batch_button and batch_input:
        texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
        
        if model is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            results = []
            for text in texts:
                cleaned = preprocessor.clean_text(text)
                if len(cleaned) > 0:
                    vec = vectorizer.transform([cleaned])
                    pred = model.predict(vec)[0]
                    results.append({'Text': text[:50] + '...', 'Sentiment': pred.upper()})
                else:
                    results.append({'Text': text[:50] + '...', 'Sentiment': 'UNKNOWN'})
            
            st.dataframe(results, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Built with ❤️ using Streamlit and scikit-learn</p>"
    "</div>",
    unsafe_allow_html=True
)
