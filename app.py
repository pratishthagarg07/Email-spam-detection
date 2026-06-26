import streamlit as st
import pickle
import string

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ── Load model, vectorizer, threshold ────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl",     "rb") as f: model     = pickle.load(f)
    with open("tfidf.pkl",     "rb") as f: tfidf     = pickle.load(f)
    with open("threshold.pkl", "rb") as f: threshold = pickle.load(f)
    return model, tfidf, threshold

model, tfidf, threshold = load_model()

# ── Stopwords (same as training) ─────────────────────────
stop_words = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','will','would','shall','should','may','might','must','can',
    'could','not','and','but','or','nor','so','yet','both','either','neither',
    'the','a','an','in','on','at','to','for','of','with','by','from','up',
    'about','into','through','during','before','after','above','below','between',
    'out','off','over','under','again','then','once','here','there','when',
    'where','why','how','all','each','every','more','most','other','some',
    'such','no','nor','only','own','same','than','too','very','just','because',
    's','t','don','won','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ll','m','mightn','mustn','needn','re','shan','shouldn',
    've','wasn','weren','wouldn'
}

# ── Stemmer (same as training) ───────────────────────────
def simple_stem(word):
    suffixes = ['ing','tion','ness','ment','able','ible','er','ed','ly','es','s']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

# ── Transform function (same as training) ────────────────
def transform_text(text):
    text   = text.lower()
    text   = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [simple_stem(t) for t in tokens]
    return ' '.join(tokens)

# ── UI ───────────────────────────────────────────────────
st.title("📧 Email Spam Detector")
st.markdown("Enter an email message below to check if it is **spam** or **ham**.")

st.divider()

input_msg = st.text_area(
    "✉️ Paste your email here",
    height=200,
    placeholder="Type or paste an email message..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    check = st.button("🔍 Check Email", use_container_width=True)

if check:
    if input_msg.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # Preprocess & predict
        transformed = transform_text(input_msg)
        vectorized  = tfidf.transform([transformed]).toarray()
        probability = model.predict_proba(vectorized)[0][1]
        prediction  = int(probability >= threshold)

        st.divider()

        # Result
        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (Not Spam)")

        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Spam Probability",  f"{probability*100:.1f}%")
        col2.metric("Threshold",         f"{threshold:.3f}")
        col3.metric("Word Count",        len(transformed.split()))

        # Confidence bar
        st.markdown("### Confidence")
        st.progress(float(probability))

        # Preprocessing details
        with st.expander("🔎 See preprocessing details"):
            st.markdown("**Original message:**")
            st.info(input_msg)
            st.markdown("**After preprocessing:**")
            st.code(transformed)

st.divider()
st.caption("Built with Naive Bayes + TF-IDF | Accuracy: 99% | Precision: 97.5% | Recall: 90.8%")