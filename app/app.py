# app.py
import re
import streamlit as st
import pandas as pd

# ------------------------
# Tokenizer Class
# ------------------------
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="Custom Tokenizer", page_icon="coin.png", layout="wide")

st.title("📝 Custom Tokenizer App")
st.markdown("Upload a **raw text file**, and this app will tokenize it, build a vocabulary, and let you encode/decode text.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a text file", type=["txt"])

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")

    # ------------------------
    # Preprocessing & Vocabulary
    # ------------------------
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_tokens = sorted(set(preprocessed))
    # Add special tokens
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    # ------------------------
    # Layout
    # ------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 Original Text")
        st.text_area("Raw text:", raw_text, height=250)

        with st.expander("📑 Vocabulary (full)"):
            df_vocab = pd.DataFrame(vocab.items(), columns=["Token", "ID"])
            st.dataframe(df_vocab, use_container_width=True, height=400)

    with col2:
        st.subheader("🔡 Tokenized Data")
        st.text(f"Total tokens: {len(preprocessed)}")
        st.text(f"Vocabulary size: {len(vocab)}")

        # Show all tokenized words
        with st.expander("📜 View All Tokenized Words"):
            df_tokens = pd.DataFrame(preprocessed, columns=["Tokenized Words"])
            st.dataframe(df_tokens, use_container_width=True, height=400)

        # Tokenizer
        tokenizer = SimpleTokenizerV2(vocab)

        st.subheader("🎯 Try Encoding & Decoding")
        user_input = st.text_area("Enter text:", "Hello, do you like tea?")

        if st.button("Encode"):
            ids = tokenizer.encode(user_input)
            st.write("✅ Encoded IDs:", ids)

        if st.button("Decode"):
            ids = tokenizer.encode(user_input)
            decoded = tokenizer.decode(ids)
            st.write("✅ Decoded text:", decoded)

        # Search functionality
        st.subheader("🔍 Search Token ID")
        search_word = st.text_input("Enter a word/token to search:")
        if search_word:
            token_id = vocab.get(search_word, "Not found")
            st.write(f"Result: `{search_word}` → ID = **{token_id}**")

else:
    st.info("👆 Please upload a `.txt` file to get started.")
