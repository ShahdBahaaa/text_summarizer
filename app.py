import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("./flan_t5_finetuned")
    model = T5ForConditionalGeneration.from_pretrained("./flan_t5_finetuned")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Text Summarization using FLAN-T5")
st.write("Enter a long text and get a concise summary.")

input_text = st.text_area("Input Text", height=200)

if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            "summarize: " + input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=120,
                num_beams=4
            )

        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("Summary")
        st.write(summary)
