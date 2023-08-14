import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # Pre-trained BERT model for QA tasks
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def get_answer(question, context):
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    model.to(encoding["input_ids"].device)
    model.eval()
    
    with torch.no_grad():
        output = model(**encoding)

        start_logits = output.start_logits
        end_logits = output.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)
    
    if start_idx < end_idx and start_idx >= 0 and end_idx < len(encoding["input_ids"][0]):
        all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        answer_tokens = all_tokens[start_idx: end_idx + 1]
        answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
        return answer
    else:
        return "Out of context"


def main():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        file_contents = uploaded_file.read().decode("utf-8")
        legal_document = file_contents
        st.title("Legal Chatbot")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Please ask a question"):
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})
            process_input(prompt,legal_document)

def process_input(user_input,legal_document):

    user_input = str(user_input)
    if user_input.split() == []:
        m = "Legal Chatbot: Please ask a valid question"

    if user_input.lower() == "exit":
        m = "Legal Chatbot: Goodbye!"
        
    answer = get_answer(user_input, legal_document)

    if answer == "Out of context":
        m = "Legal Chatbot: Out of context"
    else:
        m = "Legal Chatbot: " + answer

    res = m
    with st.chat_message("assistant"):
        st.markdown(res)

    st.session_state.messages.append({"role": "assistant", "content": res})

if __name__ == "__main__":
    main()