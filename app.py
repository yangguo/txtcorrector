import streamlit as st
import operator
import torch
from transformers import BertTokenizer, BertForMaskedLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
model = model.to(device)

# text list encoding
def sent2emb(texts):
    with torch.no_grad():
        outputs = model(**tokenizer(texts, padding=True, return_tensors='pt').to(device))
    return outputs

def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

def corrector(texts):
    outputs = sent2emb(texts)
    result = []
    for ids, text in zip(outputs.logits, texts):
        _text = tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
        corrected_text = _text[:len(text)]
        corrected_text, details = get_errors(corrected_text, text)
        result.append((corrected_text, details))
    return result


def main():

    st.subheader("Chinese Spelling Correction")

    search_text = st.sidebar.text_area('Input Text')
    if search_text != '':
        search_list = search_text.split()
        search_list = list(filter(None, search_list))

    result_list=corrector(search_list)

    # list of search list and result list   
    for search_word, result in zip(search_list, result_list):
        st.write('Input text:'+search_word)
        corrected_text, details = result
        st.write('Corrected text:'+corrected_text)
        st.write('Details:')
        for detail in details:
            st.write(detail)

if __name__ == '__main__':
    main()