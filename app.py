import streamlit as st
import operator
import torch
from transformers import BertTokenizer, BertForMaskedLM
import streamlit_authenticator as stauth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    "shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
model = model.to(device)


# text list encoding
def sent2emb(texts):
    with torch.no_grad():
        outputs = model(
            **tokenizer(texts, padding=True, return_tensors='pt').to(device))
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
                corrected_text = corrected_text[:i] + ori_char + corrected_text[
                    i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


def corrector(texts):
    outputs = sent2emb(texts)
    result = []
    for ids, text in zip(outputs.logits, texts):
        _text = tokenizer.decode(torch.argmax(ids, dim=-1),
                                 skip_special_tokens=True).replace(' ', '')
        corrected_text = _text[:len(text)]
        corrected_text, details = get_errors(corrected_text, text)
        result.append((corrected_text, details))
    return result


# replace each word in list based on start and end index
def highlight_word(text_list, start, end):
    new_text_list = text_list[start:end]
    upttxt = []
    for txt in new_text_list:
        upttxt.append('<span style="color:red">{}</span>'.format(txt))
    return text_list[:start] + upttxt + text_list[end:]


def main():

    st.subheader("Chinese Spelling Correction")

    search_text = st.text_area('Input Text')

    if st.button('Correct'):
        with st.spinner('Processing...'):
            if search_text != '':
                search_list = search_text.split()
                search_list = list(filter(None, search_list))

                result_list = corrector(search_list)

                # list of search list and result list
                for search_word, result in zip(search_list, result_list):

                    corrected_text, details = result
                    # if details is not empty, show details
                    if details:
                        txt_hl = list(search_word)
                        corrected_hl = list(corrected_text)
                        for detail in details:
                            # highlight word in text
                            txt_hl = highlight_word(txt_hl, detail[2], detail[3])
                            corrected_hl = highlight_word(corrected_hl, detail[2],
                                                        detail[3])

                        txt_hlstr = ''.join(txt_hl)
                        corrected_hlstr = ''.join(corrected_hl)
                        st.markdown('Input text: ' + txt_hlstr,
                                    unsafe_allow_html=True)
                        st.markdown('Corrected text: ' + corrected_hlstr,
                                    unsafe_allow_html=True)
                    else:
                        # if details is empty, no error found
                        st.markdown('NO ERRORS FOUND')

if __name__ == '__main__':
    names = ['John Smith','Rebecca Briggs']
    usernames = ['jsmith','rbriggs']
    passwords = ['123','456']
    hashed_passwords = stauth.hasher(passwords).generate()
    authenticator = stauth.authenticate(names,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=3)

    name, authentication_status = authenticator.login('Login','sidebar')
    if authentication_status:
        main()
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')