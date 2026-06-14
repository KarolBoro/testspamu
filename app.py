import streamlit as st
import pickle


def reczne_czyszczenie(tekst, stop_words):
    oczyszczony = ""
    for znak in str(tekst):
        if znak.isalnum() or znak == ' ':
            oczyszczony += znak.lower()
        else:
            oczyszczony += ' '
    slowa = oczyszczony.split()
    przefiltrowane = [s for s in slowa if s not in stop_words]
    return " ".join(przefiltrowane)


@st.cache_resource
def load_model():
    with open('model_bayes.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    with open('stop_words.pkl', 'rb') as f:
        sw = pickle.load(f)
    return model, vec, sw


model, vectorizer, stop_words = load_model()

st.set_page_config(page_title="Anty-Spam 2.0", page_icon="🛡️")

st.title("🛡️ System Wykrywania Spamu 2.0")
st.subheader("Oparty na autorskim algorytmie Naiwnego Bayesa")

st.write("Wpisz treść wiadomości poniżej, a nasz algorytm sprawdzi, czy jest bezpieczna.")

user_input = st.text_area("Treść wiadomości:", placeholder="Np. Gratulacje! Wygrałeś iPhone, kliknij tutaj...")

if st.button("Sprawdź wiadomość"):
    if user_input.strip() == "":
        st.warning("Proszę wpisać jakąś treść.")
    else:
        cleaned_text = reczne_czyszczenie(user_input, stop_words)

        vectorized_text = vectorizer.transform([cleaned_text]).toarray()

        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.error("🚨 UWAGA: Ta wiadomość została zaklasyfikowana jako SPAM!")
            st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
        else:
            st.success("✅ To jest bezpieczna wiadomość (HAM).")
            st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)

st.divider()
st.caption("Autor: Karol Borowski ")