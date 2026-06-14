import streamlit as st
import pickle
import numpy as np
import csv
import os


class WlasnyNaiwnyBayes:
    def fit(self, X, y):
        liczba_probek, liczba_cech = X.shape
        self.klasy = np.unique(y)
        self.priors = np.zeros(len(self.klasy))
        self.zliczenia_slow = np.zeros((len(self.klasy), liczba_cech))
        self.suma_slow_klasy = np.zeros(len(self.klasy))

        for klasa in self.klasy:
            X_klasy = X[y == klasa]
            self.priors[klasa] = X_klasy.shape[0] / float(liczba_probek)
            self.zliczenia_slow[klasa, :] = np.sum(X_klasy, axis=0) + 1
            self.suma_slow_klasy[klasa] = np.sum(self.zliczenia_slow[klasa, :])

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriory = []
        for klasa in self.klasy:
            prior = np.log(self.priors[klasa])
            prawd_slow = np.log(self.zliczenia_slow[klasa, :] / self.suma_slow_klasy[klasa])
            posterior = prior + np.sum(prawd_slow * x)
            posteriory.append(posterior)
        return self.klasy[np.argmax(posteriory)]

    def predict_proba(self, X):
        return np.array([self._predict_proba_single(x) for x in X])

    def _predict_proba_single(self, x):
        posteriory = []
        for klasa in self.klasy:
            prior = np.log(self.priors[klasa])
            prawd_slow = np.log(self.zliczenia_slow[klasa, :] / self.suma_slow_klasy[klasa])
            posterior = prior + np.sum(prawd_slow * x)
            posteriory.append(posterior)

        posteriory = np.array(posteriory)
        max_post = np.max(posteriory)
        exp_post = np.exp(posteriory - max_post)

        return exp_post / np.sum(exp_post)

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


def zapisz_do_bazy(tekst, prawdziwa_kategoria):
    sciezka_csv = "C:\\Users\\karol\\Downloads\\email.csv"
    with open(sciezka_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([prawdziwa_kategoria, tekst])


st.set_page_config(page_title="Anty-Spam 2.0", page_icon="🛡️")

st.title("🛡️ System Wykrywania Spamu 2.0")
st.subheader("Oparty na autorskim algorytmie Naiwnego Bayesa")

if 'sprawdzone' not in st.session_state:
    st.session_state.sprawdzone = False
    st.session_state.ostatni_tekst = ""
    st.session_state.ostatnia_predykcja = None
    st.session_state.ostatnia_pewnosc = 0.0
    st.session_state.opinia_zapisana = False

user_input = st.text_area("Treść wiadomości:", placeholder="Np. Gratulacje! Wygrałeś iPhone...")

if st.button("Sprawdź wiadomość"):
    if user_input.strip() == "":
        st.warning("Proszę wpisać jakąś treść.")
    else:
        cleaned_text = reczne_czyszczenie(user_input, stop_words)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()

        # POBIERAMY WERDYKT I PROCENTY
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = probabilities[prediction] * 100  # Wyciągamy procent dla wygranej klasy

        st.session_state.sprawdzone = True
        st.session_state.ostatni_tekst = user_input
        st.session_state.ostatnia_predykcja = prediction
        st.session_state.ostatnia_pewnosc = confidence
        st.session_state.opinia_zapisana = False

if st.session_state.sprawdzone:
    st.divider()

    # Wyświetlanie wyniku z uwzględnieniem PEWNOŚCI MODELU
    if st.session_state.ostatnia_predykcja == 1:
        st.error(
            f"🚨 UWAGA: Ta wiadomość została zaklasyfikowana jako SPAM! (Pewność: {st.session_state.ostatnia_pewnosc:.2f}%)")
        # Pasek postępu (czerwony/pomarańczowy wizualnie w Streamlicie przy błędzie)
        st.progress(st.session_state.ostatnia_pewnosc / 100)
    else:
        st.success(f"✅ To jest bezpieczna wiadomość (HAM). (Pewność: {st.session_state.ostatnia_pewnosc:.2f}%)")
        st.progress(st.session_state.ostatnia_pewnosc / 100)

    # System opinii
    if not st.session_state.opinia_zapisana:
        st.write("**Czy algorytm ocenił wiadomość poprawnie?**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Tak, ocena jest trafna"):
                kategoria = 'spam' if st.session_state.ostatnia_predykcja == 1 else 'ham'
                zapisz_do_bazy(st.session_state.ostatni_tekst, kategoria)
                st.session_state.opinia_zapisana = True
                st.rerun()

        with col2:
            if st.button("👎 Nie, to pomyłka algorytmu"):
                kategoria = 'ham' if st.session_state.ostatnia_predykcja == 1 else 'spam'
                zapisz_do_bazy(st.session_state.ostatni_tekst, kategoria)
                st.session_state.opinia_zapisana = True
                st.rerun()
    else:
        st.info("Dziękujemy! Twoja wiadomość została dopisana do bazy danych.")

st.divider()
st.caption("Autorzy: Karol Borowski")