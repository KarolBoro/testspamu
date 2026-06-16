import streamlit as st
import pickle
import numpy as np
import csv
import pandas as pd
import os
import io

st.set_page_config(page_title="Anty-Spam", page_icon="🛡️", layout="wide")


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
sciezka_csv = "C:\\Users\\karol\\PycharmProjects\\testspamu\\email_polski.csv"


def zapisz_do_bazy(tekst, prawdziwa_kategoria):
    czysty_tekst = str(tekst).replace('\n', ' ').replace('\r', '')
    with open(sciezka_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([prawdziwa_kategoria, czysty_tekst])



def przetrenuj_model():
    try:
        df = pd.read_csv(sciezka_csv, sep=',', on_bad_lines='skip')
        df = df.dropna(subset=['Category', 'Message'])
        df['Category'] = df['Category'].str.lower().str.strip()
        df = df[df['Category'].isin(['ham', 'spam'])]
        df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

        czyste_teksty = df['Message'].astype(str).apply(lambda x: reczne_czyszczenie(x, stop_words))
        X_nowe = vectorizer.fit_transform(czyste_teksty).toarray()
        y_nowe = df['Spam'].values

        nowy_model = WlasnyNaiwnyBayes()
        nowy_model.fit(X_nowe, y_nowe)

        with open('model_bayes.pkl', 'wb') as f:
            pickle.dump(nowy_model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        st.cache_resource.clear()
        return True, len(df)
    except Exception as e:
        return False, str(e)


with st.sidebar:
    st.header("📊 Statystyki bazy danych")

    if os.path.exists(sciezka_csv):
        try:
            df_stats = pd.read_csv(sciezka_csv, sep=',', on_bad_lines='skip')
            df_stats = df_stats.dropna(subset=['Category'])
            df_stats['Category'] = df_stats['Category'].str.lower().str.strip()

            razem = len(df_stats)
            ham_count = len(df_stats[df_stats['Category'] == 'ham'])
            spam_count = len(df_stats[df_stats['Category'] == 'spam'])

            st.metric("Wszystkie wiadomości", razem)
            col_b1, col_b2 = st.columns(2)
            col_b1.metric("HAM", ham_count)
            col_b2.metric("SPAM", spam_count)

            st.divider()
            st.subheader("📈 Rozkład procentowy")
            if razem > 0:
                procent_spam = (spam_count / razem) * 100
                st.write(f"Udział SPAMu w bazie: **{procent_spam:.1f}%**")
                st.progress(procent_spam / 100)

        except Exception as e:
            st.error("Nie udało się załadować statystyk.")
    else:
        st.warning("Nie znaleziono pliku bazy danych email_polski.csv")


    st.divider()
    with st.expander("⚙️ Panel Administratora"):
        st.write("Wymuś aktualizację wagi słów w modelu na podstawie najnowszych wpisów z bazy danych.")
        if st.button("🧠 Przetrenuj model"):
            with st.spinner("Trwa analiza i przebudowa macierzy..."):
                sukces, info = przetrenuj_model()
                if sukces:
                    st.success(f"Sukces! Model odświeżony i wyuczony na {info} wiadomościach.")
                else:
                    st.error(f"Błąd trenowania: {info}")

st.title("🛡️ System Wykrywania Spamu")
st.subheader(
    "Oparty na autorskim algorytmie Naiwnego Bayesa, Algorytm został domyślnie wyszkolony na danych z serwisu Kaggle. Natomiast stwierdziłem, że lepszym pomysłem jest baza danych w języku polskim, więc musiałem wygenerować takową. Jest ona zamieszczona również na Githubie ma ona w sobie ponad 50 000 insertów. ")

tab1, tab2 = st.tabs(["✉️ Sprawdź pojedynczą wiadomość", "📁 Przetwarzanie masowe (Pliki CSV)"])


with tab1:
    if 'sprawdzone' not in st.session_state:
        st.session_state.sprawdzone = False
        st.session_state.ostatni_tekst = ""
        st.session_state.ostatnia_predykcja = None
        st.session_state.ostatnia_pewnosc = 0.0
        st.session_state.opinia_zapisana = False

    user_input = st.text_area("Treść wiadomości:", placeholder="Np. Gratulacje użytkowniku! Wygrałeś darmowego Iphone 13 Pro Max!")

    if st.button("Sprawdź wiadomość"):
        if user_input.strip() == "":
            st.warning("Proszę wpisać jakąś treść.")
        else:
            model, vectorizer, stop_words = load_model()

            cleaned_text = reczne_czyszczenie(user_input, stop_words)
            vectorized_text = vectorizer.transform([cleaned_text]).toarray()

            prediction = model.predict(vectorized_text)[0]
            probabilities = model.predict_proba(vectorized_text)[0]
            confidence = probabilities[prediction] * 100

            st.session_state.sprawdzone = True
            st.session_state.ostatni_tekst = user_input
            st.session_state.ostatnia_predykcja = prediction
            st.session_state.ostatnia_pewnosc = confidence
            st.session_state.opinia_zapisana = False

    if st.session_state.sprawdzone:
        st.divider()

        if st.session_state.ostatnia_predykcja == 1:
            st.error(
                f"🚨 UWAGA: Ta wiadomość została zaklasyfikowana jako SPAM! (Pewność: {st.session_state.ostatnia_pewnosc:.2f}%)")
            st.progress(st.session_state.ostatnia_pewnosc / 100)
        else:
            st.success(f"✅ To jest bezpieczna wiadomość (HAM). (Pewność: {st.session_state.ostatnia_pewnosc:.2f}%)")
            st.progress(st.session_state.ostatnia_pewnosc / 100)

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

with tab2:
    st.write(
        "Wgraj plik `.csv` zawierający kolumnę z tekstami wiadomości w języku polskim. System masowo przeanalizuje plik i doda kolumny z werdyktami.")

    wgrany_plik = st.file_uploader("Wybierz plik CSV do analizy", type=['csv'])

    if wgrany_plik is not None:
        try:
            df_batch = pd.read_csv(wgrany_plik, on_bad_lines='skip')
            kolumna_tekstowa = st.selectbox("Wskaż kolumnę zawierającą treść wiadomości:", df_batch.columns)

            if st.button("🚀 Rozpocznij analizę masową", key="btn_masowa"):
                with st.spinner('Trwa przetwarzanie danych (NLP & Operacje macierzowe)...'):
                    model, vectorizer, stop_words = load_model()

                    df_batch = df_batch.dropna(subset=[kolumna_tekstowa])
                    czyste_teksty = df_batch[kolumna_tekstowa].astype(str).apply(
                        lambda x: reczne_czyszczenie(x, stop_words))
                    macierz_batch = vectorizer.transform(czyste_teksty).toarray()

                    predykcje = model.predict(macierz_batch)
                    prawdopodobienstwa = model.predict_proba(macierz_batch)
                    pewnosci = [prob[pred] * 100 for prob, pred in zip(prawdopodobienstwa, predykcje)]

                    df_batch['Werdykt_AI'] = ['SPAM' if p == 1 else 'HAM' for p in predykcje]
                    df_batch['Pewność_Algorytmu(%)'] = [round(c, 2) for c in pewnosci]

                st.success(f"Ukończono! Pomyślnie przeanalizowano {len(df_batch)} wierszy.")
                st.write("Podgląd pierwszych wyników:")
                st.dataframe(df_batch.head())

                csv_buffer = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Pobierz przeanalizowany plik CSV",
                    data=csv_buffer,
                    file_name='wyniki_analizy_antyspam.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error("Wystąpił błąd podczas odczytu pliku. Upewnij się, że struktura pliku CSV jest poprawna.")

st.divider()
st.caption("Autorzy: Karol Borowski")