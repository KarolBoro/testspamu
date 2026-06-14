import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

print("1. Wczytywanie i czyszczenie danych")
df = pd.read_csv("C:\\Users\\karol\\Downloads\\email.csv")

df = df.dropna(subset=['Category', 'Message'])
df = df[df['Category'].isin(['ham', 'spam'])]
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)


moje_stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
                   'she', 'her', 'it', 'they', 'them', 'what', 'which', 'who', 'this',
                   'that', 'am', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
                   'do', 'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                   'because', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'to',
                   'from', 'in', 'out', 'on', 'off'}


def reczne_czyszczenie(tekst):
    oczyszczony = ""
    for znak in str(tekst):
        if znak.isalnum() or znak == ' ':
            oczyszczony += znak.lower()
        else:
            oczyszczony += ' '

    slowa = oczyszczony.split()
    przefiltrowane = [s for s in slowa if s not in moje_stop_words]
    return " ".join(przefiltrowane)


df['Message'] = df['Message'].apply(reczne_czyszczenie)


print("2. Generowanie wykresow danych...")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Category', hue='Category', palette={'ham': '#4C72B0', 'spam': '#DD8452'}, legend=False)
plt.title('Rozkład klas w zbiorze danych', fontsize=14)
plt.xlabel('Kategoria', fontsize=12)
plt.ylabel('Liczba wiadomości', fontsize=12)
plt.savefig('class_distribution.png', bbox_inches='tight')
plt.close()

spam_text = " ".join(df[df['Category'] == 'spam']['Message'])
wordcloud_spam = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud_spam.png', bbox_inches='tight')
plt.close()

ham_text = " ".join(df[df['Category'] == 'ham']['Message'])
wordcloud_ham = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(ham_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud_ham.png', bbox_inches='tight')
plt.close()


print("3. Przygotowywanie wektoryzacji")
X = df['Message']
y = df['Spam'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorizer_count = CountVectorizer()
X_train_count = vectorizer_count.fit_transform(X_train).toarray()
X_test_count = vectorizer_count.transform(X_test).toarray()

vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer_tfidf.transform(X_test).toarray()



class  WlasnyNaiwnyBayes:
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


class regresja:
    def __init__(self, learning_rate=0.1, epoki=1000):
        self.learning_rate = learning_rate
        self.epoki = epoki

    def funkcja_sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        liczba_maili, liczba_slow = X.shape
        self.wagi = np.zeros(liczba_slow)
        self.bias = 0

        for epoka in range(self.epoki):
            model_liniowy = np.dot(X, self.wagi) + self.bias
            przewidywania = self.funkcja_sigmoid(model_liniowy)
            blad = przewidywania - y
            korekta_wag = (1 / liczba_maili) * np.dot(X.T, blad)
            korekta_biasu = (1 / liczba_maili) * np.sum(blad)
            self.wagi -= (self.learning_rate * korekta_wag)
            self.bias -= (self.learning_rate * korekta_biasu)

    def predict(self, X):
        model_liniowy = np.dot(X, self.wagi) + self.bias
        przewidywania = self.funkcja_sigmoid(model_liniowy)
        return np.array([1 if p > 0.5 else 0 for p in przewidywania])


print("4. Trenowanie modeli")
bayes_count =  WlasnyNaiwnyBayes()
bayes_count.fit(X_train_count, y_train)
pred_bayes_count = bayes_count.predict(X_test_count)

bayes_tfidf =  WlasnyNaiwnyBayes()
bayes_tfidf.fit(X_train_tfidf, y_train)
pred_bayes_tfidf = bayes_tfidf.predict(X_test_tfidf)

regresja_count = regresja(learning_rate=0.1, epoki=1000)
regresja_count.fit(X_train_count, y_train)
pred_regresja_count = regresja_count.predict(X_test_count)

regresja_tfidf = regresja(learning_rate=0.1, epoki=1000)
regresja_tfidf.fit(X_train_tfidf, y_train)
pred_regresja_tfidf = regresja_tfidf.predict(X_test_tfidf)


print("5. Generowanie wykresów z wynikami")
wyniki = [
    accuracy_score(y_test, pred_bayes_count),
    accuracy_score(y_test, pred_bayes_tfidf),
    accuracy_score(y_test, pred_regresja_count),
    accuracy_score(y_test, pred_regresja_tfidf)
]

plt.figure(figsize=(10, 6))
nazwy_modeli = ['Bayes\n(Count)', 'Bayes\n(TF-IDF)', 'Regresja\n(Count)', 'Regresja\n(TF-IDF)']
kolory = ['#4C72B0', '#739EDB', '#55A868', '#7BC78C']
plt.bar(nazwy_modeli, wyniki, color=kolory)
plt.ylim(0.80, 1.0)
plt.ylabel('Trafność (Accuracy)')
plt.title('Porównanie skuteczności (Accuracy)')
for i, v in enumerate(wyniki):
    plt.text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold')
plt.savefig('wykres1_accuracy.png', bbox_inches='tight')
plt.close()


def narysuj_macierz(os, macierz, tytul, kolor):
    sns.heatmap(macierz, annot=True, fmt='d', cmap=kolor, ax=os, annot_kws={"size": 15}, cbar=False)
    os.set_title(tytul, fontsize=14, pad=10)
    os.set_xlabel('Przewidziane przez model', fontsize=11)
    os.set_ylabel('Prawdziwa kategoria', fontsize=11)
    os.set_xticklabels(['Ham (0)', 'Spam (1)'])
    os.set_yticklabels(['Ham (0)', 'Spam (1)'])


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
narysuj_macierz(ax[0], confusion_matrix(y_test, pred_bayes_count), 'Bayes - Zwykłe zliczanie', 'Blues')
narysuj_macierz(ax[1], confusion_matrix(y_test, pred_bayes_tfidf), 'Bayes - TF-IDF', 'Blues')
plt.savefig('macierze_bayes.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
narysuj_macierz(ax[0], confusion_matrix(y_test, pred_regresja_count), 'Regresja - Zwykłe zliczanie', 'Greens')
narysuj_macierz(ax[1], confusion_matrix(y_test, pred_regresja_tfidf), 'Regresja - TF-IDF', 'Greens')
plt.savefig('macierze_regresja.png', bbox_inches='tight')
plt.close()

lr_eksperyment = 1.0

reg_1k = regresja(learning_rate=lr_eksperyment, epoki=1000)
reg_1k.fit(X_train_tfidf, y_train)
cm_1k = confusion_matrix(y_test, reg_1k.predict(X_test_tfidf))

reg_3k = regresja(learning_rate=lr_eksperyment, epoki=3000)
reg_3k.fit(X_train_tfidf, y_train)
cm_3k = confusion_matrix(y_test, reg_3k.predict(X_test_tfidf))

reg_5k = regresja(learning_rate=lr_eksperyment, epoki=5000)
reg_5k.fit(X_train_tfidf, y_train)
cm_5k = confusion_matrix(y_test, reg_5k.predict(X_test_tfidf))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
narysuj_macierz(ax[0], cm_1k, '1000 epok (Niedouczenie)', 'Reds')
narysuj_macierz(ax[1], cm_3k, '3000 epok (Widoczna poprawa)', 'Oranges')
narysuj_macierz(ax[2], cm_5k, '5000 epok (Optymalna wydajność)', 'Greens')
plt.savefig('ewolucja_epok.png', bbox_inches='tight')
plt.close()
print("Wygenerowano plik ewolucja_epok.png")

print("\nGOTOWE!")

with open('model_bayes.pkl', 'wb') as f:
    pickle.dump(bayes_count, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer_count, f)

with open('stop_words.pkl', 'wb') as f:
    pickle.dump(moje_stop_words, f)

print("Bayers gotowy i zapisany!")