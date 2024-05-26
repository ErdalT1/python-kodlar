import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Veri setini yükleme
data = pd.read_csv("C:\\Users\\erdal\\Downloads\\Yeniklasör(2)\\email.csv", sep='\t', header=None, names=['label', 'text'])
data = data.dropna()

# Veri ön işleme fonksiyonu
def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()
    # URL'leri kaldırma
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # E-posta adreslerini kaldırma
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Noktalama işaretlerini kaldırma
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Fazla boşlukları kaldırma
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# Metin verilerini temizleme
data['text'] = data['text'].apply(preprocess_text)
# Metin verileri ve etiketleri ayırma
X = data['text']
y = data['label']
# Öznitelik vektörlerini oluşturma
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)
# Modeli eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)
# Naive Bayes modelini oluşturma ve eğitme
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# Modeli test etme
y_pred = nb_model.predict(X_test)

# Doğruluk değerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Model accuracy:", accuracy)
from sklearn.metrics import classification_report

# Performans raporu
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
from sklearn.metrics import confusion_matrix

# Karışıklık matrisini hesaplama
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)