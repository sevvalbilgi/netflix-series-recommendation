from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Veri setini yükleyelim ve ön işlemleri yapalım
try:
    df = pd.read_csv(r'C:\Users\sbilgi\Desktop\Netflix\data\netflix_titles.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(r'C:\Users\sbilgi\Desktop\Netflix\data\netflix_titles.csv', encoding='latin1')

# Gereksiz sütunları kaldırma
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Başlıkları küçük harfe dönüştürme ve boşlukları temizleme
df['title'] = df['title'].str.lower().str.strip()

# İçerik tabanlı öneri motoru için özelliklerin birleştirilmesi
df['content'] = (
    df['title'].astype(str) + ' ' +
    df['director'].astype(str).fillna('') + ' ' +
    df['cast'].astype(str).fillna('') + ' ' +
    df['listed_in'].astype(str) + ' ' +
    df['description'].astype(str)
)

# TF-IDF matrisinin oluşturulması ve cosine similarity matrisinin hesaplanması
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Dizilerin indekslerini ve başlıklarını içeren bir eşleme oluşturma
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Öneri fonksiyonu
def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.lower().strip()
    if title not in indices:
        print(f"Title '{title}' not found in dataset")
        return ["Title not found"]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    show_indices = [i[0] for i in sim_scores]
    print(f"Recommendations for '{title}': {df['title'].iloc[show_indices].tolist()}")
    return df['title'].iloc[show_indices].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        print("Title parameter is missing")
        return jsonify({"error": "Title parameter is required"}), 400
    title = title.lower().strip()
    print(f"Received title: {title}")
    print(f"All titles: {df['title'].unique()[:10]}")  # İlk 10 başlığı yazdır
    recommendations = get_recommendations(title)
    print(f"Recommendations returned: {recommendations}")
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)  # Hata ayıklama modunu etkinleştirdik
