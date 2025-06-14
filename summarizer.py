import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
import os
import PyPDF2

# Download NLTK resource (hanya sekali)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# Mapping POS tag ke WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

# Tahap 1: Preprocessing dan Document Indexing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    processed = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged  = pos_tag(tokens)
        words   = [
            lemmatizer.lemmatize(tok.lower(), get_wordnet_pos(pos))
            for tok, pos in tagged
            if tok.isalnum() and tok.lower() not in stop_words
        ]
        processed.append(words)
    return sentences, processed

# Tahap 2: Membangun Indeks Term
# Mengumpulkan daftar term unik dari semua kalimat
def build_term_index(processed_sentences):
    terms = set()
    for words in processed_sentences:
        terms.update(words)
    return sorted(list(terms))

# Tahap 3: Term Weighting (TF)
def compute_tf(processed_sentences, term_index):
    tf = np.zeros((len(processed_sentences), len(term_index)))
    for i, words in enumerate(processed_sentences):
        for word in words:
            j = term_index.index(word)
            tf[i, j] += 1
    return tf

# Tahap 4: Log-Frequency Weighting
def compute_log_frequency_weighting(tf):
    log_tf = np.zeros_like(tf)
    mask = tf > 0
    log_tf[mask] = 1 + np.log10(tf[mask])
    return log_tf

# Tahap 5: Document Frequency (DF)
def compute_df(tf):
    df = np.count_nonzero(tf > 0, axis=0)
    return df

# Tahap 6: Inverse Document Frequency (IDF)
def compute_idf(df, N):
    idf = np.log10(N / df)
    return idf

# Tahap 7: TF-IDF
def compute_tfidf(log_tf, idf):
    tfidf = log_tf * idf
    return tfidf

# Tahap 8: Cosine Similarity antar Kalimat
def compute_cosine_similarity(tfidf_matrix):
    num_sentences = tfidf_matrix.shape[0]
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                dot_product = np.dot(tfidf_matrix[i], tfidf_matrix[j])
                norm_i = np.linalg.norm(tfidf_matrix[i])
                norm_j = np.linalg.norm(tfidf_matrix[j])
                cosine_sim = dot_product / (norm_i * norm_j) if norm_i != 0 and norm_j != 0 else 0
                similarity_matrix[i, j] = cosine_sim

    return similarity_matrix

# Tahap 9: Ranking dan Generate Summary
def rank_sentences_by_similarity(sentences, tfidf, top_n):
    similarity_matrix = compute_cosine_similarity(tfidf)
    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_indices = np.argsort(sentence_scores)[::-1]
    top_indices = sorted(ranked_indices[:top_n])
    summary = ' '.join([sentences[i] for i in top_indices])
    return summary

# Membaca file PDF
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Membaca file TXT
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Fungsi Summary utama
def summaryFunction(text, top_n=None):
    sentences, processed_sentences = preprocess_text(text)
    N = len(sentences)
    if N == 0:
        return "(No valid content found.)"
    if top_n is None:
        top_n = int(N * 0.3)

    term_index = build_term_index(processed_sentences)
    tf = compute_tf(processed_sentences, term_index)
    log_tf = compute_log_frequency_weighting(tf)
    df = compute_df(tf)
    idf = compute_idf(df, N)
    tfidf = compute_tfidf(log_tf, idf)
    summary = rank_sentences_by_similarity(sentences, tfidf, top_n)

    return summary

# Fungsi Pembacaan dan Ringkasan file
def summarize_document(file_path, top_n=None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = read_pdf(file_path)
    elif ext == '.txt':
        text = read_txt(file_path)
    else:
        return "Unsupported file format."

    return summaryFunction(text, top_n)