import tkinter as tk
import nltk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


def summarize_text():
    marathi_text = text_input.get("1.0", tk.END)
    sentences = nltk.sent_tokenize(marathi_text)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    sentence_scores = list(enumerate(cosine_sim[0]))
    print("\n",sentence_scores)
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    num_sentences = int(summary_length.get())
    
    summary_sentences = [sentences[score[0]] for score in sentence_scores[:num_sentences]]
    summary = ' '.join(summary_sentences)
    
    summary_text.delete(1.0, tk.END)
    summary_text.insert(tk.END, summary)

window = tk.Tk()
window.title("Extractive Marathi Text Summarizer")

text_label = tk.Label(window, text="Enter Marathi Text:")
text_label.pack()

text_input = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=15)
text_input.pack()

length_label = tk.Label(window, text="Enter Summary Length (number of sentences):")
length_label.pack()

summary_length = tk.Entry(window)
summary_length.pack()

summarize_button = tk.Button(window, text="Summarize", command=summarize_text)
summarize_button.pack()

summary_label = tk.Label(window, text="Summary:")
summary_label.pack()

summary_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=50)
summary_text.pack()

# Start the GUI main loop
window.mainloop()
