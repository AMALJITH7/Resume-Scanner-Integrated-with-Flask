import numpy as np
import re
import time
import PyPDF2
from PyPDF2 import PdfReader
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = r'res\uploads'

nltk.download('punkt')  # Download 'punkt' tokenizer
nltk.download('stopwords')  # Download stopwords

def remove_punctuation(text):
    import string
    remover = str.maketrans('', '', string.punctuation)
    return text.translate(remover)

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resume', methods=['GET', 'POST'])
def checking():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        # Save the uploaded PDF file to the designated folder
        if file:
            # Ensure the folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            # Save the uploaded PDF file with a secure filename
            uploaded_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_resume.pdf')
            file.save(uploaded_pdf_path)
            
            readere = open(uploaded_pdf_path, 'rb')
            
            reader = PdfReader(readere)

            # Extract text from the uploaded PDF
            resume_text = ''

            # Loop through each page of the PDF
            for page_num in range(len(reader.pages)):
                # Extract text from the current page
                page = reader.pages[page_num]
                resume_text += page.extract_text()
             
            words = word_tokenize(resume_text)
            
            # Load NLTK stopwords
            stop_words = set(stopwords.words('english'))
            # Remove stopwords from the resume words
            filtered_words = [word for word in words if word.lower() not in stop_words]
            
            # Join the filtered words back into a string
            filtered_resume_text = ' '.join(filtered_words)
            
            filtered_resume_text = ''.join(map(remove_punctuation, filtered_resume_text))
            filtered_resume_text = filtered_resume_text.lower()
           
           # Tokenize filtered_resume_text into words
            filtered_words = filtered_resume_text.split()

            # Check if imported skills are present anywhere in the uploaded file
            uploaded_skills = request.form['sample']
            uploaded_skills = ''.join(remove_punctuation(char) for char in uploaded_skills).lower()
            
            common_words = []

            for word in filtered_words:
                if word in uploaded_skills:
                    common_words.append(word)
                
            unique_common_words = list(set(common_words))
            
            content = [unique_common_words, uploaded_skills]
            content = np.array(content, dtype=object)
            content = np.array([str(item) for item in content])
            
            cv = CountVectorizer()
            matrix = cv.fit_transform(content)
         
            similar = cosine_similarity(matrix)
            acc = similar[1][0] * 100
            precision = "%.2f" % acc

            return render_template('output.html', result=precision)

if __name__ == '__main__':
    app.run(debug=True)