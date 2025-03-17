import nltk
import streamlit as st
import pickle
import re

nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Fixed variable name

def cleanResume(txt):
    cleanTxt = re.sub(r'http\S+\s', ' ', txt)  # Remove URLs
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)  # Remove mentions
    cleanTxt = re.sub(r'#\S+\s', ' ', cleanTxt)  # Remove hashtags
    cleanTxt = re.sub(r'\b(RT|cc)\b', ' ', cleanTxt, flags=re.IGNORECASE)  # Remove RT and cc
    cleanTxt = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""), ' ', cleanTxt)  # Remove special characters
    cleanTxt = re.sub(r'[^\x00-\x7F]+', ' ', cleanTxt)  # Remove non-ASCII characters
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()  # Remove extra spaces
    return cleanTxt

# Web App
def main():
    st.title("Resume Screening")
    Upload_file = st.file_uploader("Upload resume", type=['txt', 'pdf'])
    
    if Upload_file is not None:
        try:
            resume_bytes = Upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)  # Fixed function call

        input_features = tfidf.transform([cleaned_resume])  # Fixed variable name
        prediction_id = clf.predict(input_features)[0]  # Fixed variable name
        st.write("Predicted ID:", prediction_id)

        category_mapping = {
            0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain", 4: "Business Analyst",
            5: "Civil Engineer", 6: "Data Science", 7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
            10: "ETL Developer", 11: "Electrical Engineering", 12: "HR", 13: "Hadoop", 14: "Health and fitness",
            15: "Java Developer", 16: "Mechanical Engineer", 17: "Network Security Engineer", 18: "Operations Manager",
            19: "PMO", 20: "Python Developer", 21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Category Prediction:", category_name)

# Run App
if __name__ == "__main__":
    main()
    