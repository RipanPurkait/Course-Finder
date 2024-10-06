import streamlit as st
import csv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize lists to store data
course_names = []
course_links = []
embeddings = []

# Load courses from CSV and create embeddings
def load_courses_from_csv(uploaded_file):
    global course_names, course_links, embeddings
    
    # Use csv.DictReader directly on the uploaded file
    csv_reader = csv.DictReader(uploaded_file.read().decode('utf-8').splitlines())
    
    for row in csv_reader:
        course_name = row['title']
        course_link = row['link']
        
        # Create an embedding for the course name
        embedding = model.encode(course_name)
        
        course_names.append(course_name)
        course_links.append(course_link)
        embeddings.append(embedding)
    
    # Convert embeddings list to a numpy array
    embeddings = np.array(embeddings).astype('float32')
    
    return embeddings

# Build the FAISS index
def build_faiss_index(embeddings):
    # Get the dimensionality of the embeddings
    d = embeddings.shape[1]
    
    # Create a new index
    index = faiss.IndexFlatL2(d)
    
    # Add the vectors to the index
    index.add(embeddings)
    
    return index

# Query function to search courses
def query_courses(index, query_text, top_k=5, min_score=0.4):
    # Create an embedding for the query text
    query_embedding = model.encode(query_text).reshape(1, -1).astype('float32')
    
    # Perform the search
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        score = 1 / (1 + distances[0][i])  # Convert distance to a similarity score
        if score > min_score:  # Filter based on minimum similarity score
            results.append({
                "name": course_names[idx],
                "link": course_links[idx],
                "score": score
            })
    
    return results

# Streamlit App UI
st.title("Course Finder")
st.write("Search for courses by entering a topic or keyword.")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload course data CSV", type="csv")

if uploaded_file:
    # Load courses and create index
    embeddings = load_courses_from_csv(uploaded_file)
    index = build_faiss_index(embeddings)

    # Input box for query
    query = st.text_input("Enter your search query:")

    # Perform search on button click
    if st.button("Search") and query:
        results = query_courses(index, query)
        
        # Display results
        st.write("### Search Results:")
        for match in results:
            st.write(f"**Course:** {match['name']}")
            st.write(f"**Link:** [View Course]({match['link']})")
            st.write(f"**Similarity Score:** {match['score']:.4f}")
            st.write("---")
