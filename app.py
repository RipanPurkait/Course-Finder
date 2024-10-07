import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import gradio as gr

# Scraping function to get courses data along with descriptions
def get_course_description(course_url):
    response = requests.get(course_url)
    course_soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the course description inside the custom-theme div
    description_div = course_soup.find('div', class_='custom-theme')
    if description_div:
        description = ' '.join(p.get_text(strip=True) for p in description_div.find_all('p'))
    else:
        description = "No description available"
    
    return description

# Function to scrape course data from Analytics Vidhya
def scrape_course_data():
    url = 'https://courses.analyticsvidhya.com/pages/all-free-courses'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    courses = []
    course_list = soup.find_all('a', class_='course-card course-card__public published')
    
    for course in course_list:
        link = 'https://courses.analyticsvidhya.com' + course['href']
        title_tag = course.find('h3')
        title = title_tag.get_text(strip=True) if title_tag else "No title"
        
        # Get course description by visiting the course page
        description = get_course_description(link)
        
        # Append each course's title, URL, and description
        courses.append({
            'title': title,
            'link': link,
            'description': description
        })
        
        # Pause briefly to avoid overwhelming the server
        time.sleep(1)
    
    return pd.DataFrame(courses)

# Scrape course data only once and store it in a variable
courses_df = scrape_course_data()

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Store course data and embeddings in memory variables
course_names = courses_df['title'].tolist()
course_links = courses_df['link'].tolist()
course_descriptions = courses_df['description'].tolist()

# Create embeddings only once and store in a variable
def create_embeddings(names, descriptions):
    combined_texts = [f"{name}. {desc}" for name, desc in zip(names, descriptions)]
    return model.encode(combined_texts)

embeddings = np.array(create_embeddings(course_names, course_descriptions)).astype('float32')

# Build the FAISS index once
def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

index = build_faiss_index(embeddings)

# Query function to search courses
def query_courses(query_text, top_k=5, min_score=0.4):
    query_embedding = model.encode([query_text]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        score = 1 / (1 + distances[0][i])  # Convert distance to similarity score
        if score > min_score:
            results.append({
                "name": course_names[idx],
                "link": course_links[idx],
                "score": score
            })
    
    return results

# Gradio interface function
def search_courses(query):
    results = query_courses(query)
    output = ""
    for match in results:
        output += f"Course: {match['name']}\n"
        output += f"Link: {match['link']}\n"
        output += f"Similarity Score: {match['score']:.4f}\n\n"
    return output

# Create Gradio interface
iface = gr.Interface(
    fn=search_courses,
    inputs=gr.Textbox(lines=2, placeholder="Enter your search query..."),
    outputs="text",
    title="Course Finder",
    description="Search for courses by entering a topic or keyword.",
    examples=[["machine learning"], ["data visualization"], ["python programming"],["generative ai"]]
)

# Launch the interface
iface.launch()
