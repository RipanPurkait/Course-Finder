# import requests
# from bs4 import BeautifulSoup
# import pandas as pd

# def scrape_course_data():
#     url = 'https://courses.analyticsvidhya.com/courses'
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     courses = []
    
#     # Find all course links and titles
#     course_list = soup.find_all('h3', class_='course-card__body')
    
#     for course in course_list:
#         title = course.get_text(strip=True)  # Get the title text
#         link = 'https://courses.analyticsvidhya.com' + course['href']  # Create full URL
        
#         # Append each course's title and URL
#         courses.append({
#             'title': title,
#             'link': link
#         })
    
#     return pd.DataFrame(courses)

# # Save the scraped data to a CSV for further processing
# courses_df = scrape_course_data()
# print(courses_df)
# courses_df.to_csv('courses_data.csv', index=False)



import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_course_data():
    url = 'https://courses.analyticsvidhya.com/pages/all-free-courses'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    courses = []
    
    # Find all course card elements
    course_list = soup.find_all('a', class_='course-card course-card__public published')
    
    for course in course_list:
        # Extract the link from the 'href' attribute
        link = 'https://courses.analyticsvidhya.com' + course['href']
        
        # Find the h3 tag inside each course card for the title
        title_tag = course.find('h3')
        title = title_tag.get_text(strip=True) if title_tag else "No title"
        
        # Append each course's title and URL
        courses.append({
            'title': title,
            'link': link
        })
    
    return pd.DataFrame(courses)

# Save the scraped data to a DataFrame for further processing
courses_df = scrape_course_data()
courses_df.to_csv('courses_data.csv', index=False)
