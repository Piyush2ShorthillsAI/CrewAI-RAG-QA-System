import json
import random

# Function to load data from output.json file
def load_articles_from_json(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
            return articles
    except Exception as e:
        print(f"⚠️ Error loading JSON file: {e}")
        return []

# Function to generate a simple test case based on article content
def generate_test_case_from_article(article):
    title = article["title"]
    content = article["content"]
    
    # Generating a test case (question-answer pair) based on content
    question = f"Can you explain what the '{title}' page on CrewAI is about?"
    answer = content[:100]  # Just a part of the content as an answer (you can adjust this part)

    return {
        "question": question,
        "answer": answer
    }

# Path to your output.json file
output_json_file = "output.json"  # Change this to the path where your output.json file is located

# Load articles data from output.json file
articles_data = load_articles_from_json(output_json_file)

if not articles_data:
    print("⚠️ No articles data found. Exiting.")
else:
    # Generate 1000 test cases based on available data
    test_cases = []
    for _ in range(1000):
        article = random.choice(articles_data)  # Randomly pick an article
        test_case = generate_test_case_from_article(article)
        test_cases.append(test_case)

    # Save the generated test cases to a file
    with open("generated_test_cases.json", "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=4, ensure_ascii=False)

    print("✅ 1000 Test cases generated and saved to 'generated_test_cases.json'")
