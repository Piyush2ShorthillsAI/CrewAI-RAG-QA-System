import json
import random


class TestCaseGenerator:
    def __init__(self, json_file="data/output.json", output_file="src/tests/generated_test_cases.json"):
        """
        Initializes the TestCaseGenerator with input and output file paths.

        Args:
            json_file (str): Path to the input JSON file.
            output_file (str): Path to the output JSON file where test cases will be stored.
        """
        self.json_file = json_file
        self.output_file = output_file
        self.articles_data = self.load_articles_from_json(json_file)

    def load_articles_from_json(self, json_file):
        """
        Loads data from the specified JSON file.

        Args:
            json_file (str): Path to the input JSON file.

        Returns:
            list: List of articles if successfully loaded, otherwise an empty list.
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                articles = json.load(f)
                return articles
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []

    def generate_test_case_from_article(self, article):
        """
        Generates a simple test case based on article content.

        Args:
            article (dict): A dictionary containing 'title' and 'content' keys.

        Returns:
            dict: A dictionary containing 'question' and 'answer'.
        """
        title = article["title"]
        content = article["content"]

        # Generating a test case (question-answer pair) based on content
        question = f"Can you explain what the '{title}' page on CrewAI is about?"
        answer = content[:100]  # Just a part of the content as an answer (you can adjust this part)

        return {
            "question": question,
            "answer": answer
        }

    def generate_test_cases(self, num_cases=1000):
        """
        Generates a specified number of test cases from the available articles.

        Args:
            num_cases (int): Number of test cases to generate.

        Returns:
            list: List of generated test cases.
        """
        if not self.articles_data:
            print("No articles data found. Exiting.")
            return []

        test_cases = []
        for _ in range(num_cases):
            article = random.choice(self.articles_data)  # Randomly pick an article
            test_case = self.generate_test_case_from_article(article)
            test_cases.append(test_case)

        return test_cases

    def save_test_cases(self, test_cases):
        """
        Saves the generated test cases to a JSON file.

        Args:
            test_cases (list): List of generated test cases.
        """
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(test_cases, f, indent=4, ensure_ascii=False)
            print(f"{len(test_cases)} Test cases generated and saved to '{self.output_file}'")
        except Exception as e:
            print(f"Error saving test cases: {e}")


def main():
    """
    Main function to execute the test case generation and saving.
    """
    generator = TestCaseGenerator()
    test_cases = generator.generate_test_cases(num_cases=1000)

    if test_cases:
        generator.save_test_cases(test_cases)


if __name__ == "__main__":
    main()
