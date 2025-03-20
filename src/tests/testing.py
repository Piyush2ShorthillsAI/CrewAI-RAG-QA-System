import json
import os
from llm_ops.llm2 import LLMHandler
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from llm_ops.query_to_pinecone_final import PineconeQueryHandler


class ModelValidator:
    def __init__(self, test_cases_file="tests/generated_test_cases.json", log_file="tests/test_results.log"):
        """
        Initializes the ModelValidator with paths for test cases and log file.

        Args:
            test_cases_file (str): Path to the test cases JSON file.
            log_file (str): Path to the log file to store results.
        """
        self.test_cases_file = test_cases_file
        self.log_file = log_file
        self.similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.llm_handler = LLMHandler()  # Load LLM Handler
        self.pinecone_handler = PineconeQueryHandler()

    def load_test_cases(self):
        """Loads test cases from the specified JSON file."""
        if not os.path.isfile(self.test_cases_file):
            print(f"Error: File '{self.test_cases_file}' not found.")
            return []

        try:
            with open(self.test_cases_file, "r", encoding="utf-8") as file:
                test_cases = json.load(file)
            print(f"Successfully loaded {len(test_cases)} test cases.")
            return test_cases
        except Exception as e:
            print(f"Error loading test cases: {e}")
            return []

    def is_similar(self, new_answer, expected_answer, threshold=0.5):
        """
        Checks if the new_answer is semantically or textually similar to the expected_answer.

        Args:
            new_answer (str): The model-generated answer.
            expected_answer (str): The expected correct answer.
            threshold (float): Similarity threshold (default: 0.5).

        Returns:
            bool: True if the answers are similar, False otherwise.
        """
        # Textual similarity using FuzzyWuzzy
        fuzz_score = fuzz.ratio(new_answer.lower(), expected_answer.lower()) / 100
        if fuzz_score >= threshold:
            return True

        # Semantic similarity using SentenceTransformer
        new_answer_embedding = self.similarity_model.encode(new_answer, convert_to_tensor=True)
        expected_answer_embedding = self.similarity_model.encode(expected_answer, convert_to_tensor=True)

        similarity_score = util.pytorch_cos_sim(new_answer_embedding, expected_answer_embedding)
        return similarity_score[0][0].item() >= threshold

    def validate_model_with_test_cases(self, test_cases):
        """
        Validates the model against test cases and logs the results.

        Args:
            test_cases (list): List of test cases with 'question' and 'answer' fields.
            llm_pipeline (LLMHandler): The LLM pipeline for querying.
        """
        passed = 0
        failed = 0

        # Create the directory if not present
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, "w") as file:
            for i, test_case in enumerate(test_cases, start=1):
                question = test_case["question"]
                expected_answer = test_case["answer"]

                # Query Pinecone for context
                context =  self.pinecone_handler.query(question)
                print(f"\nQuerying model for question {i}: '{question}'")

                # Get model response
                model_response = self.llm_handler.query_llm(context, question)

                # Compare similarity between model response and expected answer
                if self.is_similar(model_response, expected_answer):
                    passed += 1
                else:
                    failed += 1

                # Calculate overall similarity percentage
                similarity_percentage = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0

                # Log the results
                log_entry = {
                    "query_number": i,
                    "question": question,
                    "expected_answer": expected_answer,
                    "model_response": model_response,
                    f"similarity_percentage_query_1_to_{i}": similarity_percentage,
                    "passed": passed,
                    "failed": failed,
                }

                file.write(json.dumps(log_entry) + "\n")
                file.flush()

                print(f"Query {i}: Similarity Percentage 1 to {i}: {similarity_percentage:.2f}%")

        print(f"Test results logged to '{self.log_file}'")

    def run_validation(self):
        """Main method to load test cases and validate the model."""
        # Load test cases
        test_cases = self.load_test_cases()
    
        if test_cases:
            # Validate model with test cases
            self.validate_model_with_test_cases(test_cases)


# Run validation if executed directly
if __name__ == "__main__":
    validator = ModelValidator()
    validator.run_validation()
