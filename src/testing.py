import pandas as pd
import os
from bert_score import score
import openpyxl
from llm_ops.llm2 import LLMHandler
from llm_ops.query_to_pinecone_final import PineconeQueryHandler



class BertProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.llm_handler = LLMHandler()
        self.pinecone_handler = PineconeQueryHandler()
        self.new_columns = [
            "LLM_Answer", "Retrieved_Context", "Faithfulness_Score",
            "Answer_Correctness_Score", "Answer_Relevancy_Score",
            "Context_Precision_Score", "Context_Recall_Score",
            "BERT_Final_Score"
        ]
        self.df = None
        self.prev_df = None

    def load_input_data(self):
        """Load input data from the original Excel file."""
        self.df = pd.read_excel(self.input_file, engine="openpyxl")
        print(f"Loaded input data from: {self.input_file}")

    def load_previous_data(self):
        """Load previous results to preserve processed rows."""
        if os.path.exists(self.output_file):
            self.prev_df = pd.read_excel(self.output_file)
            print(f"Previous results loaded from: {self.output_file}")
        else:
            self.prev_df = pd.DataFrame()

    def merge_previous_data(self):
        """Merge previous records to preserve existing scores."""
        if not self.prev_df.empty:
            self.df.update(self.prev_df)

    def add_missing_columns(self):
        """Add missing columns if they are not in the input file."""
        for column in self.new_columns:
            if column not in self.df.columns:
                self.df[column] = None

    def calculate_bertscore(self, reference, hypothesis):
        """Calculate BERTScore using BERT-base for semantic similarity."""
        if pd.isnull(reference) or pd.isnull(hypothesis):
            return 0.0

        # BERTScore with BERT-base as the model
        P, R, F1 = score(
            [hypothesis], [reference], lang="en", model_type="bert-base-uncased"
        )
        return round(F1.item(), 4)

    def calculate_final_score(self, scores):
        """Calculate final BERTScore-based score."""
        # Average of all metric scores
        return round(sum(scores) / len(scores), 4)

    def process_row(self, index, row):
        """Process a single row to get LLM answer, retrieve context, and calculate scores."""
        query = row["Question"]
        ground_truth = row["Ground_truth"]

        # Query Pinecone for context
        retrieved_context = self.pinecone_handler.query(query)
        llm_answer = self.llm_handler.query_llm(retrieved_context, query)

        # Debug info for verification
        print(f"Processing Row {index + 1}: Query: {query}")
        print(f"Retrieved Context: {retrieved_context}")
        print(f"LLM Answer: {llm_answer}")

        # BERT-based metric evaluations
        faithfulness_score = self.calculate_bertscore(retrieved_context, llm_answer)  # Faithfulness
        correctness_score = self.calculate_bertscore(ground_truth, llm_answer)  # Answer Correctness
        relevancy_score = self.calculate_bertscore(query, llm_answer)  # Answer Relevancy
        context_precision_score = self.calculate_bertscore(query, retrieved_context)  # Context Precision
        context_recall_score = self.calculate_bertscore(retrieved_context, query)  # Context Recall

        # Calculate final BERT-based score
        final_score = self.calculate_final_score([
            faithfulness_score, correctness_score, relevancy_score,
            context_precision_score, context_recall_score
        ])

        # Update DataFrame with results
        self.df.at[index, "LLM_Answer"] = llm_answer
        self.df.at[index, "Retrieved_Context"] = retrieved_context
        self.df.at[index, "Faithfulness_Score"] = faithfulness_score
        self.df.at[index, "Answer_Correctness_Score"] = correctness_score
        self.df.at[index, "Answer_Relevancy_Score"] = relevancy_score
        self.df.at[index, "Context_Precision_Score"] = context_precision_score
        self.df.at[index, "Context_Recall_Score"] = context_recall_score
        self.df.at[index, "BERT_Final_Score"] = final_score
        print(f"Row {index + 1}/{len(self.df)} processed successfully. Final Score: {final_score}")

    def save_progress(self):
        """Save progress to the output file to preserve results."""
        self.df.to_excel(self.output_file, index=False)
        print(f" Progress saved at: {self.output_file}")

    def process_excel(self):
        """Main function to process the entire Excel and calculate scores."""
        self.load_input_data()
        self.load_previous_data()
        self.add_missing_columns()
        self.merge_previous_data()

        # Process only unprocessed or zero-score rows
        for index, row in self.df.iterrows():
            try:
                # Check if BERT_Final_Score is valid and not zero
                if pd.isnull(row["BERT_Final_Score"]) or row["BERT_Final_Score"] == 0:
                    self.process_row(index, row)
                    # Save progress after each row to prevent data loss
                    self.save_progress()
                else:
                    print(f"Skipping row {index + 1}/{len(self.df)}: Already scored.")
            except Exception as e:
                print(f" Error processing row {index + 1}: {str(e)}")
                self.save_progress()
                break

        # Final save after completion
        self.save_progress()
        print(f" All rows processed. Final file saved at: {self.output_file}")


# Configuration and setup
if __name__ == "__main__":
    # Paths and API configurations
    INPUT_FILE = "data/q&a_rag_application.xlsx"
    OUTPUT_FILE = "src/test_results/bert_base_scores_.xlsx"

    # Initialize and run the BertProcessor
    processor = BertProcessor(INPUT_FILE, OUTPUT_FILE)
    processor.process_excel()

# Precision (P): Proportion of relevant information in the predicted answer.

# Recall (R): Proportion of ground truth information captured in the predicted answer.

# F1-Score (F1): Harmonic mean of Precision and Recall.

# Faithfulness Score:

#     LLM answer compared to the retrieved context.

# Answer Correctness Score:

#     LLM answer compared to the ground truth.

# Answer Relevancy Score:

#     LLM answer compared to the original query.

# Context Precision Score:

#     Context compared to the query to check relevance.

# Context Recall Score:

#     Checks if the context contains enough information for the query.

# Use BERTScore with bert-base-uncased to evaluate:

#     Faithfulness: Measures how well the LLM answer aligns with the retrieved context.

#     Answer Correctness: Measures similarity between the LLM answer and the ground truth.

#     Answer Relevancy: Checks if the answer addresses the query correctly.

#      Context Precision: Evaluates how much of the context is relevant to the query.

# #     Context Recall: Measures whether the context contains all necessary information.

