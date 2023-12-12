import time
from helper.interview_assistant import InterviewAssistant
import logging
import matplotlib.pyplot as plt

from models.answer import AnswerEvaluationRequest, QuestionAnswerPair


# Configuration
MODEL_NAMES = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]  # Corrected model names
NUM_REQUESTS = 20 # Number of requests to simulate

# Placeholder for timing results
results = {}
errors = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='evaluate_answers_benchmark.log',
    filemode='w'
)


# Payload for benchmarking
qa_pairs_payload = {
  "qa_pairs": [
    {
        "question": "What is Data Science and how does it differ from traditional statistics?",
        "answer": "Data Science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It differs from traditional statistics in that it incorporates techniques and theories from broader fields like mathematics, statistics, information science, and computer science. Traditional statistics, on the other hand, is more focused on data analysis and interpretation in isolation."
    },
    {
        "question": "Can you briefly explain the data science lifecycle?",
        "answer": "The data science lifecycle typically includes the following steps: 1) Business Understanding - defining the problem and desired outcomes. 2) Data Mining - gathering the required data. 3) Data Cleaning - removing inconsistencies and inaccuracies in the data. 4) Data Exploration - using statistical techniques to identify patterns or anomalies in the data. 5) Feature Engineering - modifying or creating new variables to better represent the underlying data patterns. 6) Predictive Modeling - creating a statistical model for prediction. 7) Data Visualization - visualizing the data and model outcomes. 8) Deployment - implementing the model into production. 9) Monitoring - tracking the model's performance over time."
    },
    {
        "question": "What are some common libraries in Python used for data science and what are their uses?",
        "answer": "Some common Python libraries used in data science include: 1) NumPy - used for numerical computations and handling arrays. 2) Pandas - used for data manipulation and analysis. 3) Matplotlib - used for data visualization. 4) Scikit-learn - used for machine learning and statistical modeling. 5) TensorFlow and PyTorch - used for deep learning."
    },
    {
        "question": "What is a pandas DataFrame in Python and how is it used in data science?",
        "answer": "A pandas DataFrame is a two-dimensional, size-mutable, heterogeneous tabular data structure with labeled axes (rows and columns). It can be thought of as a dictionary-like container for Series objects. In data science, it is used for data manipulation and analysis. It provides various functionalities like slicing, indexing, joining, merging, reshaping, etc."
    },
    {
        "question": "Can you describe what 'data cleaning' is and why it's important in data science?",
        "answer": "Data cleaning, also known as data cleansing or data scrubbing,we use it to add value to the data by adding stopwords to make it longer for our model to understand"
    }
]
}


def benchmark_evaluate_answers(model_name):
    assistant = InterviewAssistant(model_name=model_name)
    timings = []
    error_count = 0
    qa_pairs = [QuestionAnswerPair(**qa_pair) for qa_pair in qa_pairs_payload["qa_pairs"]]
    evaluation_request = AnswerEvaluationRequest(qa_pairs=qa_pairs)

    for i in range(NUM_REQUESTS):
        try:
            start_time = time.time()
            # Pass the list of QuestionAnswerPair objects to the evaluate_answers function
            evaluation_results = assistant.evaluate_answers(evaluation_request.qa_pairs)
            elapsed_time = time.time() - start_time
            timings.append(elapsed_time)
            print(f"Evaluation {i+1} for model {model_name} succeeded in {elapsed_time:.2f} seconds.")
            logging.info(f"Evaluation {i+1} for model {model_name} succeeded in {elapsed_time:.2f} seconds.")
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_count += 1
            print(f"Evaluation {i+1} for model {model_name} failed: {e} in {elapsed_time:.2f} seconds.")
            logging.error(f"Evaluation {i+1} for model {model_name} failed: {e} in {elapsed_time:.2f} seconds.")

    error_rate = error_count / NUM_REQUESTS
    errors[model_name] = error_rate
    return timings, error_rate  # Return both timings and error rate

# Run benchmark for each model
for model in MODEL_NAMES:
    print(f"Benchmarking evaluation for {model}...")
    results[model], errors[model] = benchmark_evaluate_answers(model)  # Store both results

# Increase default font sizes for better readability
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16})

# Plotting the results with a larger figure size
fig, ax1 = plt.subplots(figsize=(14, 8))  # Increase figure size

for model, timings in results.items():
    ax1.plot(timings, label=f"{model} Response Time", marker='o')  # Added markers for clarity

ax1.set_title('Model Response Times and Error Rates')
ax1.set_xlabel('Request Number')
ax1.set_ylabel('Response Time (seconds)')
ax1.legend(loc='upper left')
ax1.grid(True)  # Add gridlines

# Plot error rates on the same graph with a secondary y-axis
ax2 = ax1.twinx()
for model, error_rate in errors.items():
    ax2.plot([error_rate] * NUM_REQUESTS, label=f"{model} Error Rate", linestyle='--', marker='x')  # Added markers

ax2.set_ylabel('Error Rate')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--')  # Add dashed gridlines

# Display the plot
plt.tight_layout()  # Adjust the padding between and around subplots
plt.show()

# Save the plot to the local directory with a higher resolution
fig.savefig('benchmark_results.png', dpi=300) 