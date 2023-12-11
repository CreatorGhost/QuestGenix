import time
from helper.interview_assistant import InterviewAssistant
import logging


import matplotlib.pyplot as plt
# Configuration
MODEL_NAMES = ["gpt-4-1106-preview","gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-1106"]
NUM_REQUESTS = 25  # Number of requests to send for each model

# Placeholder for timing and error results
results = {}
errors = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='benchmark.log',
    filemode='w'
)

def benchmark_model(model_name):
    assistant = InterviewAssistant(model_name=model_name)
    timings = []
    error_count = 0
    for i in range(NUM_REQUESTS):
        start_time = time.time()
        try:
            questions_dict = assistant.generate_questions("data science", ["Python", "SQL", "Maths"], "beginner")
            end_time = time.time()
            elapsed_time = end_time - start_time
            timings.append(elapsed_time)
            logging.info(f"Request {i+1} for model {model_name} succeeded in {elapsed_time:.2f} seconds.")
            print(f"Request {i+1} for model {model_name} succeeded in {elapsed_time:.2f} seconds.")
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            error_count += 1
            logging.error(f"Request {i+1} for model {model_name} failed: {e} in {elapsed_time:.2f} seconds.")
            print(f"Request {i+1} for model {model_name} failed: {e} in {elapsed_time:.2f} seconds.")
    error_rate = error_count / NUM_REQUESTS
    errors[model_name] = error_rate
    return timings

# Run benchmark for each model
for model in MODEL_NAMES:
    print(f"Benchmarking {model}...")
    results[model] = benchmark_model(model)
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