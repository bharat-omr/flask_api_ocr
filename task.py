import google.generativeai as genai

# Configure the Google Generative AI API key
genai.configure(api_key="api")

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to evaluate a single user's answer using the LLM
def evaluate_answer_llm(user_answer):
    """
    Evaluate a single user's answer (in JSON format) and return a score and feedback.
    """
    # Prepare the prompt for evaluation
    prompt = f"""
    Evaluate the following user's answer:

    User's Answer: {user_answer}

    Example format for user's answer:
    {{
        "ID": "S1_Q1",
        "Text": "There is the text in the provided image."
    }}

    Based on your knowledge, provide the following:
    Format the response as:
    1. **Score**: (numerical score out of 100)
    2. **Feedback**
    """
    
    # Generate content with the model
    response = model.generate_content(prompt)
    
    # Extract and return the evaluation response
    evaluation = response.text.strip()
    return evaluation

# Example user answers in JSON format
user_answers = [
    {
        "ID": "S1_Q1",
        "Text": "Exercise helps in maintaining physical health, improves mental well-being, and reduces the risk of chronic diseases."
    },
    {
        "ID": "S1_Q2",
        "Text": "The water cycle involves evaporation, condensation, and precipitation, helping in the circulation of water on Earth."
    },
    {
        "ID": "S1_Q3",
        "Text": "The sky is blue because of the ocean's reflection."
    },
]

# Evaluate each user answer and print the results
for answer in user_answers:
    evaluation = evaluate_answer_llm(answer)
    print(f"Evaluation for ID {answer['ID']}:\n{evaluation}\n")
