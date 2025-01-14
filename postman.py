from flask import Flask, request, jsonify
import google.generativeai as genai
import os

# Configure the Google Generative AI API key
genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Flask app initialization
app = Flask(__name__)

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

# Flask route to evaluate user answers
@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Endpoint to evaluate user answers.
    Expects a JSON payload with a list of user answers.
    """
    try:
        # Get the JSON data from the request
        user_answers = request.json.get("answers", [])
        
        if not user_answers or not isinstance(user_answers, list):
            return jsonify({"error": "Invalid input. Expected a list of answers."}), 400

        # Evaluate each answer
        evaluations = []
        for answer in user_answers:
            # Ensure each answer has the required fields
            if "ID" not in answer or "Text" not in answer:
                return jsonify({"error": f"Missing 'ID' or 'Text' in answer: {answer}"}), 400
            
            evaluation = evaluate_answer_llm(answer)
            evaluations.append({"ID": answer["ID"], "Evaluation": evaluation})
        
        # Return the evaluations
        return jsonify({"evaluations": evaluations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
