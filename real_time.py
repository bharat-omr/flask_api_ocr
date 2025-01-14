from flask import Flask, request, jsonify

import google.generativeai as genai
from langchain_community.utilities import SerpAPIWrapper
import os
import re

# Configure the Google Generative AI API key
genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Flask app initialization
app = Flask(__name__)

# Configure SerpAPIWrapper
os.environ["SERPAPI_API_KEY"]
search = SerpAPIWrapper()  

# Function to evaluate a single user's answer using the LLM
def evaluate_answer_llm(user_answer):
    """
    Evaluate a single user's answer and return a score and feedback extracted from the model's response.
    """
    
    serp_result = search.run(user_answer)
    # Prepare the prompt for evaluation
    prompt = f"""
    Evaluate the following user's answer:

    User's Answer: {user_answer} 
    serp_API :{serp_result} (if you have not real time knowledge so use this serp api result data)
    Provide the evaluation in this format:
    1. **Score**: (numerical score out of 100)
    2. **Feedback**: (detailed feedback text)
    """
    
    # Generate content with the model
    try:
        response = model.generate_content(prompt)
        evaluation_text = response.text.strip()

        # Extract score and feedback using regex
        score_match = re.search(r"\*\*Score\*\*:\s*(\d+)", evaluation_text)
        feedback_match = re.search(r"\*\*Feedback\*\*:\s*(.+)", evaluation_text, re.DOTALL)

        # Extract and validate the score
        score = int(score_match.group(1)) if score_match else 0
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided."

        # Return as a JSON-like dictionary
        return {
            "Score": score,
            "Feedback": feedback
        }
    except Exception as e:
        # Handle errors gracefully
        return {
            "Score": 0,
            "Feedback": f"Error processing answer: {str(e)}"
        }

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
            
            # Evaluate the answer
            evaluation = evaluate_answer_llm(answer["Text"])
            evaluations.append({"ID": answer["ID"], "Evaluation": evaluation})
        
        print(f"Evaluations: {evaluations}")    
        # Return the evaluations
        return jsonify({"evaluations": evaluations}), 200

    except Exception as e:
        print(f"Error in evaluate route: {str(e)}") 
        return jsonify({"error": str(e)}), 500

@app.route("/hello",methods=["GET"])
def hello():
    return jsonify({"message":"hello"})


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, threaded=True)
   
