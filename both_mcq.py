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


# Function to evaluate a question-answer pair
def evaluate_question_answer(question, user_answer, question_type):
    """
    Evaluate a question-answer pair based on its type (MCQ or Paragraph).
    """
    serp_result = search.run(question)
    
    # Prepare the prompt based on question type
    if question_type == "MCQ":
        prompt = f"""
        Evaluate the following Multiple Choice Question (MCQ) and user's selected answer:
        
        Question: {question}
        User's Answer: {user_answer}
        serp_API: {serp_result} (if LLM lacks real-time knowledge, use this result to evaluate the answer.)
        
        Provide the evaluation in this format:
        1. **Score**: (Numerical score out of 100)
        2. **Feedback**: (Detailed feedback for MCQ evaluation.)
        """
    else:  # Assume Paragraph-based question
        prompt = f"""
        Evaluate the following paragraph-based question and user's answer:
        
        Question: {question}
        User's Answer: {user_answer}
        serp_API: {serp_result} (if LLM lacks real-time knowledge, use this result to evaluate the answer.)
        
        Provide the evaluation in this format:
        
        1. **Score**: (Numerical score out of 100 based on accuracy, relevance, and depth.)
        2. **Feedback**: (Detailed feedback for paragraph-based evaluation.)
        """
    
    try:
        # Generate content with the model
        response = model.generate_content(prompt)
        evaluation_text = response.text.strip()

        # Extract score and feedback using regex
        score_match = re.search(r"\*\*Score\*\*:\s*(\d+)", evaluation_text)
        feedback_match = re.search(r"\*\*Feedback\*\*:\s*(.+)", evaluation_text, re.DOTALL)

        # Extract and validate the score
        score = int(score_match.group(1)) if score_match else 0
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided."

        return {
            "Question Type": question_type,
            "Score": score,
            "Feedback": feedback
        }
    except Exception as e:
        return {
            "Question Type": question_type,
            "Score": 0,
            "Feedback": f"Error processing answer: {str(e)}"
        }


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Endpoint to evaluate question-answer pairs.
    Expects a JSON payload with lists of questions and answers.
    """
    try:
        data = request.json
        questions = data.get("questions", [])
        answers = data.get("answers", [])
        
        # Validate input
        if not questions or not answers or len(questions) != len(answers):
            return jsonify({"error": "Invalid input. Ensure matching lists of questions and answers."}), 400

        evaluations = []
        for question, answer in zip(questions, answers):
            if "ID" not in question or "Text" not in question or "Type" not in question or "ID" not in answer or "Text" not in answer:
                return jsonify({"error": "Missing 'ID', 'Text', or 'Type' in question/answer."}), 400
            
            question_text = question["Text"]
            answer_text = answer["Text"]
            question_type = question["Type"]  # "MCQ" or "Paragraph"

            # Evaluate question-answer pair
            evaluation = evaluate_question_answer(question_text, answer_text, question_type)
            evaluations.append({"ID": question["ID"], "Evaluation": evaluation})
        
        return jsonify({"evaluations": evaluations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/hello", methods=["GET"])
def hello():
    """
    Health check route.
    """
    return jsonify({"message": "Server is running", "status": "OK"})


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, threaded=True)
