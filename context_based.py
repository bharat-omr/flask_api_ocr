from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain_community.utilities import SerpAPIWrapper
import os
import re
import time
# Configure the Google Generative AI API key
genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")
# Upload sample PDF for context
try:
    sample_pdf = genai.upload_file("answer.pdf")
except Exception as e:
    print(f"Error uploading PDF: {e}")
    sample_pdf = None  # Handle case where PDF upload fails


# Flask app initialization
app = Flask(__name__)

# Configure SerpAPIWrapper (commented out)
#os.environ["SERPAPI_API_KEY"]
#search = SerpAPIWrapper()


# Function to evaluate a single question-answer pair using the LLM
def evaluate_question_answer(question, user_answer):
    """
    Evaluate a question-answer pair and return a score and feedback extracted from the model's response.
    """
    if sample_pdf is None:
      return {
            "Score": 0,
            "Feedback": "PDF upload failed, cannot evaluate"
        }
    #serp_result = search.run(question)
    # Prepare the prompt for evaluation
    prompt = f"""
    Evaluate the following question and user's answer:
    
    Question: {question}
    User's Answer: {user_answer} 
    
    
    Instructions:
    1. Compare the user's answer with the information in the provided PDF.
    2. Assign a score based in pdf mention. If the user's answer is completely correct and aligns perfectly with the PDF, assign full marks; otherwise, the score should reflect the degree of correctness.
    3. Provide feedback concisely (2-3 sentences), why you are cut some marks like user answer is correct or accurate. 
    4. If the question language is Hindi, provide feedback in Hindi; otherwise, use the language of the question.
    
    Provide the evaluation in this format:
    1. **Score**: (based of accurate pdf mention)
    2. **Feedback**: (Only give clarity or mistake two or three sentence or also give feedback in same as question language if question language is hindi.)
    """
    
    
    try:
        # Generate content with the model
        response = model.generate_content([prompt,sample_pdf])
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
    Endpoint to evaluate question-answer pairs.
    Expects a JSON payload with lists of questions and answers.
    """
    try:
        # Get the JSON data from the request
        data = request.get_json() # Use get_json() for better error handling
        if not data or "questions" not in data or "answers" not in data:
            return jsonify({"error": "Invalid JSON format. Expects 'questions' and 'answers' keys."}), 400


        questions = data.get("questions", [])
        answers = data.get("answers", [])
        
        # Validate the input
        if not isinstance(questions, list) or not isinstance(answers, list) or len(questions) != len(answers):
            return jsonify({"error": "Invalid input. Ensure matching lists of questions and answers."}), 400

        evaluations = []
        for question, answer in zip(questions, answers):
            # Ensure each question and answer has the required fields
            if not isinstance(question, dict) or "ID" not in question or "Text" not in question or \
               not isinstance(answer, dict) or "ID" not in answer or "Text" not in answer:
                return jsonify({"error": "Missing 'ID' or 'Text' in question/answer."}), 400
            
            # Evaluate the question-answer pair
            evaluation = evaluate_question_answer(question["Text"], answer["Text"])
            evaluations.append({"ID": question["ID"], "Evaluation": evaluation})
        
        print(f"Evaluations: {evaluations}")    
        # Return the evaluations
        return jsonify({"evaluations": evaluations}), 200

    except Exception as e:
        print(f"Error in evaluate route: {str(e)}") 
        return jsonify({"error": str(e)}), 500

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "hello"})


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, threaded=True, debug=True)