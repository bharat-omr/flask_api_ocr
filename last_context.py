from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import re

# Configure the Google Generative AI API key
genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")
sample_pdf = genai.upload_file( "answer.pdf")
app = Flask(__name__)
# Function to evaluate a single question-answer pair using the LLM
def evaluate_question_answer(question, user_answer, class_name, board, word_count,sample_pdf):
    """
    Evaluate a question-answer pair and return a score and feedback extracted from the model's response.
    """
    prompt = f"""
Evaluate the given question and user's answer based on the following context:

Class: {class_name}
Board: {board}
Expected Word Count: {word_count}
Question: {question}
User's Answer: {user_answer}
correct answer :{sample_pdf}
Instructions:
1. Need to elvauate user answer with {sample_pdf} because correct answer of all question are in this pdf.
1. Verify if the user's answer adheres to the expected word count ({word_count}). If the word count matches or exceeds the requirement, proceed to evaluate the answer based on its correctness, clarity, and completeness.
2. If the user's answer has fewer words than required, deduct marks appropriately and provide feedback explaining the shortfall.
3. If there are minor mistakes in the user's answer (0-10% of the content), do not penalize the score but in maths not use this. Focus instead on constructive feedback to address the minor inaccuracies.
4. If the question or answer is correct in one word and matches the expected accuracy, assign a score of 100 instead of 95.
5. For mathematical answers:
   - Penalize minor errors (0-10%) by reducing the score based on the level of accuracy. Even small inaccuracies (e.g., in calculations or results) should impact the score.
   - Provide feedback explaining where the mistake occurred and how to correct it.
   
Question: {question}
User's Answer: {user_answer}
correct answer :{sample_pdf}
Provide the output in the following format:

1. **Score**: (Numerical score out of 100, calculated based on adherence to word count, accuracy, clarity, and completeness. Minor mistakes (0-10%) should not heavily impact the score.)
2. **Feedback**: (Provide 2-3 sentences highlighting the strengths and areas for improvement. If the question is in a specific language, give the feedback in the same language.)
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
            "Score": score,
            "Feedback": feedback
        }
    except Exception as e:
        return {
            "Score": 0,
            "Feedback": f"Error processing answer: {str(e)}"
        }

# Flask route to evaluate user answers
@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Endpoint to evaluate question-answer pairs.
    Expects a JSON payload with the format:
    {
        "Class": "10th",
        "Board": "CBSE",
        "Type": "one mark",
        "questions": [
            {"ID": "1", "Text": "Where is India located?"}
        ],
        "answers": [
            {"ID": "1", "Text": "India is in USA"}
        ]
    }
    """
    try:
        # Get the JSON data from the request
        data = request.json
        class_name = data.get("Class", "")
        board = data.get("Board", "")
        word_count = data.get("word_count", "")
        questions = data.get("questions", [])
        answers = data.get("answers", [])

        # Validate the input
        if not questions or not answers or len(questions) != len(answers):
            return jsonify({"error": "Invalid input. Ensure matching lists of questions and answers."}), 400

        evaluations = []
        for question, answer in zip(questions, answers):
            # Ensure each question and answer has the required fields
            if "ID" not in question or "Text" not in question or "ID" not in answer or "Text" not in answer:
                return jsonify({"error": "Missing 'ID' or 'Text' in question/answer."}), 400

            # Evaluate the question-answer pair
            evaluation = evaluate_question_answer(
                question["Text"], answer["Text"], class_name, board, word_count,sample_pdf
            )
            evaluations.append({"ID": question["ID"], "Evaluation": evaluation})
            
        print(f"Evaluations: {evaluations}")
        return jsonify({"evaluations": evaluations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "hello"})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, threaded=True)
