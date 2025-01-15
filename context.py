import google.generativeai as genai
import os
genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
sample_pdf = genai.upload_file( "Written Assessment (With Answers) - Build Passive Income with Proven Affiliate Marketing Strategies (1).pdf")
response = model.generate_content(["Give me a summary of this pdf file in 50 words.", sample_pdf])
print(response.text)
