import genai
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

response = model.generate_content(["What is the meaning of black hole"])