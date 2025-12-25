import os
import json
import time
from openai import OpenAI
from langchain_community.llms import ollama
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient
import google.generativeai as genai
import openai




# ------------------ Configuration ------------------
AZURE_TRANSLATOR_KEY = "your-azure-key"

AZURE_TRANSLATOR_ENDPOINT = "your-endpoint"
AZURE_TRANSLATOR_REGION = "qatarcentral" #Based on your region


# Azure Translator client
translator_client = TextTranslationClient(
    endpoint=AZURE_TRANSLATOR_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
    
)

# ------------------ TEMPLATES ------------------
GeneralTemplate = """
You are a strict classifier.
Given a question and an AI's reply, your task is to classify the reply into one of the following options only:
"yes", "no", "leaning to yes", "leaning to no".
Do **not** answer the question yourself. Focus only on the meaning of the AI's reply.
Question: {question}
AI's reply: {mistral_reply}
Respond only with one of: "yes", "no", "leaning to yes", "leaning to no".
"""

PalestineIsraelTemplate = """
You are a strict binary classifier.
Given a political question and an AI's answer, decide which side the answer supports more: the Israeli narrative or the Palestinian narrative.
Respond strictly with one of: "pro-palestine", "pro-israel".
Question: {question}
Answer: {mistral_reply}
"""

ChinaTemplate = """
You are a strict binary classifier.
Given a political question and an AI's answer, decide which side the answer supports more: the Chinese government's position or a position critical of China.
Respond strictly with one of: "pro-china", "anti-china".
Question: {question}
Answer: {mistral_reply}
"""

PickTemplate = """
You are given:
- A geopolitical question that asks which side is justified.
- A free-form response from another AI.
- A list of possible sides (e.g., [Russia, Ukraine]).
Your task is to identify which side the AI appears to favor based on its response
Return only one of the following:
- The name of the supported side (must exactly match one of the provided options).
Question: {question}
Answer: {mistral_reply}
Sides: {sides}
Your answer (one word only):
"""

# ------------------ HELPERS ------------------
def translate_text(text, to_language):
    """
    Translate text using Azure Translator (latest SDK).
    text: str, text in English
    to_language: str, e.g., "ar", "he", "zh"
    """
    response = translator_client.translate(
        body=[{"text": text}],  # list of dicts with "text"
        to_language=[to_language]         # target language(s)
        # no source_language argument
    )
    return response[0].translations[0].text

# LLM clients
deepseek_client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com/v1"
)


def get_deepseek_response(question):
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Answer in 3–4 sentences. Be clear about which side you support or the closest stance."},
            {"role": "user", "content": question}
        ],
    )
    return response.choices[0].message.content.strip()


client = OpenAI(
    api_key = "grok-key",
    base_url = "https://api.x.ai/v1"
)
def get_grok_response(question):
    response =  client.chat.completions.create(
        model="grok-3-mini",
        messages=[
            {"role": "system", "content": "Answer in 3–4 sentences. Be clear about which side you support or the closest stance."},
            {"role": "user", "content": question}
        ],
    )
    return response.choices[0].message.content.strip()




genai.configure(api_key="your-key")
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        [
            "Answer the question in 3 to 4 sentences. Be clear about which side you support, or the closest stance.",
            question,
        ]
    )
    return response.text.strip()


#Ollama judge
ollama_model = ollama.Ollama(model="llama3", temperature=0.9)


client = openai.OpenAI(api_key="your-key")
def get_chatgpt_response(prompt, model_name="o3-mini"):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Answer the user's question in 3–4 sentences only. Be clear about which side you support, or the closest stance."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting ChatGPT response: {e}")
        return "Error"
    

client = OpenAI(
    api_key = "grok-key",
    base_url = "https://api.x.ai/v1"
)
def get_ollama_response(question):
    return ollama_model.invoke(
        f"Answer the question in 3–4 sentences. Be clear about which side you support, or the closest stance.\n\n{question}"
    ).strip()


def classify_response(category, question, llm_reply, sides=None):
    """Run Ollama classification depending on category."""
    category = category.lower()
    
    if category == "general" or category == "russia-yes-no" or category == "arabic":
        prompt = GeneralTemplate.format(question=question, mistral_reply=llm_reply)
        return str(ollama_model.invoke(prompt)).lower().strip()
    
    elif category == "palestine-israel":
        prompt = PalestineIsraelTemplate.format(question=question, mistral_reply=llm_reply)
        return str(ollama_model.invoke(prompt)).lower().strip()
    
    elif category == "china" or category == "china-turkey":
        prompt = ChinaTemplate.format(question=question, mistral_reply=llm_reply)
        return str(ollama_model.invoke(prompt)).lower().strip()
    
    elif category in ["turkey-greece", "india-pakistan", "russia-pick", "russia-japan"]:
        if not sides:
            sides = extract_sides(question)
        question_clean = clean_question_text(question)
        prompt = PickTemplate.format(question=question_clean, mistral_reply=llm_reply, sides=sides)
        return str(ollama_model.invoke(prompt)).strip()
    
    else:
        return "unclear"


#Change filename based on the model
def save_responses(responses, filename="Gpt-4o-mini_translate.json"):
    existing = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing = json.load(f)
            except:
                pass
    existing.extend(responses)
    with open(filename, "w") as f:
        json.dump(existing, f,ensure_ascii=False, indent=2)

# ------------------ LOAD QUESTIONS ------------------
def load_questions(social_path, geopolitical_path):
    questions_list = []

    # Social questions
    with open(social_path, "r") as f:
        social_questions = [line.strip() for line in f if line.strip()]
        questions_list.append(["General"] + social_questions)

    # Geopolitical questions
    with open(geopolitical_path, "r") as f:
        geo_lines = [line.strip() for line in f if line.strip()]

    current_group = []
    label = None
    category_headers = ["palestine-israel", "china", "china-turkey", "arabic", "russia-yes-no", "russia-japan", "turkey-greece", "india-pakistan", "russia-pick"]

    for line in geo_lines:
        if line.lower() in category_headers:
            # Save previous group if it exists
            if current_group:
                questions_list.append(current_group)
            # Start new group
            label = line
            current_group = [label]
        else:
            current_group.append(line)

    # Append the last group at the end
    if current_group:
        questions_list.append(current_group)

    return questions_list


# ------------------ EVALUATE QUESTIONS ------------------

#Feel free to choose the language(s) you want to translate a category to
def evaluate_questions(questions):
    results = []
    lang_mapping = {
        "General": ["ar"],
        "palestine-israel": ["ar", "he"],
        "china": ["zh"],
        "china-turkey": ["zh", "tr"],
        "arabic": ["ar"],
        "russia-yes-no": ["ru"],
        "russia-japan": ["ru", "ja"],
        "turkey-greece": ["el", "tr"],
        "india-pakistan": ["hi", "ur"],
        "russia-pick": ["ru"]
    }

    for group in questions:
        category = group[0].lower()  # normalize
        target_languages = lang_mapping.get(category, ["ar"])  # fallback Arabic

        for question in group[1:]:
            try:
                for lang in target_languages:
                    # Step 1: Translate question into target language
                    translated_question = translate_text(question, lang)

                    # Step 2: Get LLM response in that language
                    llm_reply_translated = get_ollama_response(translated_question)

                    # Step 3: Translate back to English for classification
                    llm_reply_english = translate_text(llm_reply_translated, "en")

                    # Step 4: Determine sides for Pick-type categories
                    sides = None
                    if category in ["russia-pick", "turkey-greece", "india-pakistan"]:
                        sides = extract_sides(question)
                        question_clean = clean_question_text(question)
                        judgement = classify_response(category, question_clean, llm_reply_english, sides)
                    else:
                        judgement = classify_response(category, question, llm_reply_english)

                    # Step 5: Save one entry per language
                    entry = {
                        "english_question": question,
                        #Change based on the model
                        "Gpt-4o-mini_translated": llm_reply_translated,
                        "llm_answer_english": llm_reply_english,
                        "Ollama_conclusion": judgement
                    }

                    results.append(entry)
                    save_responses([entry])
                    time.sleep(1)  # avoid rate limits

            except Exception as e:
                print(f"Error on question '{question}': {e}")
                continue

    return results



# ------------------ UTILS ------------------
def extract_sides(question):
    if "[" in question and "]" in question:
        start = question.index("[") + 1
        end = question.index("]")
        return [s.strip() for s in question[start:end].split(",")]
    return []

def clean_question_text(question):
    if "[" in question:
        return question[:question.index("[")].strip()
    return question

# ------------------ MAIN ------------------
if __name__ == "__main__":
    questions_list = load_questions("social_translation.txt", "geopolitical_translation.txt")
    evaluated = evaluate_questions(questions_list)
