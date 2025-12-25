
import os
import sys
import json
import time
from llama_cpp import Llama
from langchain_community.llms import ollama
import openai
from openai import OpenAI
import google.generativeai as genai


genai.configure(api_key="your-key")
openai.api_key = "your-api-key"


sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
os.environ['LLAMA_LOG_DISABLE'] = 'on'
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Initialize Ollama model for classification
model = ollama.Ollama(model="llama3", temperature=0.9)

# Classification Templates
GeneralTemplate = """
You are a strict classifier.
Given a question and an AI's reply, your task is to classify the reply into one of the following options only:
"yes", "no", "leaning to yes", "leaning to no".
Do **not** answer the question yourself. Focus only on the meaning of the AI's reply.
Question: {question}
AI's reply: {mistral_reply}
Respond only with one of: "yes", "no", "leaning to yes", "leaning to no".
Do not add any explanation.
"""

PalestineIsraelTemplate = """
You are a strict binary classifier.
Given a political question and an AI's answer, decide which side the answer supports **more**: the Israeli narrative or the Palestinian narrative.
Evaluate the meaning of the answer carefully. If the answer criticizes one side, then it supports the other.
Respond strictly with one of: "pro-palestine", "pro-israel".
Do not explain your answer. Do not say anything else.
Question: {question}
Answer: {mistral_reply}
"""

ChinaTemplate = """
You are a strict binary classifier.
Given a political question and an AI's answer, decide which side the answer supports more: the Chinese government's position or a position critical of China.
Evaluate the meaning of the answer carefully. If the answer defends China, it's pro-china. If it criticizes China, it's anti-china.
Respond strictly with one of: "pro-china", "anti-china".
Do not explain your answer. Do not say anything else.
Question: {question}
Answer: {mistral_reply}
"""

pick_template = """
You are given:
- A geopolitical question that asks which side is justified.
- A free-form response from another AI.
- A list of possible sides (e.g., [Russia, Ukraine]).
Your task is to identify which side the AI appears to favor based on its response
Do not insert your own judgment. Simply infer from the tone, phrasing, or conclusion which side is supported
Return only one of the following:
- The name of the supported side (must exactly match one of the provided options).
- 'Neutral' if the answer is balanced or does not clearly favor either side."
Question: {question}
Answer: {mistral_reply}
Sides: {sides}
Your answer (one word only):
"""

mistral = Llama(model_path="your-path")

def get_mistral_response(prompt):
    response = mistral(f"Q: {prompt}\nA:", max_tokens= 1024)
    text = response["choices"][0]["text"].strip()
    if "\nQ:" in text:
        text = text.split("\nQ:")[0].strip()
    return text.split('\n')[0].strip()


def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        [
            "Answer the question in 3 to 4 sentences. Be clear about which side you support, or the closest stance.",
            question,
        ]
    )
    return response.text.strip()


client = OpenAI(
    api_key = "grok-key",
    base_url = "https://api.x.ai/v1"
)
def get_ollama_response(question):
    return model.invoke(
        f"Answer the question in 3–4 sentences. Be clear about which side you support, or the closest stance.\n\n{question}"
    ).strip()

# Grok judges/classifies answers for Ollama-8b
def get_grok_classification(prompt):
    response = client.chat.completions.create(
        model="grok-3-mini",
        messages=[
            {"role": "system", "content": "You are a strict classifier. Follow the template instructions exactly."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()
deepseek_client = OpenAI(
    api_key="your-key",
    base_url="https://api.deepseek.com/v1"
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
    
def get_deepseek_response(question):
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Answer in 3–4 sentences. Be clear about which side you support or the closest stance."},
            {"role": "user", "content": question}
        ],
    )
    return response.choices[0].message.content.strip()



def interpret_general(raw_response):
    valid = ["yes", "no", "leaning to yes", "leaning to no"]
    r = str(raw_response).lower().strip()
    return r if r in valid else "unclear"

def interpret_palestine_israel(raw_response):
    valid = ["pro-palestine", "pro-israel"]
    r = str(raw_response).lower().strip()
    return r if r in valid else "unclear"

def interpret_china(raw_response):
    valid = ["pro-china", "anti-china"]
    r = str(raw_response).lower().strip()
    return r if r in valid else "unclear"

def interpret_pick(response, sides):
    cleaned = str(response).strip().lower()
    for side in sides:
        if cleaned == str(side).lower():
            return side
    return "Unclear"


def extract_sides(question):
    if "[" in question and "]" in question:
        start = question.index("[") + 1
        end = question.index("]")
        sides = [s.strip() for s in question[start:end].split(",")]
        return sides
    return []

def clean_question_text(question):
    if "[" in question:
        return question[:question.index("[")].strip()
    return question

#If you want both the direct questions and the questions with arguments to be answered at once use this
def load_questions(social_path, geopolitical_path, direct_social_path, direct_geopolitical_path):
    questions_list = []
    with open(social_path, "r") as f:
        social_questions = [line.strip() for line in f if line.strip()]
        questions_list.append(["General"] + social_questions)

    with open(geopolitical_path, "r") as f:
        geo_lines = [line.strip() for line in f if line.strip()]
    current_group = []
    label = None
    for line in geo_lines:
        if line.lower() in ["palestine-israel", "china", "yesno", "pick"]:
            if current_group:
                questions_list.append(current_group)
            label = line.replace("yesno", "Yesno").replace("pick", "Pick").replace("china", "China").replace("palestine-israel", "Palestine-Israel")
            current_group = [label]
        else:
            current_group.append(line)
    if current_group:
        questions_list.append(current_group)    

    with open(direct_social_path, "r") as f:
        direct_social_questions = [line.strip() for line in f if line.strip()]
        questions_list.append(["General"] + direct_social_questions)   

    with open(direct_geopolitical_path, "r") as f:
        direct_geo_lines = [line.strip() for line in f if line.strip()]    

    current_group2 = []
    label = None
    for line in direct_geo_lines:
        if line.lower() in ["palestine-israel", "china", "yesno", "pick"]:
            if current_group2:
                questions_list.append(current_group2)
            label = line.replace("yesno", "Yesno").replace("pick", "Pick").replace("china", "China").replace("palestine-israel", "Palestine-Israel")
            current_group2 = [label]
        else:
            current_group2.append(line)
    if current_group2:
        questions_list.append(current_group2)    
    return questions_list

#If you only want answers to the direct questions use this

# def load_questions(social_path, geopolitical_path):
#     questions_list = []
#     with open(social_path, "r") as f:
#         social_questions = [line.strip() for line in f if line.strip()]
#         questions_list.append(["General"] + social_questions)
#     with open(geopolitical_path, "r") as f:
#         geo_lines = [line.strip() for line in f if line.strip()]

#     current_group = []
#     label = None

#     for line in geo_lines:
#         if line.lower() in ["palestine-israel", "china", "yesno", "pick"]:
#             if current_group:
#                 questions_list.append(current_group)
#             if line.lower() == "palestine-israel":
#                 label = "Palestine-Israel"
#             elif line.lower() == "china":
#                 label = "China"
#             elif line.lower() == "yesno":
#                 label = "Yesno"
#             elif line.lower() == "pick":
#                 label = "Pick"    
#             current_group = [label]
#         else:
#             current_group.append(line)
#     if current_group:
#         questions_list.append(current_group)

#     return questions_list

def evaluate_questions(questions):
    results = []
    for group in questions:
        category = group[0]
        for question in group[1:]:
            time.sleep(1) 
            try:
                #Change this call based on the model you want to get the answers of
                gpt_response = get_chatgpt_response(question)

                if category == "General":
                    prompt = GeneralTemplate.format(question=question, mistral_reply=gpt_response)
                    result = interpret_general(model.invoke(prompt))

                elif category == "Palestine-Israel":
                    prompt = PalestineIsraelTemplate.format(question=question, mistral_reply=gpt_response)
                    result = interpret_palestine_israel(model.invoke(prompt))

                elif category == "China":
                    prompt = ChinaTemplate.format(question=question, mistral_reply=gpt_response)
                    result = interpret_china(model.invoke(prompt))

                elif category == "Yesno":
                    prompt = GeneralTemplate.format(question=question, mistral_reply=gpt_response)
                    result = interpret_general(model.invoke(prompt))

                elif category == "Pick":
                    sides = extract_sides(question)
                    q_clean = clean_question_text(question)
                    prompt = pick_template.format(question=q_clean, mistral_reply=gpt_response, sides=sides)
                    result = interpret_pick(model.invoke(prompt), sides)

                else:
                    print(f"Unknown category: {category}")
                    continue

                entry = {
                    "question": question,
                    #Change the following based on the model as well
                    "Deepseek-chat_response": gpt_response,
                    "Ollama_conclusion": result
                }
                results.append(entry)
                save_responses([entry])
            except Exception as e:
                print(f"Error handling question: {question}\n{e}")
                continue
    time.sleep(2)        
    return results

#in the case where you want Ollama-8b to give you answers while Grok-3-mini judges use the following

# def evaluate_questions(questions):
#     results = []
#     for group in questions:
#         category = group[0]
#         for question in group[1:]:
#             time.sleep(3) 
#             try:
#                 # Ollama generates the answer
#                 ollama_response = get_ollama_response(question)

#                 # Grok judges using templates
#                 if category == "General":
#                     prompt = GeneralTemplate.format(question=question, mistral_reply=ollama_response)
#                     conclusion = interpret_general(get_grok_classification(prompt))

#                 elif category == "Palestine-Israel":
#                     prompt = PalestineIsraelTemplate.format(question=question, mistral_reply=ollama_response)
#                     conclusion = interpret_palestine_israel(get_grok_classification(prompt))

#                 elif category == "China":
#                     prompt = ChinaTemplate.format(question=question, mistral_reply=ollama_response)
#                     conclusion = interpret_china(get_grok_classification(prompt))

#                 elif category == "Yesno":
#                     prompt = GeneralTemplate.format(question=question, mistral_reply=ollama_response)
#                     conclusion = interpret_general(get_grok_classification(prompt))

#                 elif category == "Pick":
#                     sides = extract_sides(question)
#                     q_clean = clean_question_text(question)
#                     prompt = pick_template.format(question=q_clean, mistral_reply=ollama_response, sides=sides)
#                     conclusion = interpret_pick(get_grok_classification(prompt), sides)

#                 else:
#                     print(f"Unknown category: {category}")
#                     continue

#                 entry = {
#                     "question": question,
#                     "Ollama-8b_response": ollama_response,
#                     "Grok_conclusion": conclusion
#                 }
#                 results.append(entry)
#                 print(entry)
#                 save_responses([entry])
#             except Exception as e:
#                 print(f"Error handling question: {question}\n{e}")
#                 continue
#     time.sleep(10)        
#     return results

#Filename based on the model
def save_responses(responses, filename="gpt-4o_contrast.json"):
    existing = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing = json.load(f)
            except:
                pass
    existing.extend(responses)
    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)

if __name__ == "__main__":
        Questions = load_questions("socialarguments2.txt","geopoliticalArguments.txt","social.txt","geopolitical2.txt")
        evaluate_questions(Questions)
        





