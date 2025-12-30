# LLM Political Stance and Consistency

This repository contains the code, prompts, and experimental artifacts used to study **political stance and consistency in Large Language Models (LLMs)**, as described in the paper:

> **Measuring Political Stance and Consistency in Large Language Models**  
> *(To appear at ECIR 2026)*

The project systematically evaluates how different LLMs respond to politically sensitive questions and how model stances change under different prompting strategies, including **argument framing**, **question reformulation**, and **language variation**.

---

## Paper Overview

Large Language Models are increasingly used for information-seeking, including on political and geopolitical topics where disagreement is common. However, LLMs may reflect biases from training data or alignment choices, and their responses can shift depending on how questions are asked.

This work evaluates:

- **9 LLMs**
- **24 political and social issues**
- **5 prompting strategies**
- **3 paraphrases per prompt**
- **Multiple languages** for region-specific conflicts

The goal is to measure:

- Whether LLMs adopt political stances  
- How consistent those stances are  
- Which prompting methods are most effective at shifting model outputs  

---

## Models Evaluated

The experiments include the following models:

- GPT-5  
- GPT-4o  
- GPT-4o-mini  
- o3-mini  
- Gemini-2.5-Flash
- Grok-3-mini  
- Mistral-7B  
- Ollama-8B  
- DeepSeek-Chat

Some models are accessed via official APIs, while others are run locally.

## Repository Structure
``` 
LLM-Political-Stance/
│
├── Codes/
│   ├── Scripts for querying models
│   ├── Stance classification logic
│   ├── Experiment orchestration
|   └── MainExperiment.py is used for all experiments except translation
│
├── Contrast_Experiment/
│   └── Experiments testing stance shifts under contrasting prompts
│
├── DirectAndOppositeArguments/
|   ├── Experiments testing prompts with no arguments (direct),
│   └── Arguments supporting one side, arguments of the other side, and arguments of both sides at once
│
├── Translation_Answers/
│   └── Prompts and responses in multiple languages
│
├── DataSummary.xlsx
│   └── Aggregated results and stance summaries
│
└── README.md
```

## Prompting Strategies

The repository implements the five prompting strategies used in the paper:

- **Direct Questions**  
  Models are asked for their stance without additional context.

- **Providing Opposite Argument**  
  A short argument opposing the model’s initial stance is prepended.

- **Providing Arguments for Both Sides**  
  Brief arguments for both positions are included before asking the question.

- **Changing Question Formulation**  
  The same issue is asked using logically inverted or reframed questions.

- **Prompting in Different Languages**  
  Prompts are translated into the primary languages of the countries involved in each issue.

Each prompt is paraphrased multiple times to reduce randomness, and final stances are determined via **majority vote**.

---

## Political Issues Covered

The study covers **24 political and social issues**, including:

- Israel–Palestine conflict  
- Russia–Ukraine war  
- China’s policies toward Uyghurs and Tibet  
- Aegean Islands dispute (Turkey–Greece)  
- Kashmir dispute (India–Pakistan)  
- Arab Spring  
- Qatar blockade  
- Women’s rights, children’s autonomy, and religion in politics  

A full list of issues is available in the paper and summarized in **`DataSummary.xlsx`**.

---

## Stance Classification

- One LLM is used to **generate responses**
- A second LLM acts as a **stance classifier**
- An additional external model is used to **audit classification reliability**

Manual validation on a random sample confirms **high classification accuracy**.

---

## Note

The **`DirectAndOppositeArguments`** folder has Json files structured as follows:
- Each 9 questions consecutive entries starting from the top of the will have 3 paraphrases for the argument of the first side
- Then there will be 3 paraphrases for the argument of the second side
- Lastly there will be 3 paraphrases for both arguments at once
- This ends at line 1171 in each file, after which there will be the direct questions where each one has 3 paraphrases
- Each question is asked a total of 12 times

## Key Findings (High Level)

- LLMs almost always **pick a side** on political issues
- Some stances are **highly persistent**, while others are easily shifted
- **Language choice** is one of the strongest factors influencing stance
- Models developed in different regions may reflect **regional political alignment**
- Certain issues (e.g., oppression of Palestinians, Qatar blockade) remain **stable across all prompting strategies**

---

## Reproducibility

This repository is designed to support reproducibility by:

- Sharing prompts and paraphrases
- Providing stance aggregation logic
- Including summarized experimental results

Exact API keys, rate limits, and proprietary model access are **not included**.




