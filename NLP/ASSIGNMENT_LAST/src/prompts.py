BASE_PROMPT = """
Select appropriate answer for a given question from the list of options.
As answer respond only with the correct option letter (A, B, etc.).
Question: {question}
Options: {options}
Answer:
"""

RAG_PROMPT = """
You are given a question and a list of context passages. Use the context passages to answer the question.
As answer respond only with the correct option letter (A, B, etc.).
Question: {question}
Context Passages:
{context}
Answer:
"""

FEW_SHOT_PROMPT = """
Select appropriate answer for a given question from the list of options.
As answer respond only with the correct option letter (A, B, etc.).

1. Question: {question1}
   Options: {options1}
   Answer: {answer1}

2. Question: {question2}
   Options: {options2}
   Answer: {answer2}

Now, answer the question:
Question: {question}
Options: {options}
Answer:
"""

COT_PROMPT = """
Select appropriate answer for a given question from the list of options.
Think step by step and explain your reasoning before giving the final answer.
As answer respond with json object containing \"reasoning\" and \"answer\" fields.
JSON object format:
\{
  \"reasoning\": \"<your step by step reasoning>\",
  \"answer\": \"<the correct option letter (A, B, etc.)>\"
\}


Question: {question}
Options: {options}
Answer:
"""