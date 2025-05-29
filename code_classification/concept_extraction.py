import json
import re
import pandas as pd

# Dizionario che mappa ogni linguaggio alla sua regex per i commenti


def extract_comments(json_list):
    COMMENT_PATTERNS = {
        "python": r"#.*",
        "ruby": r"#.*",
        "javascript": r"//.*",
        "java": r"//.*",
        "go": r"//.*",
        "php": r"(?:#.*|//.*)"
    }
    extracted_comments = []
    # Esempio di json: {"code":"def build_evaluation(variant_specific, variant_id, user_id, user_name,\n                     institute_id, case_id, classification, criteria):\n    \"\"\"Build a evaluation object ready to be inserted to database\n\n    Args:\n        variant_specific(str): md5 string for the specific variant\n        variant_id(str): md5 string for the common variant\n        user_id(str)\n        user_name(str)\n        institute_id(str)\n        case_id(str)\n        classification(str): The ACMG classification\n        criteria(list(dict)): A list of dictionaries with ACMG criterias\n\n    Returns:\n        evaluation_obj(dict): Correctly formatted evaluation object\n\n    \"\"\"\n    criteria = criteria or []\n    evaluation_obj = dict(\n        variant_specific = variant_specific,\n        variant_id = variant_id,\n        institute_id = institute_id,\n        case_id = case_id,\n        classification = classification,\n        user_id = user_id,\n        user_name = user_name,\n        created_at = datetime.datetime.now(),\n    )\n    criteria_objs = []\n    for info in criteria:\n        criteria_obj = {}\n        # This allways has to exist\n        # We might want to check if the term is valid here...\n        criteria_obj['term'] = info['term']\n        if 'comment' in info:\n            criteria_obj['comment'] = info['comment']\n        if 'links' in info:\n            criteria_obj['links'] = info['links']\n        criteria_objs.append(criteria_obj)\n\n    evaluation_obj['criteria'] = criteria_objs\n\n    return evaluation_obj","language":"python"}
    for item in json_list:
        # print(item)
        language = item.get("language", "").lower()
        # print(language)
        text = item.get("code", "")
        lines = text.splitlines()
        # print(text)

        comments = []
        for t in lines:
            # print(f"Processing line: {t}")
            if language in COMMENT_PATTERNS:
                pattern = COMMENT_PATTERNS[language]
                comment = re.findall(pattern, t)
                # print(comment)
                if comment:
                    comments.append(comment)
        for i in range(len(comments)):            
            extracted_comments.append({"language": language, "text": comments[i][0]})

    return extracted_comments

def extract_function_declarations(json_list):
    extracted_functions = []
    FUNCTION_PATTERNS = {
        "python": r"(def\s+\w+\s*\(.*\):)",
        "ruby": r"(def\s+\w+\s*\(.*\))",
        "javascript": r"(function\s+\w+\s*\(.*\))",
        "java": r"((?:\w+\s+)+\w+\s*\(.*\)\s*\{)",
        "go": r"(func\s+\w+\s*\(.*\))",
        "php": r"(function\s+\w+\s*\(.*\))"
    }
    for item in json_list:
        language = item.get("language", "").lower()
        text = item.get("code", "")

        if language in FUNCTION_PATTERNS:
            pattern = FUNCTION_PATTERNS[language]
            functions = re.findall(pattern, text)
            print(functions)
            for func in functions:
                extracted_functions.append({"language": language, "text": func})

    return extracted_functions

# Esempio di input JSON
json_data = pd.read_json("./data/test_dataset.jsonl", lines=True).to_dict(orient='records')
# print(json_data)

# Estrazione commenti
# comments = extract_comments(json_data)
# with open("./data/comments_dataset.jsonl", "w", encoding="utf-8") as f:
#     for comment in comments:
#         f.write(json.dumps(comment, ensure_ascii=False) + "\n")

functions = extract_function_declarations(json_data)
files = {
    "tot": open("./data/function_declarations_dataset.jsonl", "w", encoding="utf-8"),
    "go": open("./data/go_function_declarations_dataset.jsonl", "w", encoding="utf-8"),
    "java": open("./data/java_function_declarations_dataset.jsonl", "w", encoding="utf-8"),
    "javascript": open("./data/javascript_function_declarations_dataset.jsonl", "w", encoding="utf-8"),
    "php": open("./data/php_function_declarations_dataset.jsonl", "w", encoding="utf-8"),
    "python": open("./data/python_function_declarations_dataset.jsonl", "w", encoding="utf-8"),
    "ruby": open("./data/ruby_function_declarations_dataset.jsonl", "w", encoding="utf-8"),
}  


for function in functions:
    files["tot"].write(json.dumps(function, ensure_ascii=False) + "\n")
    lang = function["language"]
    files[lang].write(json.dumps(function, ensure_ascii=False) + "\n")

for x in files.values():
    x.close()