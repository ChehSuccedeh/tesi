#%% 
# Constants
MODEL_PATH = "huggingface/CodeBERTa-language-id"
js_code = """
import mod189 from './mod189';
var value=mod189+1;
export default value;
"""
python_code = """
def ciao():
    end = idk() + random + "return_"
    return
"""
CODE_TO_TEST = python_code
DEBUGGING = True
SEPARATOR = "-"*64
#! pip install transformers, keybert, bertviz

#%% 
# Loading the model and tokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from transformers import TextClassificationPipeline

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)


#%% 
# Base Testing


# not working
# inputs = tokenizer(CODE_TO_TEST)
# print(inputs["input_ids"])
# output = model(inputs)[0]
# language_id = output.argmax()
# print(language_id)

results = pipeline(CODE_TO_TEST)
print(results)
#%% 
# trying keyBert
from keybert import KeyBERT

if not DEBUGGING:
    kb_model = KeyBERT(model=model)
    kb_model.extract_keywords(CODE_TO_TEST, highlight=True)
#%%
# trying Bertviz

from bertviz import head_view, model_view, neuron_view

inputs = tokenizer.encode(CODE_TO_TEST, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]
tokens = tokenizer.convert_ids_to_tokens(inputs[0])
print(tokenizer.decode(inputs[0]))

if( not DEBUGGING):
    head_view(attention, tokens)
    model_view(attention, tokens)

# neuron view is not working (new(): str type not accepted for attention number)
# neuron_view.show(model=model,model_type="bert", tokenizer=tokenizer, sentence_a=CODE_TO_TEST)

# %%
# Trying LIME
import lime, torch
from lime.lime_text import LimeTextExplainer
class_names = [
    "go",
    "java",
    "javascript",
    "php",
    "python",
    "ruby",
]

def prediction(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    tensor_logits = outputs[0]
    # print(tensor_logits)
    probs = torch.nn.functional.softmax(tensor_logits).detach().numpy()
    # print(probs)
    return probs

# print(tokenizer(CODE_TO_TEST, return_tensors='pt', padding=True))

explainer = LimeTextExplainer(class_names=class_names)

explaination = explainer.explain_instance(CODE_TO_TEST, prediction, num_features=15, num_samples=100, top_labels=1)

important_words = explaination.as_list(label=explaination.top_labels[0])
# print(important_words)
explaination.show_in_notebook(text=CODE_TO_TEST)

#%%
# Load tree_sitter
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo
import tree_sitter_ruby as tsruby
import tree_sitter_php as tsphp
import tree_sitter_java as tsjava

from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
GO_LANGUAGE = Language(tsgo.language())
JS_LANGUAGE = Language(tsjavascript.language())
RB_LANGUAGE = Language(tsruby.language())
# JAVA_LANGUAGE = Language(tsjava.language())
PHP_LANGUAGE = Language(tsphp.language_php_only())

from operator import itemgetter
results = sorted(results[0], key=itemgetter("score"), reverse=True)

print(results)
#%%
# Select language of the parser
def select_language(language):
    match language:
        case "python":
            output = PY_LANGUAGE
            print("Python parser selected")
        case "javascript":
            output = JS_LANGUAGE
            print("JavaScript parser selected")
        case "go":
            output = GO_LANGUAGE
            print("Go parser selected")
        case "ruby":
            output = RB_LANGUAGE
            print("Ruby parser selected")
        case _:
            raise ValueError("Language not supported")
    return output
        
# Assume correct label
guess_language = select_language(results[0]["label"])
parser = Parser(guess_language)
#%%
# Parse the code, look for errors
guess_tree = parser.parse(bytes(CODE_TO_TEST, "utf8"))
print(guess_tree.root_node)



#%%
# Print the tree
def printAST(node, indent=0):
    prefix = "\t" * indent
    text = node.text.decode("utf-8").replace("\n", "\\n")
    print(f"{prefix}Node type: {node.type} | Node text: {text}")
    # print(f"{prefix}Node start position: {node.start_point}")
    # print(f"{prefix}Node end position: {node.end_point}")
    # print(f"{prefix}Node children: {len(node.children)}")

    for child in node.children:
        printAST(child, indent + 1)

def find_errors(node, words, error_found= False):
    if node.type == "ERROR":
        if error_found:
            print(SEPARATOR)
        text = node.text.decode("utf-8").replace("\n", "\\n")
        print(f"Error found at {node.start_point} - {node.end_point}: \"{text}\"")
        error_found = True
        for word in words:
            if word[0] in node.text.decode("utf-8"):
                print(f"Important word found: {word[0]} - LIME value: {word[1]}")
    for child in node.children:
        error_found = find_errors(child, words, error_found)
    return error_found
# Print whole tree     
if DEBUGGING:
    print("Printing AST")
    printAST(guess_tree.root_node)

# Print only errors
print(SEPARATOR)    
print("Finding errors:")
has_errors = find_errors(guess_tree.root_node, important_words)
print(has_errors)
#%%
# Find correct language
def iterate_languages(attempts):
    for guess in attempts:
        parser = Parser(select_language(guess["label"]))
        tree = parser.parse(bytes(CODE_TO_TEST, "utf8"))
        
        print(f"Trying {guess['label']} language:")
        if DEBUGGING:
            printAST(tree.root_node)
        found = find_errors(tree.root_node, important_words)
        if not found:
            print(f"No errors found, {guess['label']} is the correct language")
            return tree, guess["label"]
        print(SEPARATOR)

if not has_errors:
    print("No errors found, guess was correct")
else:
    correct_tree, correct_language = iterate_languages(results[1:])

# %%
# Get all error nodes with associated important words
def get_errors(node, words):
    errors = []
    if node.type == "ERROR":
        for child in node.children:
            error = {"text": child.text.decode("utf-8"), "start": child.start_point, "end": child.end_point, "important_words": [], "node": child}
            for word in words:
                if word[0] in child.text.decode("utf-8"):
                    error["important_words"].append(word)
            errors.append(error)
    else:
        for child in node.children:
            errors += get_errors(child, words)
    return errors
if has_errors:
    error_nodes = get_errors(guess_tree.root_node, important_words)
    print(error_nodes)
#%%
# Get nodes with important words in the correct language tree
def get_nodes_with_important_words(node, words):
    nodes = []
    to_append = {
                "text": node.text.decode("utf-8"),
                "start": node.start_point,
                "end": node.end_point,
                "important_words": [],
                "node": node
    }
    for word in words:
        if word in node.text.decode("utf-8"):
            to_append["important_words"].append(word)
    
    if len(to_append["important_words"]) > 0:
        nodes.append(to_append)
    for child in node.children:
        nodes += get_nodes_with_important_words(child, words)
    return nodes

if has_errors:
    error_words = []
    for x in error_nodes:
        error_words.append(x["text"])

    correct_nodes = get_nodes_with_important_words(correct_tree.root_node, error_words)
    for w in correct_nodes:
        print(w)
#%%
# Eliminate bigger blocks from tree

if has_errors:    
    correspondence = []
    for nodex in error_nodes:
        found = False
        for nodey in correct_nodes:
            if nodex["text"] == nodey["text"]:
                found = True
                correspondence.append({"correct_language": nodey, "wrong_language": nodex})
                break
        if not found:
            correspondence.append({"correct_language": None, "wrong_language": nodex, "error": True})

    for x in correspondence:
        if x["correct_language"] is not None:
            print(f"Found Correspondence for error \"{x['wrong_language']['text']}\": Node type in correct language is {x['correct_language']['node'].type}")
        else:
            print(f"Missing correspondence for: {x['wrong_language']['text']} - {x['wrong_language']['start']} - {x['wrong_language']['end']}")

        print(SEPARATOR)



# %%
# Check if in the wrong language there are node kind compatible with the correct language
if has_errors:
    for i in range(1, guess_language.node_kind_count):
        if guess_language.node_kind_for_id(i) in [n["correct_language"]["node"].type for n in correspondence]:
            print(f"Node kind {guess_language.node_kind_for_id(i)} exists also in wrong language")
            

# %%
# Check if within errors there are some keywords

def check_keywords(errors, keywords):
    for error in errors:
        if error in keywords:
            print(f"Keyword found: {error}")
        else:
            print(f"Not a keyword: {error}")
if has_errors:
    keywords = open(f"./keywords/{results[0]['label']}.txt", "r").readlines()
    keywords = [x.strip() for x in keywords]
    # print(keywords)

    check_keywords([n["correct_language"]["text"] for n in correspondence], keywords)
#%%
# Check other errors to analyse manually

def check_other_errors(errors, keywords):
    for error in errors:
        if error["text"] not in keywords:
            print(f"Error to analyse: {error['text']}")

if has_errors:            
    check_other_errors([n["wrong_language"] for n in correspondence], keywords)
# %%
# Insert keywords in other code languages