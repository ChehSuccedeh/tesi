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

from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
GO_LANGUAGE = Language(tsgo.language())
JS_LANGUAGE = Language(tsjavascript.language())
RB_LANGUAGE = Language(tsruby.language())

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
language = select_language(results[0]["label"])
parser = Parser(language)
#%%
# Parse the code, look for errors
tree = parser.parse(bytes(CODE_TO_TEST, "utf8"))
print(tree.root_node)



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
    print(SEPARATOR)
    print("Printing AST")
    printAST(tree.root_node)

# Print only errors
print(SEPARATOR)    
print("Finding errors:")
has_errors = find_errors(tree.root_node, important_words)
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
            break
        print(SEPARATOR)

if not has_errors:
    print("No errors found, guess was correct")
else:
    iterate_languages(results[1:])

# %%
