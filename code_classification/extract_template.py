BASE_LANGUAGE = "javascript"

CODE = """function calcolaOperazioni(base, modificatore) {
    return {
        somma: base + modificatore,
        differenza: base - modificatore,
        prodotto: base * modificatore,
        divisione: base / modificatore,
        potenza: Math.pow(base, modificatore),
        radice: Math.sqrt(modificatore)
    };
}"""

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo
import tree_sitter_ruby as tsruby
import tree_sitter_java as tsjava
import tree_sitter_php as tsphp

from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
GO_LANGUAGE = Language(tsgo.language())
JS_LANGUAGE = Language(tsjavascript.language())
RB_LANGUAGE = Language(tsruby.language())
JAVA_LANGUAGE = Language(tsjava.language())
PHP_LANGUAGE = Language(tsphp.language_php_only())

language = ""
match BASE_LANGUAGE:
    case "python":
        language = PY_LANGUAGE
        print("Python parser selected")
    case "javascript":
        language = JS_LANGUAGE
        print("JavaScript parser selected")
    case "go":
        language = GO_LANGUAGE
        print("Go parser selected")
    case "ruby":
        language = RB_LANGUAGE
        print("Ruby parser selected")
    case "java":
        language = JAVA_LANGUAGE
        print("Java parser selected")
    case "php":
        language = PHP_LANGUAGE
        print("PHP parser selected")
    case _:
        raise ValueError("Language not supported")

parser = Parser(language)

# Parse the code and extract node types

def extract_node_types(node, indent=0):
    string = ""
    prefix = "\t" * indent
    text = node.text.decode("utf-8").replace("\n", "\\n")
    string = f"{prefix}Node type: {node.type} | Node text: {text}\n"

    for child in node.children:
        string += extract_node_types(child, indent + 1)
    return string

tree = parser.parse(bytes(CODE, "utf8"))

out = extract_node_types(tree.root_node)
print(out)
f = open(f"./code_classification/templates/{BASE_LANGUAGE}_template.txt", "w")
f.write(out)
f.close()