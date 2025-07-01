import pandas as pd

test_df = pd.DataFrame()
NUM_SAMPLES = 1000
##### insert GO data #####
df = pd.read_json("./datasets/codesearch_data/go_test_0.jsonl/go_test_0.jsonl", lines=True)
df.drop(['repo', 'path', 'func_name', 'code',
       'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url',
       'partition'], axis=1, inplace=True)
df.rename(columns={'original_string': 'code'}, inplace=True)
# print(df.columns)
tmp = df.sample(n=NUM_SAMPLES, random_state=42)
test_df = pd.concat([test_df, tmp], ignore_index=True)

##### insert JAVA data #####
df = pd.read_json("./datasets/codesearch_data/java_test_0.jsonl/java_test_0.jsonl", lines=True)
df.drop(['repo', 'path', 'func_name', 'code',
       'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url',
       'partition'], axis=1, inplace=True)
df.rename(columns={'original_string': 'code'}, inplace=True)
tmp = df.sample(n=NUM_SAMPLES, random_state=42)
test_df = pd.concat([test_df, tmp], ignore_index=True)

##### insert JS data #####
df = pd.read_json("./datasets/codesearch_data/javascript_test_0.jsonl/javascript_test_0.jsonl", lines=True)
df.drop(['repo', 'path', 'func_name', 'code',
       'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url',
       'partition'], axis=1, inplace=True)
df.rename(columns={'original_string': 'code'}, inplace=True)
tmp = df.sample(n=NUM_SAMPLES, random_state=42)
test_df = pd.concat([test_df, tmp], ignore_index=True)

##### insert PHP data #####
df = pd.read_json("./datasets/codesearch_data/php_test_0.jsonl/php_test_0.jsonl", lines=True)
df.drop(['repo', 'path', 'func_name', 'code',
       'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url',
       'partition'], axis=1, inplace=True)
df.rename(columns={'original_string': 'code'}, inplace=True)
tmp = df.sample(n=NUM_SAMPLES, random_state=42)
test_df = pd.concat([test_df, tmp], ignore_index=True)

##### insert PY data #####
df = pd.read_json("./datasets/codesearch_data/python_test_0.jsonl/python_test_0.jsonl", lines=True)
df.drop(['repo', 'path', 'func_name', 'code',
       'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url',
       'partition'], axis=1, inplace=True)
df.rename(columns={'original_string': 'code'}, inplace=True)
tmp = df.sample(n=NUM_SAMPLES, random_state=42)
test_df = pd.concat([test_df, tmp], ignore_index=True)

##### insert RB data #####
df = pd.read_json("./datasets/codesearch_data/ruby_test_0.jsonl/ruby_test_0.jsonl", lines=True)
df.drop(['repo', 'path', 'func_name', 'code',
       'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url',
       'partition'], axis=1, inplace=True)
df.rename(columns={'original_string': 'code'}, inplace=True)
tmp = df.sample(n=NUM_SAMPLES, random_state=42)
test_df = pd.concat([test_df, tmp], ignore_index=True)

# print(test_df.columns)

##### Scramble test data #####
import random

test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

test_df.to_json("./code_classification/data/test2_dataset.jsonl", orient="records", lines=True)