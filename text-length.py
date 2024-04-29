import json
from collections import defaultdict

question_lengths = defaultdict(int)

with open('data_RAD/trainset.json', 'r') as file:
    data = json.load(file)
    for item in data:
        question = item['question']
        question_length = len(question.split())
        question_lengths[question_length] += 1

for length, count in sorted(question_lengths.items()):
    print(f'Question length {length} has {count} instances.')
