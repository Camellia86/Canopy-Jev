# SPDX-License-Identifier: MIT
"""Explicit zero-shot and legacy prompts; target fields never enter templates."""
import json


def clean_questions(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions must be a nonempty list')
    result = []
    for i, question in enumerate(questions):
        kind = question.get('type', 'choice')
        instruction = question.get('instructions', question.get('question'))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Each question needs nonempty instructions or question text')
        if kind == 'choice':
            if 'criteria' in question:
                criteria = question['criteria']
                if not isinstance(criteria, dict) or not criteria:
                    raise ValueError('Choice criteria must be a nonempty ordered mapping')
                keys = list(criteria)
                options = [str(k) + ': ' + criteria[k] for k in keys]
            else:
                options = question.get('options')
                keys = question.get('keys', [str(j) for j in range(len(options))] if isinstance(options, list) else [])
        elif kind == 'noul':
            criteria = question.get('criteria') or {}
            if not isinstance(criteria, dict):
                raise ValueError('Noul criteria must map false/true to descriptions')
            keys = ['no', 'yes']
            options = ['No: ' + criteria.get('false', 'The condition is false.'),
                       'Yes: ' + criteria.get('true', 'The condition is true.')]
        elif kind == 'score':
            levels = question.get('criteria', question.get('levels'))
            if not isinstance(levels, list) or not all(isinstance(x, str) and x.strip() for x in levels):
                raise ValueError('Score requires an ordered list of nonempty level descriptions')
            keys = [str(j) for j in range(len(levels))]
            options = [str(j) + ': ' + text for j, text in enumerate(levels)]
        else:
            raise ValueError('Supported types: choice, noul, score (candidate adaptations)')
        if not isinstance(options, list) or not 1 <= len(options) <= 10 or not all(isinstance(x, str) and x.strip() for x in options):
            raise ValueError('Exactly 1..10 nonempty string candidates are required')
        if not isinstance(keys, list) or len(keys) != len(options) or not all(isinstance(k, str) and k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('Each candidate needs a unique nonempty string key')
        result.append({'id': str(question.get('id', i)), 'type': kind, 'instructions': instruction,
                       'options': list(options), 'keys': list(keys), 'category': str(question.get('category', 'other'))})
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('Question IDs must be unique')
    return result


def zero_shot_prompt(tokenizer, state, row):
    # Exact measured JevBench adapter text and option-boundary prefix rule.
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    body = 'Application state:\n' + state + '\n\nQuestion:\n' + row['instructions'] + '\nOptions:\n'
    options_start = len(body)
    body += ''.join(f'{chr(65+i)}. {option}\n' for i, option in enumerate(row['options']))
    body += 'Answer: '
    text = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'Choose the correct option according to the question and its criteria. Return only its letter; do not write a reasoning trace.'},
        {'role': 'user', 'content': body}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text += 'The answer is ('
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offset = text.index(body.rstrip()) + options_start
    options_token = next(i for i, (_, end) in enumerate(encoded['offset_mapping']) if end > offset)
    return list(encoded['input_ids']), options_token


def common_prefix(prompts):
    cap = min(p[1] for p in prompts)
    for i in range(cap):
        if any(p[0][i] != prompts[0][0][i] for p in prompts[1:]):
            return i
    return cap


def prepare(tokenizer, state, questions, template='zero_shot_choice', demonstrations=None):
    rows = clean_questions(questions)
    if template == 'zero_shot_choice':
        if demonstrations is not None:
            raise ValueError('Zero-shot mode cannot inject demonstrations')
        prompts = [zero_shot_prompt(tokenizer, state, r) for r in rows]
        shared = common_prefix(prompts)
    elif template == 'legacy_five_demo':
        if not demonstrations or len(demonstrations) < 5:
            raise ValueError('Legacy mode requires the separately supplied, verified demonstration file')
        contextual = ContextTokenizer(tokenizer, state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
        prompts = [(_legacy_make_prompt(contextual, {'id': r['id'], 'question': r['instructions'],
                    'options': r['options'], 'category': r['category'], 'answer_index': 0}, demonstrations)[0], 0) for r in rows]
        cap = min(len(p[0]) for p in prompts) - 1
        shared = next((i for i in range(cap) if any(p[0][i] != prompts[0][0][i] for p in prompts[1:])), cap)
    else:
        raise ValueError('Unknown prompt template')
    if shared < 1:
        raise ValueError('No nonempty exact token prefix')
    prefix = prompts[0][0][:shared]
    branches = [p[0][shared:] for p in prompts]
    return rows, prefix, branches


import random
import hashlib

class ContextTokenizer:
    """Put state in a common chat turn before category-specific question bodies."""
    def __init__(self,tokenizer,state):
        self.tokenizer,self.state = tokenizer,state

    def __getattr__(self,name):
        return getattr(self.tokenizer,name)

    def __call__(self,*args,**kwargs):
        return self.tokenizer(*args,**kwargs)

    def apply_chat_template(self,messages,**kwargs):
        if self.state:
            messages = [messages[0],{"role":"user","content":"Shared state:\n"+self.state},
                        {"role":"assistant","content":"I will use the shared state when answering."},*messages[1:]]
        return self.tokenizer.apply_chat_template(messages,**kwargs)

def _legacy_make_prompt(tokenizer, row, demonstrations, train=False):
    rng = random.Random(hashlib.sha256(row["id"].encode()).hexdigest())
    order = list(range(len(row["options"])))
    if train:
        rng.shuffle(order)
    # Sparse labels expose all A..J during training without inventing false distractors.
    labels = sorted(rng.sample(range(10), len(order))) if train and rng.random() < .5 else list(range(len(order)))
    examples = [x for x in demonstrations if x["category"] == row["category"]][:5]
    if len(examples) != 5:
        examples = demonstrations[:5]
    body = "The following are multiple choice questions (with answers) about " + row["category"] + ".\n\n"
    for x in examples:
        body += "Question: " + x["question"] + "\nOptions:\n"
        body += "".join(f"{chr(65+i)}. {opt}\n" for i, opt in enumerate(x["options"]))
        cot = x.get("cot_content", "").removeprefix("A: ")
        body += "Answer: " + (cot or f"The answer is ({chr(65+x['answer_index'])}).") + "\n\n"
    body += "Question: " + row["question"] + "\nOptions:\n"
    spans = []
    for label, original in zip(labels, order):
        body += f"{chr(65+label)}. "
        start = len(body)
        body += row["options"][original]
        spans.append((start, len(body)))
        body += "\n"
    body += "Answer: "
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": "Choose the correct option. Return only its letter; do not write a reasoning trace."},
        {"role": "user", "content": body}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # A fixed answer preamble is input, not a generated token or a sampled answer.
    text += "The answer is ("
    body_offset = text.index(body.rstrip())
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    token_spans = []
    for a, b in spans:
        ids = [i for i, (s, e) in enumerate(offsets) if e > body_offset + a and s < body_offset + b and e > s]
        if not ids:
            raise ValueError("Empty option span")
        token_spans.append((ids[0], ids[-1] + 1))
    target = order.index(row["answer_index"])
    return encoded["input_ids"], token_spans, labels, target, order

