import torch
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import json

from src.util import get_time_remaining_formatted, get_device

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/evaluations/input')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/evaluations/output')

MODEL = 'gpt-4o'
FILE_NAME = f"{MODEL}_eval_input.jsonl"
BATCH_ID = 'batch_679a82d2f1ac8190b38d0d4da50b635e'

def load_api():
  env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.env')
  assert os.path.exists(env_path), ".env file not found at {env_path}."

  load_dotenv(env_path)
  OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

  assert OPENAI_API_KEY is not None, "OpenAI API key not found in .env file."

  client = OpenAI(api_key=OPENAI_API_KEY)
  return client

SYSTEM_PROMPT = "You are a writing evaluator designed to assess student story completions. You will be provided children's stories written for a 3-4 year old audience. Your role is to provide constructive, fair, and detailed evaluations based on specific rubric criteria."

USER_PROMPT = """
In the following exercise, the student is given a pre-written beginning of a story. The student needs to complete this story. The exercise tests the student´s language abilities and creativity.

Here is the pre-written beginning:

<PROVIDED BEGINNING>
[STORY_BEGIN]
</PROVIDED BEGINNING>

And here is the students response:

<STUDENT RESPONSE>
[STORY_END]
</STUDENT RESPONSE>

First, provide a concise qualitative assessment about the student's writing. Then, give the writing a grade out of 10. These assessments should be done for each of the following rubric items:

1. Grammar:
* Is the writing grammatically correct?
* Evaluate syntax, punctuation, and sentence structure.
2. Consistency:
* Is the student's writing consistent with the provided beginning of the story?
* How well does the student complete the final sentence of the prescribed beginning?
3. Plot:
* Does the plot of the student's writing make sense (regardless of the provided beginning)?
4. Creativity: 
* How creative is the student's writing?

Format your response as follows:

<GRAMMAR>
[Qualitative assessment of grammar]
</GRAMMAR>
<GRAMMAR_GRADE>
[Grade out of 10]
</GRAMMAR_GRADE>

<CONSISTENCY>
[Qualitative assessment of consistency]
</CONSISTENCY>
<CONSISTENCY_GRADE>
[Grade out of 10]
</CONSISTENCY_GRADE>

<PLOT>
[Qualitative assessment of plot]
</PLOT>
<PLOT_GRADE>
[Grade out of 10]
</PLOT_GRADE>

<CREATIVITY>
[Qualitative assessment of creativity]
</CREATIVITY>
<CREATIVITY_GRADE>
[Grade out of 10]
</CREATIVITY_GRADE>

Provide your assessment below:
"""

def get_request_object(custom_id, content):
    return {
        "custom_id": f"{custom_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            "max_tokens": 1000
        }
    }


def generate_gpt4o_inputs(models, tokenizer, dataloaders, num_generations=200, use_ngram_skip=False):
  client = load_api()
  test_dataset = dataloaders['test']
  
  device = get_device()
  
  i = 0
  num_skipped = 0
  start_time = time.time()
  
  eval_items = []
  
  for i, batch in enumerate(test_dataset):
    
    if i >= num_generations:
      break
    
    sequence = batch['input_ids'][0].tolist()
    # if tokenizer.eos_token_id in sequence:
    #   eos_index = len(sequence) - sequence.tolist()[::-1].index(tokenizer.eos_token_id)
    #   sequence = sequence[eos_index:] # Trim sequence to include only the most recent story
    
    input_size = min(models[0].config.d_seq // 2, len(sequence) // 2)

    model_input = sequence[:input_size]
    model_input = [token for token in model_input if token != tokenizer.pad_token_id and token != tokenizer.eos_token_id]
    story_begin = tokenizer.decode(model_input)
    
    story_true_end = sequence[input_size:]
    story_true_end = [token for token in story_true_end if token != tokenizer.pad_token_id and token != tokenizer.eos_token_id]
    story_true_end = tokenizer.decode(story_true_end)
    with torch.no_grad():
      
      true_prompt = USER_PROMPT.replace('[STORY_BEGIN]', story_begin).replace('[STORY_END]', story_true_end)
      eval_items.append(get_request_object(f"request_{i}_true", true_prompt))

      if isinstance(sequence, list):
        model_input = torch.tensor(model_input).unsqueeze(0)

      model_input = model_input.to(device)
        

      for model in models:

        model.eval()
        model.to(device)
        
        beam_search_sequence = model.beam_search(model_input, max_new_tokens=100, num_beams=3, eos_token_id=tokenizer.eos_token_id, ngram_skip_size=3 if use_ngram_skip else None)
        
        # print(f"Beam search sequence: {beam_search_sequence}")
        beam_search_sequence = beam_search_sequence[0, :].tolist()
        if tokenizer.eos_token_id in beam_search_sequence:
          print(f"EOS token found in beam search sequence {i}.")
          exit()
          # print(f"Beam: {tokenizer.decode(beam_search_sequence)}")
        
        beam_search_sequence = [token for token in beam_search_sequence if token != tokenizer.pad_token_id and token != tokenizer.eos_token_id]
        story_beam_end = tokenizer.decode(beam_search_sequence)
      
        if len(story_beam_end) < 4:
          print(f"Skipping request {i} due to short story beam end.")
          num_skipped += 1
          continue

        beam_prompt = USER_PROMPT.replace('[STORY_BEGIN]', story_begin).replace('[STORY_END]', story_beam_end)
        eval_items.append(get_request_object(f"request_{i}_{model.config.get_name()}", beam_prompt))
      
      i += 1
      time_remaining = get_time_remaining_formatted(start_time, i, num_generations)
      print(f"\r{i}/{num_generations} ({100 * i / num_generations:.2f}%) | Time Remaining: {time_remaining}", end='')
    
  os.makedirs(INPUT_DIR, exist_ok=True)
  with open(f'{INPUT_DIR}/{FILE_NAME}_input.jsonl', 'w') as f:
    for item in eval_items:
      f.write(f"{json.dumps(item)}\n")
  print(f"Generated inputs for GPT model:{MODEL}\n Processed {i}, skipped {num_skipped}.")
  
def create_batch():
  client = load_api()
  batch_input_file = client.files.create(
    file=open(f'{INPUT_DIR}/{FILE_NAME}_input.jsonl', 'rb'),
    purpose="batch"
  )
  batch_input_id = batch_input_file.id
  batch = client.batches.create(
    input_file_id=batch_input_id,
    endpoint="/v1/chat/completions",
    completion_window='24h',
    metadata={
      'description': f'{MODEL} evaluation for GDGPT'
    }
  )
  print(f"Created batch with ID: {batch.id}")

def check_batch():
  client = load_api()
  assert BATCH_ID is not None, "Batch ID not provided."
  batch = client.batches.retrieve(BATCH_ID)
  print(f"Batch status: {batch.status}")
  print(batch)
  
def cancel_batch():
  client = load_api()
  assert BATCH_ID is not None, "Batch ID not provided."
  client.batches.cancel(BATCH_ID)
  print(f"Cancelled batch with ID: {BATCH_ID}")
  
def parse_batch():
  client = load_api()
  assert BATCH_ID is not None, "Batch ID not provided."
  
  batch = client.batches.retrieve(BATCH_ID)
  output_file_id = batch.output_file_id
  
  output_text = client.files.content(output_file_id).text
  
  with open(f'{INPUT_DIR}/{FILE_NAME}_input.jsonl', 'r') as f:
    input_jsons = [json.loads(line) for line in f.readlines()]
    input_texts = {
      input_json['custom_id']: input_json['body']['messages'][1]['content'] for input_json in input_jsons
    }
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  with open(f'{OUTPUT_DIR}/{FILE_NAME}_output.jsonl', 'w') as f:
    f.write(output_text)
    print("Wrote output to file.")
  
  # batch_output = [json.loads(line)['response']['body']['choices'][0]['message']['content'] for line in output_text.split('\n') if line]
  
  # input_ids = [json.loads(line)['custom_id'] for line in output_text.split('\n') if line]
  # batch_input = [input_text[input_id]['body']['messages'][1]['content'] for input_id in input_ids]
  
  score_map = {}
  num_errors = 0
  
  def parse_score(text, tag):
    try:
      text = text.split(f'<{tag}>')[1].split(f'</{tag}>')[0].strip()
    except:
      print(f"Error parsing {tag} from text: {text}")
    
    if '/' in text:
      text = text.split('/')[0].strip()
    
    try:
      score = int(text)
    except:
      return None
    
    return score
  
  for line in output_text.split('\n'):
    if not line:
      continue
    response = json.loads(line)
    custom_id = response['custom_id']
    content = response['response']['body']['choices'][0]['message']['content']
    
    grammar_score = parse_score(content, 'GRAMMAR_GRADE')
    consistency_score = parse_score(content, 'CONSISTENCY_GRADE')
    plot_score = parse_score(content, 'PLOT_GRADE')
    creativity_score = parse_score(content, 'CREATIVITY_GRADE')
    
    if None in [grammar_score, consistency_score, plot_score, creativity_score]:
      num_errors += 1
      continue
    
    label = '_'.join(custom_id.split('_')[2:])
    if label not in score_map:
      score_map[label] = {
        'grammar': [],
        'consistency': [],
        'plot': [],
        'creativity': []
      }
      
    score_map[label]['grammar'].append(grammar_score)
    score_map[label]['consistency'].append(consistency_score)
    score_map[label]['plot'].append(plot_score)
    score_map[label]['creativity'].append(creativity_score)
    
  true_scores = {
    'grammar': sum(score_map['true']['grammar']) / len(score_map['true']['grammar']),
    'consistency': sum(score_map['true']['consistency']) / len(score_map['true']['consistency']),
    'plot': sum(score_map['true']['plot']) / len(score_map['true']['plot']),
    'creativity': sum(score_map['true']['creativity']) / len(score_map['true']['creativity'])
  }
    
  for label, scores in score_map.items():
    avg_scores = {
      'grammar': sum(scores['grammar']) / len(scores['grammar']),
      'consistency': sum(scores['consistency']) / len(scores['consistency']),
      'plot': sum(scores['plot']) / len(scores['plot']),
      'creativity': sum(scores['creativity']) / len(scores['creativity'])
    }
    
    avg_scores['overall'] = sum(avg_scores.values()) / 4
    
    adjusted_scores = {
      'grammar': avg_scores['grammar'] / true_scores['grammar'],
      'consistency': avg_scores['consistency'] / true_scores['consistency'],
      'plot': avg_scores['plot'] / true_scores['plot'],
      'creativity': avg_scores['creativity'] / true_scores['creativity']
    }
    
    avg_scores['adjusted'] = sum(adjusted_scores.values()) / 4
    
    print("=" * 100)
    print(f"Average Scores for {label}:")
    print("=" * 100)
    # print("Grammar:", avg_scores['grammar'], f"({adjusted_scores['grammar']})")
    # print("Consistency:", avg_scores['consistency'], f"({adjusted_scores['consistency']})")
    # print("Plot:", avg_scores['plot'], f"({adjusted_scores['plot']})")
    # print("Creativity:", avg_scores['creativity'], f"({adjusted_scores['creativity']})")
    # print("Overall:", avg_scores['overall'], f"({avg_scores['adjusted']})")
    print("Grammar:", f"({adjusted_scores['grammar']})", avg_scores['grammar'])
    print("Consistency:", f"({adjusted_scores['consistency']})", avg_scores['consistency'])
    print("Plot:", f"({adjusted_scores['plot']})", avg_scores['plot'])
    print("Creativity:", f"({adjusted_scores['creativity']})", avg_scores['creativity'])
    print("Overall:", f"({avg_scores['adjusted']})", avg_scores['overall'])
    print("=" * 100)
    
    
    
  num_responses = len(output_text.split('\n'))
  print(f"Processed {num_responses} responses. {num_errors} errors.")