import tiktoken


models = [
  {'completion_cost_per_token': '0.002 / 1000',
   'description': 'Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with our latest model iteration.',
   'max_tokens': '4096',
   'name': 'gpt-3.5-turbo',
   'prompt_cost_per_token': '0.002 / 1000'},
  {'completion_cost_per_token': '0.06 / 1000',
   'description': 'Same capabilities as the base gpt-4 mode but with 4x the context length. Will be updated with our latest model iteration.',
   'max_tokens': '32768',
   'name': 'gpt-4-32k',
   'prompt_cost_per_token': '0.12 / 1000'},
  {'completion_cost_per_token': '0.06 / 1000',
   'description': 'More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration.',
   'max_tokens': '8192',
   'name': 'gpt-4',
   'prompt_cost_per_token': '0.03 / 1000'},
  {'completion_cost_per_token': '0.0004 / 1000',
   'description': 'Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.',
   'max_tokens': '2049',
   'name': 'text-ada-001',
   'prompt_cost_per_token': '0.0004 / 1000'},
  {'completion_cost_per_token': '0.0005 / 1000',
   'description': 'Capable of straightforward tasks, very fast, and lower cost.',
   'max_tokens': '2049',
   'name': 'text-babbage-001',
   'prompt_cost_per_token': '0.0005 / 1000'},
  {'completion_cost_per_token': '0.002 / 1000',
   'description': 'Very capable, faster and lower cost than Davinci.',
   'max_tokens': '2049',
   'name': 'text-curie-001',
   'prompt_cost_per_token': '0.002 / 1000'},
  {'completion_cost_per_token': '0.02 / 1000',
   'description': None,
   'max_tokens': '8001',
   'name': 'text-davinci-001',
   'prompt_cost_per_token': '0.02 / 1000'},
  {'completion_cost_per_token': '0.02 / 1000',
   'description': 'Similar capabilities to text-davinci-003 but trained with supervised fine-tuning instead of reinforcement learning',
   'max_tokens': '4097',
   'name': 'text-davinci-002',
   'prompt_cost_per_token': '0.02 / 1000'},
  {'completion_cost_per_token': '0.02 / 1000',
   'description': 'Most capable GPT-3 model. Can do any task the other models can do, often with higher quality.',
   'max_tokens': '4097',
   'name': 'text-davinci-003',
   'prompt_cost_per_token': '0.02 / 1000'},
 {'completion_cost_per_token': '0.06 / 1000',
  'description': 'Snapshot of gpt-4 from March 14th 2023. Unlike gpt-4, this '
                 'model will not receive updates, and will be deprecated 3 '
                 'months after a new version is released.',
  'max_tokens': '8192',
  'name': 'gpt-4-0314',
  'prompt_cost_per_token': '0.03 / 1000'},
 {'completion_cost_per_token': '0.06 / 1000',
  'description': 'Snapshot of gpt-4-32 from March 14th 2023. Unlike gpt-4-32k, '
                 'this model will not receive updates, and will be deprecated '
                 '3 months after a new version is released.',
  'max_tokens': '32768',
  'name': 'gpt-4-32k-0314',
  'prompt_cost_per_token': '0.12 / 1000'},
 {'completion_cost_per_token': '0.002 / 1000',
  'description': 'Snapshot of gpt-3.5-turbo from March 1st 2023. Unlike '
                 'gpt-3.5-turbo, this model will not receive updates, and will '
                 'be deprecated 3 months after a new version is released.',
  'max_tokens': '4096',
  'name': 'gpt-3.5-turbo-0301',
  'prompt_cost_per_token': '0.002 / 1000'},
 {'completion_cost_per_token': '0.002 / 1000',
  'description': 'Optimized for code-completion tasks',
  'max_tokens': '8001',
  'name': 'code-davinci-002',
  'prompt_cost_per_token': '0.002 / 1000'},
 {'completion_cost_per_token': '0.0200 / 1000',
  'description': 'Most capable GPT-3 model. Can do any task the other models '
                 'can do, often with higher quality.',
  'max_tokens': '2049',
  'name': 'davinci',
  'prompt_cost_per_token':'0.002 / 1000'},
 {'completion_cost_per_token': '0.0020 / 1000',
  'description': 'Very capable, but faster and lower cost than Davinci.',
  'max_tokens': '2049',
  'name': 'curie',
  'prompt_cost_per_token': None},
 {'completion_cost_per_token': '0.0005 / 1000',
  'description': 'Capable of straightforward tasks, very fast, and lower cost.',
  'max_tokens': '2049',
  'name': 'babbage',
  'prompt_cost_per_token': None},
 {'completion_cost_per_token': '0.0004 / 1000',
  'description': 'Capable of very simple tasks, usually the fastest model in '
                 'the GPT-3 series, and lowest cost.',
  'max_tokens': '2049',
  'name': 'ada'},
 {'completion_cost_per_token': '0.002 / 1000',
  'description': 'Earlier version of code-davinci-002',
  'max_tokens': '8001',
  'name': 'code-davinci-001',
  'prompt_cost_per_token': '0.002 / 1000'},
 {'completion_cost_per_token': None,
  'description': 'Almost as capable as Davinci Codex, but slightly faster. '
                 'This speed advantage may make it preferable for real-time '
                 'applications.',
  'max_tokens': '2048',
  'name': 'code-cushman-002',
  'prompt_cost_per_token': None},
 {'completion_cost_per_token': None,
  'description': 'Earlier version of code-cushman-002',
  'max_tokens': '2048',
  'name': 'code-cushman-001',
  'prompt_cost_per_token': None}
]


def count_tokens(text: str, model: str) -> int:
    """
    Counts the number of tokens in a text string using the encoding for a given LLM.
    """

    # Get the tokeniser corresponding to the model
    enc = tiktoken.encoding_for_model(model)

    # Encode the string
    token_list: list = enc.encode(text)

    # Measure the length of token_list
    token_length: int = len(token_list)

    # Return the length of token_list
    return token_length


def calculate_cost(prompt: str, completion: str, model: str) -> float:
    """
    Calculates the cost of requesting a completion for a given LLM.
    """

    # Count the number of tokens in the text
    token_count_prompt = count_tokens(prompt, model)
    token_count_completion = count_tokens(completion, model)

    # Get the respective model's cost per token
    for mdl in models:
        if mdl['name'] == model:
            cost_prompt = token_count_prompt * eval(mdl['prompt_cost_per_token'])
            cost_completion = token_count_completion * eval(mdl['completion_cost_per_token'])

    # Calculate the total cost of encoding the text
    cost = cost_prompt + cost_completion

    return cost


if __name__ == "__main__":
    import urllib3
    
    # Read an example text file from the web
    text = urllib3.PoolManager().request('GET', 'https://example-files.online-convert.com/document/txt/example.txt').data.decode('utf-8')

    # Count number of characters in the text
    character_count = len(text)
    print("String length in characters: " + str(character_count))

    # Count number of tokens in the text using GPT4 encoding
    token_count = count_tokens(text, "gpt-4")
    print("String length in tokens: " + str(token_count))

    # Measure the ratio of tokens to characters:
    tokens_to_characters_ratio = token_count / character_count
    print("Ratio of tokens to characters: " + str(tokens_to_characters_ratio))

    # Estimate the cost of a hypothetical GPT4 completion the same length of the text
    completion_cost = calculate_cost(text,text,"gpt-4")
    print("Cost of a GPT4 prompt + completion both this length: " + str(completion_cost))
