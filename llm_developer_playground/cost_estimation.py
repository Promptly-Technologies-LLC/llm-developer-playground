import tiktoken
from typing import Union

# Define a list of OpenAI models, with descriptions, max tokens, and per-token costs
# for prompts and completions for each model
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


def count_tokens(text: Union[str, list], model: str) -> int:
    """
    Counts the number of tokens in string or or chat messages list using the encoding
    for a given LLM.
    """

    # Get the tokeniser corresponding to the model
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        enc = tiktoken.get_encoding("cl100k_base")

    if isinstance(text, str):
        # Encode the string
        token_list: list = enc.encode(text)

        # Measure the length of token_list
        token_length: int = len(token_list)
    else:
        # This code for counting chat message tokens is adapted from OpenAI Cookbook:
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        if model == "gpt-3.5-turbo":
            print("Warning: gpt-3.5-turbo may change over time. Calculating num tokens assuming gpt-3.5-turbo-0301.")
            return count_tokens(text, model="gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            print("Warning: gpt-4 may change over time. Calculating num tokens assuming gpt-4-0314.")
            return count_tokens(text, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""count_tokens() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        token_length = 0
        for message in text:
            token_length += tokens_per_message
            for key, value in message.items():
                token_length += len(enc.encode(value))
                if key == "name":
                    token_length += tokens_per_name
        token_length += 3
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
    
    example_prompt = """
        You are a helpful, pattern-following assistant that translates
        corporate jargon into plain English. Translate the following:
        New synergies will help drive top-line growth.
        """

    # Count number of characters in the example_prompt
    character_count = len(example_prompt)
    print("Example prompt length in characters: " + str(character_count))

    # Count number of tokens in the example_prompt using GPT4 encoding
    token_count = count_tokens(example_prompt, "gpt-4")
    print("Example prompt length in tokens: " + str(token_count))

    # Measure the ratio of tokens to characters:
    tokens_to_characters_ratio = token_count / character_count
    print("Ratio of tokens to characters: " + str(tokens_to_characters_ratio))

    # Estimate the cost of a hypothetical GPT4 completion the same length as the example_prompt
    completion_cost = calculate_cost(example_prompt, example_prompt, "gpt-4")
    print("Cost of a GPT4 prompt + completion both this length: " + str(completion_cost))

    # This example chat completion prompt is borrowed from OpenAI Cookbook: 
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    example_chat_prompt = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        }
    ]

    # Count number of characters in the example_chat_prompt (including JSON markup)
    character_count = sum([len(str(message)) for message in example_chat_prompt])
    print("Example chat prompt length in characters: " + str(character_count))

    # Count number of tokens in the example_chat_prompt using GPT4 encoding
    token_count = count_tokens(example_chat_prompt, "gpt-4")
    print("Example chat prompt length in tokens: " + str(token_count))

    # Measure the ratio of tokens to characters:
    tokens_to_characters_ratio = token_count / character_count
    print("Ratio of tokens to characters: " + str(tokens_to_characters_ratio))

    # Estimate the cost of a hypothetical GPT4 completion the same length as the example_chat_prompt
    completion_cost = calculate_cost(example_chat_prompt, example_chat_prompt, "gpt-4")
    print("Cost of a GPT4 chat prompt + completion both this length: " + str(completion_cost))

    # The above functions allow calculation of cost for hypothetical prompts/responses.
    # However, if you're using langchain to make an actual API call, you can get the
    # real cost from the callback after the API returns a response.
    from langchain.llms import OpenAI
    from langchain.callbacks import get_openai_callback
    from dotenv import load_dotenv

    load_dotenv()

    llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

    with get_openai_callback() as cb:
        # Track token usage over multiple API calls
        result = llm("Tell me a joke")
        result2 = llm("Tell me a joke")
        print(cb)

        # Save cost to a variable of type float
        cost = cb.total_cost
        print(type(cost))
    
    # It is worth looking into the plumbing of how the langchain implementation works.
    # Supposedly it is using the OpenAI API callback, so this info may be available w/o
    # using langchain.
