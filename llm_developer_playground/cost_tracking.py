# If you're using langchain to make an actual API call, you can get the
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

# If you aren't using langchain, you can use the OpenAI API response in conjunction with
# the model cost to calculate your prompt + completion cost
import openai
