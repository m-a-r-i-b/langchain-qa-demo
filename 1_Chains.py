# Simple example of chain 
# For example, we can create a chain that takes user input, formats it with a PromptTemplate, and then passes the formatted response to an LLM.


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
load_dotenv(".env")

# Initialize LLM
llm = OpenAI(temperature=0.9)

# Setup a re-useable template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# As opposed to passing the entire prompt "What is a good name for a company that makes colorful hats"
# We only need to pass PRODUCT_NAME
# print(chain.run("colorful hats"))
print(chain.run("electric cars"))