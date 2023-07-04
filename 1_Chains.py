# Simple example of chain 

#  For example, we can create a chain that takes user input, formats it with a PromptTemplate, and then passes the formatted response to an LLM.



from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)