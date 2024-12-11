from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.tools.render import render_text_description

@tool
def get_length_of_string(string: str) -> int:
    """Returns the length of the string by characters"""
    print(f"Getting length of string: {string}")
    text = string.strip("'\n'").strip('"')
    return len(text)

if __name__ == "__main__":  # Corrected to "__main__"
    print("Initializing Google Generative AI model...")
    # Initialize the Google Generative AI model
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-1.0-pro",
        max_tokens=1024,
    )
    print("Creating tools...")
    # List of tools
    tools = [get_length_of_string]

    # Template for agent's prompt
    template = """
        Answer the following questions as best you can.
        You have access to the following tools:
        {tools}
        Use the following format:

        Question: the input question you must answer 
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:"""

    print("Creating prompt template...")
    # Create the prompt template
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    print("Creating the chain...")
    # Create the chain
    chain = {"input": lambda x: x["input"]} | prompt | llm

    # Example usage
    input_query = {"input": "what is the length in characters of the text 'Artificial Intelligence'?"}
    print(f"Invoking chain for query: {input_query['input']}")
    try:
        # Invoke the chain
        res = chain.invoke(input_query)
        # Print the result
        print(f"Result: {res}")
    except Exception as e:
        print(f"An error occurred: {e}")
