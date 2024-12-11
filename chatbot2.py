from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

api = os.getenv("GOOGLE_API_KEY")
model = 'gemini-1.5-flash'

if __name__ == "__main__":
    while True:
        print("Welcome to chatbot! How can i assist you")
        input_user = input("You: ")

        summary_prompt = """
        Give me the details about {freedom_fighters} in India
        """
        prompt_template = PromptTemplate(input_variables={"freedom_fighters"},template = summary_prompt)

        llm = ChatGoogleGenerativeAI(
            model=model,
            api_key = api)

    

        chain = prompt_template | llm | StrOutputParser()

        res = chain.invoke({ "freedom_fighters":input_user})

        print(res)

    
