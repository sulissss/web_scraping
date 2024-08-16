from langchain_community.llms.ollama import Ollama
from exa_py import Exa
from dotenv import load_dotenv
import pandas as pd
import json
import os

load_dotenv('data.env')

llm = Ollama(model="llama3")
exa = Exa(api_key=os.environ.get("EXA_API_KEY"))

policies = ["environmental", "social", "ethical governance"]

def search(query: str) -> str:
    """
    Search for webpages based on a query.

    This function uses the Exa API to search for webpages based on a query.
    It returns the search results in a text format.

    Args:
        query: The search query string.

    Returns:
        A string containing the search results.
    """
    return exa.search_and_contents(f"{query}", type="auto", use_autoprompt=True, num_results=3, text=True)

def rag(company_name: str, policy_type: str):
    """
    Generate a detailed summary of a company's policy type.

    This function uses the Ollama model to generate a detailed summary of a company's policy type,
    based on the web data scraped from the web. The summary should not include any preambles or
    concluding statements.

    Args:
        company_name (str): The name of the company.
        policy_type (str): The type of policy.

    Returns:
        str: The generated summary of the policy.
    """
    # Construct the prompt to generate the summary
    prompt = f"""Context: ESG policies. Based on the following data scraped from the web, provide a detailed summary of ONLY the {policy_type} policies of the {company_name} company.
               Do not include any preambles or concluding statements.
               Only mention the data related to the company's {policy_type} policies.
               Web data: {search(f"""{policy_type} policies of {company_name}""")}"""

    # Invoke the Ollama model to generate the summary
    return llm.invoke(prompt)

def tabularize_data(company_name: str): 
    updated_policies = {policy: rag(company_name, policy) for policy in policies}
    data = {policy: updated_policies[policy].replace('\n\n', '\n').split('\n') for policy in policies}
    data_json = data

    with open('data.txt', 'w') as file:
        file.write(json.dumps(data, indent=4))

    max_len = max(len(lst) for lst in data.values())

    for key in data:
        while len(data[key]) < max_len:
            data[key].append("")

    # Create DataFrame
    dataframe = pd.DataFrame(data)
    dataframe.to_csv('output.csv', index=False)
    return data_json, dataframe

def main():
    company_name = input("Enter the name of the company: ")
    data_json, dataframe = tabularize_data(company_name)
    print(data_json)
    print(dataframe)

if __name__ == "__main__":
    main()