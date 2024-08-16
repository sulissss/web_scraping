from langchain_community.llms.ollama import Ollama
from exa_py import Exa
from dotenv import load_dotenv
import pandas as pd
import json
import os

load_dotenv('data.env')

llm = Ollama(model="llama3")
exa = Exa(api_key=os.environ.get("EXA_API_KEY"))

policies = ["environmental", "social", "governance"]

def search(query: str):
    """Search for webpages based on a query."""
    return exa.search_and_contents(f"{query}", type="auto", use_autoprompt=True, num_results=3, text=True)

def rag(company_name: str, policy_type: str):
    return llm.invoke(f"""Based on the following data scraped from the web, provide a detailed summary of only the {policy_type} policies of the {company_name} company.
                      Do not include any preambles or concluding statements.
                      Web data: {search(f"""{policy_type} policies of {company_name}""")}""")

def tabularize_data(company_name: str): 
    updated_policies = {policy: rag(company_name, policy) for policy in policies}
    data = {policy: updated_policies[policy].replace('\n\n', '\n').split('\n') for policy in policies}
    with open('data.txt', 'w') as file:
        file.write(json.dumps(data, indent=4))

    max_len = max(len(lst) for lst in data.values())

    for key in data:
        while len(data[key]) < max_len:
            data[key].append("")

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv('output.csv')
    return df

def main():
    company_name = input("Enter the name of the company: ")
    print(tabularize_data(company_name))

if __name__ == "__main__":
    main()