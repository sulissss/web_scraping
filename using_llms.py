from langchain_community.llms.ollama import Ollama
from exa_py import Exa
from dotenv import load_dotenv
import pandas as pd
import json
import os
import re

load_dotenv('data.env')

llm = Ollama(model="llama3")
exa = Exa(api_key=os.environ.get("EXA_API_KEY"))

policies = ["Environmental", "Social", "Ethical Governance"]
# policies = ["ESG (Environmental, Social, Governance)"]

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
    return exa.search_and_contents(f"{query}", type="keyword", use_autoprompt=True, num_results=3, text=True)

def format_exa_results(exa_data):
    record_texts = re.findall(r"Text:\s*(.+?)(?=\n(?:Title:|URL:|Score:|Published Date:|Author:|Highlights:|Summary:|$))", exa_data, re.DOTALL)
    return "\n---\n".join(text.strip() for text in record_texts)

def sample_input_and_output(policy_type: str):
    if policy_type == "Environmental":
        return ("""Linde prioritizes sustainability, health, safety, and environmental responsibility across its global operations. Policies emphasize human rights, ecosystem protection, water conservation, GHG reduction, responsible animal testing, and chemical safety.""", 
    """**Ecosystem Protection**: Linde's HSE policy reflects a strong commitment to avoiding harm to the environment and communities, emphasizing the importance of ecosystem preservation.
	**Water Conservation**: Linde focuses on responsible water management, returning over 75 percent of global freshwater withdrawals to their original source at the same or better quality.
	**Greenhouse Gas Reduction**: Linde tracks and recalculates GHG emissions to meet climate neutrality targets by 2050, adjusting baseline emissions inventories as needed.
	**Chemical Safety**: Linde ensures that operations prevent harm to people and the environment, with strict adherence to chemical safety standards."
    """)
    elif policy_type == "Social":
        return ("""Corporate responsibility drives Linde to develop economic, social, and environmental solutions, focusing on sustainable growth, research, and a cleaner future while engaging in the political process to achieve productivity.""",
                """**Employee Well-being**: Linde is committed to treating employees with respect and ensuring their well-being, fostering a positive work environment.
**Community Engagement**: Linde actively participates in the communities where it operates, contributing to social development and supporting local initiatives.
**Sustainable Growth**: Linde prioritizes sustainable growth that benefits both the company and society, focusing on long-term social impact.
**Political Participation**: Linde engages in the political process to advocate for policies that align with its mission of improving productivity and societal well-being.
""")
    elif policy_type == "Ethical Governance":
        return ("""Linde is committed to human rights, ensuring fair labor practices, safety, and freedom of association. They prohibit child labor, enforce ethical supplier conduct, and engage in community improvement globally.""", 
                """	**Commitment to human rights and fair labor practices**
	**Strict prohibition of child and forced labor**
	**Focus on safety and a secure work environment**
	**Promotion of fair compensation and equal remuneration**
	**Support for freedom of association and collective bargaining**
	**Enforcement of ethical conduct in supplier relationships**
	**Active community engagement and local hiring initiatives**""")
    else:
        return ""

def rag(company_name: str, policy_type: str):
    # Search for the web data
    web_data = f"""{search(f"{policy_type} policies of the {company_name} company")}"""
    formatted_web_data = format_exa_results(web_data)

    # Construct the prompt to generate the summary
    prompt = f"""
        Based on the following web data, provide a detailed summary of ONLY the {policy_type} policies of the {company_name} company.
        Do not include any preambles or concluding statements.
        Do not hallucinate/respond if you feel like you do not have enough data.
        You'll be given a sample input and output to get a clue of the format, do not include this data in your response.
        Web data: {formatted_web_data}
    """

    sample_input, sample_output = sample_input_and_output(policy_type)

    prompt += f"""Sample Input: "{sample_input}"
                Sample Output: "{sample_output}"
                """

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