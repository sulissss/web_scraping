from exa_py import Exa
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv
import json
import re
import pandas as pd
import os

# load_dotenv('data.env')

def search(exa, company_name, num_results, search_mode="auto"):
    """
    Performs a search on the Exa API for ESG policies of a given company.

    Args:
        exa (Exa): The Exa API object.
        company_name (str): The name of the company.
        num_results (int): The number of results to return.
        search_mode (str, optional): The search mode. Defaults to "auto".

    Returns:
        str: The search results in a string format.
    """
    # Perform a search on the Exa API for ESG policies of a given company
    # with the specified search mode, number of results, and using the autoprompt.
    return exa.search_and_contents(
        f"Environmental, Social, Governance (ESG) policies of the {company_name} Company",
        type=search_mode,  # The search mode: "auto", "keyword", or "title"
        use_autoprompt=True,  # Whether to use the autoprompt or not
        num_results=num_results,  # The number of results to return
        summary=True  # Whether to include the summary or not in the results
    )

def extract_summaries(exa_results_str: str) -> str:
    """
    Extract all occurrences of the "Summary:" field from the given string.

    Args:
        exa_results_str (str): The string containing the Exa search results.

    Returns:
        str: A single string containing all the extracted summaries.
    """

    # Regular expression to find all occurrences of the "Summary:" field
    summaries = re.findall(
        r"Summary:\s*(.+?)(?=\n(?:Title:|URL:|Score:|Published Date:|Author:|Highlights:|$))",
        exa_results_str,
        re.DOTALL
    )

    # Combine all the extracted summaries into a single string
    return "\n---\n".join(summary.strip() for summary in summaries)


def llm_inference(llm,company_name, exa_results_summary):
    prompt = f"""You have been given an ESG policy summary of the {company_name} company. You are to further refine it,
                 and provide a detailed report of each policy (Environmental, Social, Governance) separately. Ensure that your responses for each
                category do not overlap each other.
                 Write in a professional tone.
                 Do not include any preambles or concluding statements.
                Write your report in bullet points. Present the final response in JSON format.
                 You'll be given a sample input and output to get a clue of the expected format, do not include this data in your response.
                 Sample Input: "Company XYZ is committed to sustainability and reducing its environmental impact. The company has invested in renewable energy sources such as wind and solar, aiming to reduce carbon emissions by 50% by 2030. Additionally, XYZ focuses on social initiatives by supporting local communities through education programs and providing healthcare services to underprivileged areas. The company also emphasizes strong corporate governance by ensuring transparency in its operations and adhering to strict ethical guidelines.""
                 Sample Output: "{{
  "environmental": "Invested in renewable energy sources such as wind and solar, aiming to reduce carbon emissions by 50% by 2030.",
  "social": "Supports local communities through education programs and provides healthcare services to underprivileged areas.",
  "governance": "Ensures transparency in operations and adheres to strict ethical guidelines."
}}"
                  The ESG policy summary is as follows: {exa_results_summary}"""
    
    llm_response = llm.invoke(prompt)
    
    # Regular expression to find the JSON part of the response
    json_match = re.search(r'(\{[\s\S]+\})', llm_response)
    retry_count = 0
    while retry_count < 3:
        json_match = re.search(r'(\{[\s\S]+\})', llm.invoke(prompt))
        if json_match:
            break
        retry_count += 1
        print(f"Retry count: {retry_count}")
        
    if json_match:
        json_text = json_match.group(1)
        json_data = json.loads(json_text)
        return json_data
    
    return llm_response

def convert_json_to_csv(data: dict, csv_file_path='esg_policies.csv'):
    max_len = max(len(lst) for lst in data.values())

    for key in data:
        while len(data[key]) < max_len:
            data[key].append("")

    # Create DataFrame
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(csv_file_path, index=False)
    return dataframe

def main():
    exa = Exa(api_key=os.environ.get('EXA_API_KEY'))
    llm = Ollama(model="llama3")
    csv_file_path = 'esg_policies.csv'

    company_name = input("Enter the name of the company: ")
    num_results = 10

    exa_results = search(exa,company_name, num_results)
    exa_result_summaries = extract_summaries(f"{exa_results}")
    # print(exa_result_summaries)

    data_json = llm_inference(llm, company_name, exa_result_summaries)
    dataframe = convert_json_to_csv(data_json, csv_file_path)

    print(data_json)
    print(dataframe)

if __name__ == "__main__":
    main()