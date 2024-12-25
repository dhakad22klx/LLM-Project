"""
LLM-Powered Application for Extracting Company Performance Metrics
Author: [Deepak Dhakad]
Date: [25th December 2024]

Description:
This application processes user queries related to company performance metrics
and converts the extracted information into a structured JSON format.
"""

# Import required libraries
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from llama_index.llms.groq import Groq

load_dotenv()

API_KEY = os.getenv('API_KEY')


class LLMPerformanceMetrics:
    """
    This class interacts with the LLM Instant to process user queries and extract
    company performance metrics in a structured JSON format.
    """

    # Constants
    API_KEY =  API_KEY # Personal API key

    def __init__(self):
        self.API_KEY = API_KEY

    def _call_llm(self, query):
        """
        Calls the model's to process the user query and extract relevant information.

        Parameters:
            query (str): The user's input query.

        Returns:
            dict: The JSON response from the LLM containing extracted information.
        """

        # Construct the prompt with the given query with proper formatting
        prompt = self._format_query(query)
        
        try:
            llm = Groq(model="llama-3.1-8b-instant", api_key=self.API_KEY)
            response = llm.complete(prompt)
            return response

        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return {"error": f"Failed to retrieve data from the LLM API : {e}"}

    def _format_query(self, query):

        """
        Formats the query into a specific prompt to send to the LLM.

        Parameters:
            query (str): The user's input query.

        Returns:
            str: The formatted prompt for the LLM API.
        """
        return f"""
        Extract the following details from the query:
        1. Entity (Company name)
        2. Metric (Performance metric like revenue, profit, GMV, etc.)
        3. Start Date (The start of the time period in YYYY-MM-DD format)
        4. End Date (The end of the time period in YYYY-MM-DD format)

        Example Queries:
        - Query: "What was Amazon's revenue in 2023?"
          Expected Output: [{{
            "Entity": "Amazon",
            "Metric": "revenue",
            "Start Date": "2023-01-01",
            "End Date": "2023-12-31"
        }}]

        - Query: "How much profit did Flipkart make in 2022?"
          Expected Output: [{{
            "Entity": "Flipkart",
            "Metric": "profit",
            "Start Date": "2022-01-01",
            "End Date": "2022-12-31"
        }}]
        Must remember these points :
        Your task is to simply give json output of the query in expected format and nothing else. 
        If the user query does not explicitly mention the start date and/or end date, assume the following defaults:
        - Start Date: Today's date minus one year.
        - End Date: Today's date.
        If query contains multiple companies then create a sepearete object for each company and return the array of object.
        Format will look like this 
        [{{
            "Entity": "Flipkart",
            "Metric": "profit",
            "Start Date": "2022-01-01",
            "End Date": "2022-12-31"
        }},
        {{
            "Entity": "Amazon",
            "Metric": "revenue",
            "Start Date": "2023-01-01",
            "End Date": "2023-12-31"
        }}]
        Query: 
        "{query}"

        """

    def process_query(self, query):
        """
        Main function to handle the query processing.

        Parameters:
            query (str): The user's input query.

        Returns:
            dict: Structured JSON response containing the extracted information.
        """

        print("Processing Query...")

        llm_response = self._call_llm(query)

        # Handle the response and print the structured output
        if "error" in llm_response:
            print(f"Error: {llm_response['error']}")
            return llm_response
        else:
            print("Query Processed Successfully!")
            return llm_response


def main():

    # Create instance of LLMPerformanceMetrics class
    llm_app = LLMPerformanceMetrics()

    # Take user input
    query = input("Enter your query (e.g., What was Microsoft's revenue in 2023?): ")

    # Process the query
    result = llm_app.process_query(query)

    # Optionally, handle the result in further steps or actions
    print(result)


# Run the main function
if __name__ == "__main__":
    main()