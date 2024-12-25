# LLM-Powered Application for Extracting Company Performance Metrics

## Overview

This application processes user queries related to company performance metrics and converts the extracted information into a structured JSON format. It utilizes a language model to extract key details such as the company name, performance metric, start date, and end date.

## Instructions

### 1. Clone the Repository

Clone the repository to your local machine by running the following command:

```bash
git clone https://github.com/your-repository-name.git
```

### 2. Navigate to the Application Folder

Once you have cloned the repository, navigate to the `LLM_Application` folder using the terminal:

```bash
cd LLM_Application
```


### 3. Set Up the .env File

Create a `.env` file in the `LLM_Application` folder to store your API key securely.

1. In the `LLM_Application` folder, create a new file named `.env`.
2. Open the `.env` file and add the following line, replacing `{your api key}` with your actual API key:
   
   ```plaintext
   API_KEY={your api key}
   ```
Replace {your api key} with the actual API key that you have received for using the LLM model. Ensure that the .env file is kept secure and not shared publicly.

### 4. Install Dependencies

Once the `.env` file is set up, you need to install all the required dependencies for the application to run.

Run the following command in your terminal inside the `LLM_Application` folder:

```bash
pip install -r requirements.txt
```
This will install all the required libraries and dependencies listed in the requirements.txt file.

### 5. Run the Application

Once the dependencies are installed, you can run the application by executing the following command in the terminal:

```bash
python -u llm_application.py
```
The application will prompt you to enter a query. After entering the query, it will process the information and display the extracted data in a structured JSON format.