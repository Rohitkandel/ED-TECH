# ED-TECH AI-Powered Q&A System

## Overview

This project is a Proof of Concept (POC) for an AI-powered Q&A system designed for the ED-TECH industry. It aims to automate student query responses using Large Language Models (LLMs) and Natural Language Processing (NLP) techniques.

## Features

- Automated responses to student queries
- Utilizes real student Q&A data for training
- Built with LangChain and Google's Generative AI
- Vector database for efficient information retrieval
- Streamlit-based user interface for easy interaction

## Installation

1. Clone the repository:
   git clone https://github.com/Rohitkandel/ED-TECH.git
   cd ED-TECH
   
2. Install required packages.

3.  Set up environment variables:

## Usage

1. Run the Streamlit app.

2. Open the provided URL in your web browser.

3. Click "Create Knowledgebase" to initialize the vector database.

4. Start asking questions in the input field.

## Project Structure

- `main.py`: Core functionality for vector database creation and Q&A chain
- `app.py`: Streamlit user interface
- `data_faqs.csv`: Sample dataset of questions and answers (not included in repo)

## Customization

To adapt this system for your specific ED-TECH needs:

1. Replace `data_faqs.csv` with your own Q&A dataset.
2. Modify the `CSVLoader` in `main.py` to match your data structure.
3. Adjust the prompt template in `get_qa_chain()` function as needed.
4. Experiment with different LLMs or embedding models for potentially better results.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/Rohitkandel/ED-TECH/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain for providing the framework for building LLM applications
- Google Generative AI for the language model
- Streamlit for the easy-to-use web interface
   
