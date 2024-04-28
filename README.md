# LeaseGPT

LeaseGPT is a personal leasing assistant built using Langchain, OpenAI, and FAISS vector search. It is designed to help users find the best apartment options based on their preferences.

## Structure

The project is structured as follows:

- `app.py`: This is the main application file. It sets up the leasing agent, generates responses based on user input, and handles the Streamlit interface.
- `chatPOC.py`: This file contains a proof of concept for the chat interface using Streamlit.
- `llama_POC.py`: This file contains a proof of concept for using the Llama language model.
- `llmTest.py`: This file contains tests for the language model and functions for getting listings.
- `raw_strings.py`: This file contains raw string data.
- `scrapeCraigslist.py`: This file is responsible for scraping data from Craigslist.
- `requirements.txt`: This file lists all Python dependencies.

## Setup

To set up the project, you need to install the required dependencies listed in the `requirements.txt` file. You can do this by running:

```sh
pip install -r requirements.txt

```

## Usage

To start the application, run:
```sh
streamlit run app.py
```

This will start the Streamlit server and you can interact with the application in your web browser.
App will be hosted soon... 

Contributing
Contributions are welcome. Please submit a pull request or open an issue if you have any improvements or bug fixes.

License
This project is licensed under the terms of the MIT license. See the LICENSE file for the full license text.



