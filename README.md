---
title: LC Gradio DocsAI
emoji: 🚀
colorFrom: gray
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# LC Gradio DocsAI 🚀

## Overview
LC-Gradio-DocAI is a demo project showcasing a privately hosted advanced Documentation AI helper, demonstrating a fine-tuned 7B model's capabilities in aiding users with software documentation. This application integrates technologies like Retrieval-Augmented Generation (RAG) using LangChain, a vector store using Chroma DB or and FAISS and Gradio for a model UI to offer insightful documentation assistance. It's designed to help users navigate and utilize software tools efficiently by retrieving relevant documentation pages and maintaining conversational flow.

## Key Features
- **AI-Powered Documentation Retrieval:** Utilizes various fine-tuned 7B models for precise and context-aware responses.
- **Rich User Interface:** Features a user-friendly interface built with Gradio.
- **Advanced Language Understanding:** Employs LangChain for implementing RAG setups and sophisticated natural language processing.
- **Efficient Data Handling:** Leverages Chroma DB and FAISS for optimized data storage and retrieval.
- **Retrieval Chain with Prompt Tuning:** Includes a retrieval chain with a prompt template for prompt tuning.
- **Conversation Memory:** Incorporates BufferMemory for short-term conversation memory, enhancing conversational flow.

## Models Used
This setup is tested with the following models:
- `mistralai/Mistral-7B-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `HuggingFaceH4/zephyr-7b-beta`
- `HuggingFaceH4/zephyr-7b-alpha`
- `tiiuae/falcon-7b-instruct`
- `microsoft/Orca-2-7b`
- `teknium/OpenHermes-2.5-Mistral-7B`

## Prerequisites
- Python 3.8 or later
- [Additional prerequisites...]

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Docs-QAchat.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Docs-QAchat
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
1. Create a `.env` file in the project root.
2. Add the following environment variables to the `.env` file:
   ```
   HUGGINGFACEHUB_API_TOKEN=""
   AWS_S3_LOCATION=""
   AWS_S3_FILE=""
   VS_DESTINATION=""
   ```

## Usage
Start the application by running:
```bash
python app.py
```
[Include additional usage instructions and examples]

## Contributing
Contributions to LC-Gradio-DocsAI are welcome. Here's how you can contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Make changes and commit (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/YourFeature).
5. Create a new Pull Request.

## Support
For support, please open an issue here on Github.

## Authors and Acknowledgement
- [Name]
- Thanks to contributors of all the awesome open-source LLMs, LangChain, HuggingFace, Chroma Vector Store, FAISS and Graido UI.

## License
This project is licensed under the [License] - see the LICENSE file for details.
