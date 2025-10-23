# bentebot
*A locally hosted AI Discord bot powered by Ollama*

## Overview
A self-hosted Discord chatbot using local LLMs. It protects access through server trust and DM permissions. Future support planned for multimodal AI such as image, music, and video generation.

## Core Features
- Locally hosted LLM chatbot through **Ollama**
- Trusted server validation
- DM whitelist and admin permission checks
- redis database to store message history and dynamic whitelist and admin rights.
- Docker to host discord bot and redis containers


## Future Roadmap
- Multiple art generation with Hugging Face pipelines


### Requirements
```bash
pip install discord.py redis python-dotenv
````


### Startup Commands
#### Manually run bentebot container only
```bash
docker build -t python-bentebot .
docker build --no-cache -t python-bentebot .
docker run python-bentebot
```

#### Create Image and containers for bentebot and redis on same docker network
```bash
`docker compose up -d --build`
```