# bentebot
*A locally hosted AI Discord bot powered by Ollama*

## Overview
A self-hosted Discord chatbot using local LLMs. It protects access through server trust and DM permissions. Future support planned for multimodal AI such as image, music, and video generation.

## Core Features
- Locally hosted LLM chatbot using **Ollama**
- redis database to store message history, whitelists, admin rights, etc.
- Docker to host discord bot and redis containers together through seamless docker-compose file


## Future Roadmap
- Different AI Tasks using Hugging Face pipelines


### Requirements
```bash
pip install --no-cache-dir -r requirements.txt
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
docker compose up -d --build
```