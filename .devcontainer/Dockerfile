FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Requis par Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Claude Code
RUN npm install -g @anthropic-ai/claude-code
