#!/bin/bash

echo "Starting Ollama server..."
ollama serve &
ollama pull llama3 &
ollama run llama3


echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done