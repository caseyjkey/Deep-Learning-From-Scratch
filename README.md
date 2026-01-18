# Model Stack
* The LLM is hosted through ollama.
1. `ollama run gemma3:12b-it-qat`
2. `curl http://localhost:11434/api/generate -d '{"model": "llama2", "prompt":"Why is the sky blue?"}'`