\# Configure Envs

Currently I added `langchain` and `openai` packages in requirements.txt. However, that might be causing the issue -- the SDK is supposed to use `/v1/embeddings` from Arm's proxy but is trying to pull it from actual OpenAI API.



Otherwise, I mostly follow instructions from here: https://dev.to/mrrishimeena/customize-chatgpt-for-your-codebase-openai-14n6



I edited `create\_rag\_vectors.py` for Verilator application.



1. `git submodule update --init --recursive` to get Verilator source files
2. Add your OpenAI API key to `.env`
3. Try `python create\_rag\_vectors.py`. I got stuck here.
