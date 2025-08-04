pip install uv

uv --version

Initialize the environment (It will create a project folder structure named AeroGuardian)
uv init AeroGuardian

conda deactivate

uv python list
uv python install cpython-3.12.10-windows-x86_64-none

To create a virtual environment using uv (I'm also mentioning my python version that I want to use in this environment) 
---> Creates a Folder named (my_uv_env). This is the same name of my environment.
uv venv my_uv_env --python cpython-3.12.10-windows-x86_64-none 


Activating the envrionment ( Inside the my_uv_env folder, we need to run "Activate.ps1" for powershell ): 
& "C:\For Drive D\All Mainstream Projects\AI PROJECTS\AeroGuardian-LangGraph\my_uv_env\Scripts\Activate.ps1"

Install packages in our environment : 
uv pip install langgraph langchain openai faiss-cpu python-dotenv ollama
uv pip install -U langchain-ollama
uv pip install -U langchain-openai


To check all the installed packages : 
uv pip list


To run the project : 
python aeroguardian.py

