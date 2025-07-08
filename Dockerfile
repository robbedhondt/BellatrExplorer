# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /src
COPY src/ /src

# EXPOSE 8080
# CMD ["python", "-m", "shiny", "run", "--host", "0.0.0.0", "--port", "8080", "--reload", "app.py"]
CMD ["python", "app.py"]
