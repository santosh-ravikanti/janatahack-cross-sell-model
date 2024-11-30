FROM python:3.8.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

#CMD ["uvicorn", "model_app:app", "--host", "0.0.0.0", "--port", "80"] # this line of code is used for fastapi

# This line of code is used for using streamlit - use ENTRYPOINT command.
ENTRYPOINT ["streamlit", "run", "webview_fileupload.py", "--server.address", "0.0.0.0", "--server.port", "80"] 