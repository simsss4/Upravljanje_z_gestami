FROM tensorflow/tensorflow:2.17.0

# Install system dependencies for tkinter
RUN apt-get update && apt-get install -y python3-tk && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
CMD ["python", "merging_gui.py"]