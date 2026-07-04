FROM python:3.11-slim
WORKDIR /app

# Node.js (for slides_generator.js / pptxgenjs)
RUN apt-get update && apt-get install -y curl libgomp1 && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Node.js dependencies
COPY package.json package-lock.json* ./
RUN npm install --production

# Application files
COPY main.py rule_engine.py slides_generator.js \
     opportunity_graph.py gap_hunter.py playbook_generator.py \
     sweden_pelagic_seed.py graph_bootstrap.py publ.py \
     datalab.py datalab_engine.py ./
COPY index.html workshop.html architecture.html nda.html publ.html login.html datalab.html ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
