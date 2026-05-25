# 1. Setăm platforma universală (Windows/Mac) și folosim Python 3.11 pentru compatibilitate cu MCP
FROM --platform=linux/amd64 python:3.11-slim

# 2. Setăm folderul de lucru în container
WORKDIR /app

# 3. Copiem fișierele de configurare
COPY requirements.txt .

# 4. Instalăm dependințele din listă, plus pachetele vitale pentru MCP
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir mcp requests fastmcp

# 5. Copiem restul proiectului (main.py, mcp_server.py, folderul models etc.)
COPY . .

# 6. Expunem porturile: 8000 pentru FastAPI, 8001 pentru MCP
EXPOSE 8000 8001