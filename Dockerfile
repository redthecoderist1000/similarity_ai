FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install build/runtime deps needed for numpy / BLAS / wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip / wheel to ensure wheel installs
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Pre-install numpy (ensure numpy is available before other packages that import it)
RUN pip install --no-cache-dir numpy

# Install remaining Python dependencies (use CPU torch wheel index)
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]