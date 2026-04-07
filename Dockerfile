# Use a lightweight Python 3.10 image (Required by Hackathon rules)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install 'uv' for lightning-fast, lockfile-based package installation
RUN pip install uv

# Copy your dependency files first (for Docker caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using the uv lockfile
RUN uv pip install --system -r pyproject.toml

# Copy the rest of your application code
COPY . .

# Expose the port the FastAPI server runs on
EXPOSE 7860

# The command that starts the environment
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
