# Base image python officielle
FROM python:3.9-slim

# Dossier de travail dans le container
WORKDIR /app

# Copier les fichiers requirements.txt et installer les dépendances
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de ton projet
COPY . .

# Expose le port 5000 (optionnel, mais good practice)
EXPOSE 5000

# Commande pour démarrer ton app Flask avec gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]