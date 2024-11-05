# Use uma imagem base oficial do Python
FROM python:3.9-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instale as dependências do Python
RUN pip install fastapi pydantic torch transformers

# Copie o restante do código da aplicação para o diretório de trabalho
COPY . .

# Defina o comando padrão para rodar a aplicação
# Substitua 'app.py' pelo nome do seu arquivo principal

RUN pip install fastapi[standard]

CMD ["fastapi","dev","app.py"]