# RAG con LangChain y pgvector

Consulta de consola sobre un PDF. El texto se parte en chunks, se convierte en embeddings con OpenAI y se guarda en Postgres. Al preguntar, se recuperan los cuatro fragmentos más cercanos y el modelo responde solo con ese contexto.

## Stack

- Python 3.12 y [uv](https://docs.astral.sh/uv/)
- LangChain, `PyPDFLoader` y `RecursiveCharacterTextSplitter`
- `text-embedding-3-small` y `gpt-4o-mini`
- Postgres con la extensión pgvector

## Configuración

```bash
uv sync
copy .env.example .env
```

En `.env` van `OPENAI_API_KEY` y `DATABASE_URL`. Ese archivo no se sube al repositorio.

Postgres local con la imagen que ya trae pgvector:

```bash
docker run --name pgvector -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag -p 5432:5432 -d pgvector/pgvector:pg16
```

El usuario de esa base es `postgres` y la contraseña es la de `POSTGRES_PASSWORD`. Dentro de la base `rag`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Coloca el PDF en `docs/` (no forma parte del repo) o indica otra ruta con `PDF_PATH`.

## Uso

Indexar una vez:

```bash
uv run python main.py --index
```

Preguntar, sin volver a indexar:

```bash
uv run python main.py "¿En qué bachillerato se realiza el estudio?"
```

Sin argumentos usa esa misma pregunta de ejemplo. Volver a pasar `--index` inserta los chunks otra vez.
