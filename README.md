# assistente-docs-rag

Assistente que responde dúvidas sobre a documentação da API do Asaas (pagamentos: cobranças, Pix, boletos, webhooks),
usando RAG (Retrieval-Augmented Generation) e avaliação automática (evals).

## Objetivo
Mostrar o ciclo completo de um app de IA em produção:
extração → chunking → embeddings → banco vetorial → respostas com fonte → evals → deploy.

## Fonte de dados
Documentação pública do Asaas: https://docs.asaas.com
Recorte inicial: Autenticação, Cobranças e Webhooks.

## Stack (planejada)
- Python, FastAPI
- PostgreSQL + pgVector
- Langfuse (observabilidade)
- Docker + Cloud Run
- GitHub Actions (evals automáticas)

## Status
🚧 Em construção — semana 1 de 12
