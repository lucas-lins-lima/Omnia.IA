"""
Ponto de entrada principal da API Omnia.IA.
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
from api.v1.endpoints.text_endpoints import router as text_router
from api.v1.endpoints.image_endpoints import router as image_router
from api.v1.endpoints.audio_endpoints import router as audio_router
from api.v1.endpoints.video_endpoints import router as video_router
from api.v1.endpoints.orchestrator_endpoints import router as orchestrator_router
from api.v1.endpoints.tasks_endpoints import router as tasks_router
from api.v1.endpoints.storage_endpoints import router as storage_router

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("omnia_ia.log")
    ]
)

logger = logging.getLogger(__name__)

# Importar routers
from api.v1.endpoints.text_endpoints import router as text_router

app = FastAPI(
    title="Omnia.IA API",
    description="API para a plataforma de IA generativa multimodal Omnia.IA",
    version="0.1.0"
)

# Middleware CORS para permitir requisições cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para produção, especifique os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para logging de requisições
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Registrar início da requisição
    logger.info(f"Requisição iniciada: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Calcular tempo de processamento
    process_time = time.time() - start_time
    
    # Registrar conclusão da requisição
    logger.info(
        f"Requisição concluída: {request.method} {request.url.path} "
        f"(Status: {response.status_code}, Tempo: {process_time:.4f}s)"
    )
    
    # Adicionar cabeçalho com o tempo de processamento
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Incluir routers
app.include_router(text_router)
app.include_router(image_router)
app.include_router(audio_router)
app.include_router(video_router)
app.include_router(orchestrator_router)
app.include_router(tasks_router)
app.include_router(storage_router)

# Rota raiz
@app.get("/")
async def root():
    return {
        "message": "Bem-vindo à API da Omnia.IA",
        "documentation": "/docs",
        "version": "0.1.0"
    }

# Executar a aplicação com uvicorn quando este script for executado diretamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
