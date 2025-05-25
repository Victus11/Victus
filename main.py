"""
Point d'entrée principal de l'application VICTUS Chat
"""
import os
import logging
from dotenv import load_dotenv
import argparse
from src.core.config import setup_logging
from src.chat.chat_interface import VictusChatInterface
from src.llm.llm_interface import LLMInterface
from src.memory.memory_handler import MemoryHandler
from fastapi import FastAPI
import uvicorn

# Chargement des variables d'environnement
load_dotenv()

def setup_app():
    """Configure et initialise l'application"""
    # Configuration des logs
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Démarrage de l'application VICTUS Chat")
    
    # Initialisation des composants
    llm = LLMInterface(
        model_name=os.getenv("DEFAULT_MODEL", "llama3:latest")
    )
    memory = MemoryHandler(
        db_path=os.getenv("CHROMA_DB_PATH", "./data/chroma")
    )
    chat = VictusChatInterface(llm=llm, memory=memory)
    
    return chat

def start_cli():
    """Démarre l'interface en ligne de commande"""
    chat = setup_app()
    chat.start_console()

def start_api():
    """Démarre le serveur API"""
    app = FastAPI(
        title="VICTUS Chat API",
        description="API pour le système de chat VICTUS",
        version="1.0.0"
    )
    
    # Import des routes
    from api.routes import chat, auth
    
    # Ajout des routes
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VICTUS Chat System")
    parser.add_argument(
        "--mode",
        choices=["cli", "api"],
        default="cli",
        help="Mode de démarrage (cli ou api)"
    )
    args = parser.parse_args()
    
    if args.mode == "cli":
        start_cli()
    else:
        app = start_api()
        uvicorn.run(
            app,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            reload=os.getenv("DEBUG", "False").lower() == "true"
        ) 