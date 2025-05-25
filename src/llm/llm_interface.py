"""
Interface unifiée pour les modèles de langage utilisant LangChain.
"""
from typing import Dict, Optional
import aiohttp
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

load_dotenv()

class LLMInterface:
    def __init__(self, model_name: str = "llama3:latest"):
        self.model_name = model_name
        self.temperature = 1.0
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Configuration du prompt de base
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es VICTUS, un assistant de développement intelligent."),
            ("user", "{message}")
        ])
        
        # Configuration des modèles
        self.ollama = Ollama(
            base_url=self.ollama_host,
            model=model_name
        )
        
        if self.openai_api_key:
            self.chatgpt = ChatOpenAI(
                api_key=self.openai_api_key,
                model="gpt-3.5-turbo"
            )
        
        # Création des chaînes de traitement
        self.ollama_chain = RunnableSequence(self.base_prompt | self.ollama)
        if self.openai_api_key:
            self.chatgpt_chain = RunnableSequence(self.base_prompt | self.chatgpt)
            
    async def generate_response(self, 
                              message: str, 
                              context: Dict[str, any]) -> str:
        """
        Génère une réponse en utilisant le modèle principal (Ollama) ou le fallback (ChatGPT).
        
        Args:
            message: Le message utilisateur
            context: Le contexte de la conversation
            
        Returns:
            str: La réponse générée
        """
        try:
            # Ajout du contexte au message
            enhanced_message = self._enhance_message_with_context(message, context)
            
            # Mise à jour de la température pour Ollama
            self.ollama.temperature = self.temperature
            
            # Génération avec Ollama
            response = await self.ollama_chain.ainvoke({
                "message": enhanced_message
            })
            return response.content
            
        except Exception as e:
            print(f"Erreur Ollama: {e}, utilisation du fallback ChatGPT")
            if hasattr(self, 'chatgpt_chain'):
                # Mise à jour de la température pour ChatGPT
                self.chatgpt.temperature = self.temperature
                
                response = await self.chatgpt_chain.ainvoke({
                    "message": message
                })
                return response.content
            raise
            
    def _enhance_message_with_context(self, 
                                    message: str, 
                                    context: Dict[str, any]) -> str:
        """
        Enrichit le message avec le contexte pour une meilleure compréhension.
        """
        context_info = []
        
        if context.get("is_greeting"):
            context_info.append("CONTEXTE: Ceci est une salutation.")
            
        if context.get("has_question_mark"):
            context_info.append("CONTEXTE: Ceci est une question qui nécessite une réponse claire.")
            
        if context.get("previous_context"):
            prev = context["previous_context"]
            if prev.get("was_greeting"):
                context_info.append("CONTEXTE: Le message précédent était une salutation.")
                
        if context_info:
            return f"{' '.join(context_info)}\n\n{message}"
        return message 