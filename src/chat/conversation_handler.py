"""
Module de gestion des conversations pour VICTUS.
Centralise la logique de traitement des messages, contextes et salutations.
"""
from typing import Dict, List, Optional, Tuple
import re
import difflib

GREETINGS = ["bonjour", "salut", "hello", "coucou", "bonsoir"]

def is_greeting(text: str) -> bool:
    txt = text.lower().strip()
    # 1) Détection par sous-chaîne
    if any(g in txt for g in GREETINGS):
        return True
    # 2) Fuzzy-match pour couvrir les typos (ex. "bonjou")
    return bool(difflib.get_close_matches(txt, GREETINGS, n=1, cutoff=0.8))

class ConversationHandler:
    def __init__(self):
        self.greeting_patterns = [
            r"^(bonjour|salut|hey|hello|hi|coucou)[\s!]*$",
            r"^bon(jour|soir|matin)[\s!]*$",
            r"^(bonsoir|bonne\s+soirée)[\s!]*$"
        ]
        
    def is_greeting(self, text: str) -> bool:
        """Détecte si le texte est une salutation simple."""
        text = text.lower().strip()
        return any(re.match(pattern, text) for pattern in self.greeting_patterns)

    def extract_conversation_context(self, 
                                  message: str, 
                                  history: List[Dict[str, str]] = None) -> Dict[str, any]:
        """
        Extrait le contexte de la conversation à partir du message et de l'historique.
        
        Args:
            message: Le message actuel
            history: L'historique des messages précédents
            
        Returns:
            Dict contenant le contexte de la conversation
        """
        context = {
            "message_length": len(message.split()),
            "is_greeting": self.is_greeting(message),
            "has_question_mark": "?" in message,
            "previous_context": None
        }
        
        if history and len(history) > 0:
            last_message = history[-1].get("content", "")
            context["previous_context"] = {
                "was_greeting": self.is_greeting(last_message),
                "had_question": "?" in last_message
            }
            
        return context

    def get_temperature(self, input_text: str) -> float:
        """
        Détermine la température appropriée en fonction de la longueur du texte.
        
        Args:
            input_text: Le texte d'entrée
            
        Returns:
            float: La température à utiliser (0.2 pour textes courts, 1.0 sinon)
        """
        if len(input_text.split()) < 3:
            return 0.2
        return 1.0

    def validate_response(self, 
                        input_text: str, 
                        response: str, 
                        context: Dict[str, any]) -> Tuple[bool, Optional[str]]:
        """
        Valide la réponse générée en fonction du contexte.
        
        Args:
            input_text: Le texte d'entrée original
            response: La réponse générée
            context: Le contexte de la conversation
            
        Returns:
            Tuple[bool, Optional[str]]: (est_valide, raison_si_invalide)
        """
        # Pour les salutations, vérifier que la réponse est courte et appropriée
        if context["is_greeting"]:
            if len(response.split()) > 10:
                return False, "Réponse trop longue pour une salutation"
            if not self.is_greeting(response):
                return False, "Réponse inadaptée à une salutation"
                
        # Vérifier la cohérence question/réponse
        if context["has_question_mark"] and not any(char in response for char in ".!?"):
            return False, "Réponse incomplète à une question"
            
        return True, None

    def on_send(self, question: str) -> str:
        """
        Génère une réponse à partir d'une question.
        
        Args:
            question: La question posée
            
        Returns:
            str: La réponse générée
        """
        # Vérifie si c'est une salutation
        if self.is_greeting(question):
            return "Victus : Bonjour ! Comment puis-je vous aider ?"
        else:
            # Implémentation de la génération de la réponse
            # Cette partie doit être implémentée en fonction de la logique de génération de réponse
            # Pour l'instant, nous allons utiliser une réponse par défaut
            return "Victus : Je ne comprends pas la question. Pouvez-vous réécrire ?" 