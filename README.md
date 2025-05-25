# VICTUS Chat System

## Description
Système de chat intelligent basé sur LangChain avec support multi-modèles (Ollama et ChatGPT).

## Structure du Projet

```
gitchat/
├── src/
│   ├── chat/          # Composants principaux du chat
│   ├── llm/           # Interfaces des modèles de langage
│   ├── memory/        # Gestion de la mémoire et du contexte
│   └── core/          # Fonctionnalités de base
├── api/               # Routes et endpoints API
├── tests/             # Tests unitaires et d'intégration
├── config/            # Fichiers de configuration
├── logs/              # Fichiers de logs
├── data/              # Données et bases de données
├── ui/                # Interfaces utilisateur
└── utils/             # Utilitaires
```

## Liste Détaillée des Fichiers

### Fichiers Principaux
| Nom du fichier | Chemin | Fonction |
|---------------|---------|----------|
| main.py | /gitchat/main.py | Point d'entrée principal, gestion des modes CLI et API |
| requirements.txt | /gitchat/requirements.txt | Liste des dépendances du projet |
| README.md | /gitchat/README.md | Documentation du projet |

### Module Chat (/gitchat/src/chat/)
| Nom du fichier | Fonction |
|---------------|----------|
| __init__.py | Export des classes VictusChatInterface et ConversationHandler |
| chat_interface.py | Interface principale du chat, gestion des interactions utilisateur |
| conversation_handler.py | Gestion des conversations, détection des salutations, contexte |

### Module LLM (/gitchat/src/llm/)
| Nom du fichier | Fonction |
|---------------|----------|
| __init__.py | Export de la classe LLMInterface |
| llm_interface.py | Interface unifiée pour Ollama et ChatGPT, gestion des modèles |

### Module Core (/gitchat/src/core/)
| Nom du fichier | Fonction |
|---------------|----------|
| __init__.py | Export des classes de base (reason, MemoireVectorielle, ReponseCache) |
| reasoner.py | Moteur de raisonnement, logique métier, gestion des réponses |

### Module Memory (/gitchat/src/memory/)
| Nom du fichier | Fonction |
|---------------|----------|
| __init__.py | Export de la classe MemoryHandler |
| memory_handler.py | Gestion de la mémoire vectorielle et du contexte |

## Composants Principaux

### 1. Chat
- Interface de chat principale
- Gestion des conversations
- Traitement des messages
- Interface utilisateur (console et GUI)

### 2. LLM
- Interface avec Ollama
- Interface avec ChatGPT
- Gestion des modèles
- Génération de réponses

### 3. Mémoire et Contexte
- Mémoire vectorielle (ChromaDB)
- Gestion du contexte
- Historique des conversations
- Base de connaissances

### 4. API et Intégrations
- Endpoints REST
- Intégration email
- Webhooks
- Authentification

## Configuration

### Variables d'Environnement (.env)
```
# Configuration des API
OPENAI_API_KEY=votre_clé_api_openai
OLLAMA_HOST=http://localhost:11434

# Configuration de la base de données
CHROMA_DB_PATH=./data/chroma
VECTOR_DB_TYPE=chroma

# Configuration du serveur
HOST=0.0.0.0
PORT=8000
DEBUG=True
ENVIRONMENT=development

# Configuration des logs
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
ERROR_LOG_FILE=./logs/error.log

# Configuration de la sécurité
JWT_SECRET_KEY=votre_clé_secrète_jwt
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Configuration du chat
DEFAULT_MODEL=llama3:latest
MAX_HISTORY_LENGTH=10
TEMPERATURE=0.7
MAX_TOKENS=1000

# Configuration de la mémoire
MEMORY_TYPE=vectorstore
CACHE_DURATION=86400  # 24 heures en secondes
MAX_CACHE_ITEMS=1000
```

### Dépendances Principales
```python
# LangChain et composants
langchain>=0.1.0
langchain-community>=0.0.10
langchain-core>=0.1.0
langchain-openai>=0.0.5

# Modèles de langage
openai>=1.10.0
ollama>=0.1.4
tiktoken>=0.5.2

# Base de données vectorielle
chromadb>=0.4.22
sentence-transformers>=2.3.1

# API et Web
fastapi>=0.109.0
uvicorn>=0.27.0
python-dotenv>=1.0.0
requests>=2.31.0
aiohttp>=3.9.0
httpx>=0.26.0

# Interface utilisateur
tkinter  # Inclus dans Python standard
PyQt6>=6.6.1  # Optionnel pour GUI avancée

# Traitement des données
pydantic>=2.6.0
numpy>=1.24.0
pandas>=2.1.0

# Sécurité et logging
python-jose>=3.3.0
passlib>=1.7.4
bcrypt>=4.1.0
python-multipart>=0.0.6
logging>=0.5.1.2

# Tests
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0

# Utilitaires
python-dateutil>=2.8.2
pytz>=2024.1
tqdm>=4.66.0
```

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-repo/gitchat.git
cd gitchat
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement :
```bash
cp .env.example .env
# Éditer .env avec vos clés API
```

5. Créer les dossiers nécessaires :
```bash
mkdir -p data/chroma logs
```

## Utilisation

### Démarrer le Chat (Mode CLI)
```bash
python main.py --mode cli
```

### Démarrer l'API
```bash
python main.py --mode api
# ou
uvicorn api.main:app --reload
```

### Exécuter les Tests
```bash
pytest tests/
```

## Fonctionnalités

1. **Chat Intelligent**
   - Détection de contexte
   - Gestion des salutations
   - Support multilingue
   - Historique des conversations

2. **Modèles de Langage**
   - Support Ollama (local)
   - Support ChatGPT (cloud)
   - Fallback automatique
   - Gestion de la température

3. **Mémoire et Contexte**
   - Stockage vectoriel
   - Recherche sémantique
   - Persistance des données
   - Cache des réponses

4. **Sécurité**
   - Protection des données
   - Validation des entrées
   - Logs de sécurité
   - Gestion des accès

## Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## Licence

MIT License 