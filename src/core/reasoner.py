print("[DEBUG] Chargement de VICTUS_reasoner.py — greeting patch actif")

import datetime
import subprocess
import difflib
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.orchestrator import VictusOrchestrator
import json
from datetime import timedelta
import os
import time
import re

class MemoireVectorielle:
    def __init__(self, persist_path="./chroma_data"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.persist_path = persist_path
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection("victus_memoire")
            self.use_chroma = True
        except Exception as e:
            print(f"[Mémoire] Erreur ChromaDB : {e} (fallback RAM)")
            self.use_chroma = False
            self._data = []
        
    def ajouter(self, type_info, entite, information, source):
        # Conversion des métadonnées en chaînes (pas de None pour ChromaDB)
        meta = {
            "type_info": str(type_info) if type_info is not None else "",
            "entite": str(entite) if entite is not None else "",
            "source": str(source) if source is not None else "",
            "date": str(datetime.now())
        }
        if self.use_chroma:
            emb = self.model.encode([information])[0].tolist()
            self.collection.add(
                documents=[information],
                embeddings=[emb],
                metadatas=[meta],
                ids=[str(hash((meta['type_info'], meta['entite'], information, meta['source'], meta['date'])))]
            )
        else:
            self._data.append({
                'type_info': meta['type_info'],
                'entite': meta['entite'],
                'information': information,
                'source': meta['source'],
                'date': meta['date']
            })

    def rechercher(self, type_info, entite=None, requete=None, top_k=3):
        if self.use_chroma:
            if not requete:
                return []
            emb = self.model.encode([requete])[0].tolist()
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=top_k,
                where={"type_info": type_info} if type_info else None
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            # Filtrage entite si précisé
            if entite:
                docs = [doc for doc, meta in zip(docs, metas) if meta.get("entite") == entite]
            return docs
        else:
            resultats = []
            for entry in self._data:
                if entry['type_info'] != type_info:
                    continue
                if entite is not None and entry['entite'] != entite:
                    continue
                if requete is not None and requete.lower() not in entry['information'].lower():
                    continue
                resultats.append(entry['information'])
                if len(resultats) >= top_k:
                    break
            return resultats

# Dictionnaires pour les jours de la semaine en français et les mois en français
days_fr = {
    "Lundi": 0,
    "Mardi": 1,
    "Mercredi": 2,
    "Jeudi": 3,
    "Vendredi": 4,
    "Samedi": 5,
    "Dimanche": 6
}
months_fr = {
    "Janvier": 1,
    "Février": 2,
    "Mars": 3,
    "Avril": 4,
    "Mai": 5,
    "Juin": 6,
    "Juillet": 7,
    "Août": 8,
    "Septembre": 9,
    "Octobre": 10,
    "Novembre": 11,
    "Décembre": 12
}

def nettoyer_texte_web(texte):
    """Nettoie le texte extrait du web."""
    if not texte:
        return ""
    
    try:
        # Conversion en unicode si nécessaire
        if isinstance(texte, bytes):
            texte = texte.decode('utf-8', errors='ignore')
            
        # Suppression des caractères non imprimables
        texte = ''.join(char for char in texte if char.isprintable() or char in ['\n', '\t'])
        
        # Correction des caractères spéciaux mal encodés
        corrections = {
            "Ã©": "é", "Ã¨": "è", "Ã ": "à",
            "Ãª": "ê", "Ã®": "î", "Ã´": "ô",
            "Ã»": "û", "Ã§": "ç", "Å": "œ",
            "â€™": "'", "â€\"": "-", "â€œ": "\"",
            "â€": "\"", "Â«": "«", "Â»": "»",
            "Â": " ", "\u0080": "€"
        }
        
        for ancien, nouveau in corrections.items():
            texte = texte.replace(ancien, nouveau)
            
        # Nettoyage des espaces et sauts de ligne
        lignes = [ligne.strip() for ligne in texte.split('\n')]
        texte = '\n'.join(ligne for ligne in lignes if ligne)
        
        # Suppression des doublons de lignes
        lignes = []
        for ligne in texte.split('\n'):
            if ligne not in lignes[-3:]:  # Évite les répétitions proches
                lignes.append(ligne)
        
        return '\n'.join(lignes)
        
    except Exception as e:
        print(f"[ERREUR] Nettoyage texte : {str(e)}")
        return texte

def extraire_texte_page(url):
    """Extrait le texte principal d'une page web."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; VICTUSBot/1.0)'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'lxml')
        
        # Suppression des éléments non pertinents
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
            tag.decompose()
            
        # Extraction du texte principal
        textes = []
        for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
            texte = p.get_text().strip()
            if len(texte) > 40:  # Filtre les textes trop courts
                textes.append(nettoyer_texte_web(texte))
                
        return "\n".join(textes[:30])  # Limite à 30 paragraphes
        
    except Exception as e:
        return f"[Erreur lors de l'extraction de {url} : {e}]"

def recherche_web(query):
    """Recherche web approfondie avec multiples sources."""
    try:
        with DDGS() as ddgs:
            # Préparation des requêtes spécialisées
            requetes = [
                query,  # Requête principale
                f"{query} avis clients commentaires",  # Avis clients
                f"{query} histoire entreprise fondation",  # Histoire
                f"{query} produits services gamme",  # Produits
                f"{query} valeurs engagements responsabilité"  # Valeurs
            ]
            
            resultats_combines = []
            sources_vues = set()
            
            for requete in requetes:
                results = [r for r in ddgs.text(requete, max_results=5)]
                for r in results:
                    if r['href'] not in sources_vues:
                        texte = extraire_texte_page(r['href'])
                        if texte and not texte.startswith('[Erreur'):
                            resultats_combines.append({
                                'source': r['href'],
                                'titre': r['title'],
                                'texte': texte,
                                'type': 'web'
                            })
                            sources_vues.add(r['href'])
            
            return resultats_combines

    except Exception as e:
        print(f"Erreur lors de la recherche web : {str(e)}")
        return []

def recherche_documentation_technique(query):
    """Recherche spécifique pour la documentation technique."""
    try:
        # Sites de documentation prioritaires
        sites_docs = [
            "github.com/KillianLucas/open-interpreter",
            "docs.openinterpreter.com",
            "pypi.org/project/open-interpreter"
        ]
        
        with DDGS() as ddgs:
            # Construction de la requête avec restriction aux sites de doc
            search_query = f"{query} site:({' OR site:'.join(sites_docs)})"
            results = [r for r in ddgs.text(search_query, max_results=5)]
            
            if not results:
                return "Je n'ai pas trouvé de documentation technique correspondant à votre requête."
            
            # Extraction et synthèse des informations
            synthese = []
            for r in results:
                texte = extraire_texte_page(r['href'])
                if texte and not texte.startswith('[Erreur'):
                    synthese.append(f"Source : {r['href']}\n{texte}")
            
            if not synthese:
                return "Je n'ai pas pu extraire le contenu des pages de documentation trouvées."
            
            return "\n\n".join(synthese)
            
    except Exception as e:
        return f"Erreur lors de la recherche de documentation : {str(e)}"

# Cache pour les réponses fréquentes
class ReponseCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReponseCache, cls).__new__(cls)
            cls._instance.init()
        return cls._instance
    
    def init(self):
        self.cache_file = "reponse_cache.json"
        self.duree_cache = 24 * 3600  # 24 heures en secondes
        self.cache = self._charger_cache()
    
    def _charger_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                # Nettoyage des entrées expirées
                maintenant = time.time()
                cache = {
                    k: v for k, v in cache.items() 
                    if maintenant - float(v['timestamp']) < self.duree_cache
                }
                return cache
            except:
                return {}
        return {}
    
    def _sauver_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[INFO] Erreur sauvegarde cache : {e}")
    
    def obtenir(self, question):
        try:
            if question in self.cache:
                timestamp = float(self.cache[question]['timestamp'])
                if time.time() - timestamp < self.duree_cache:
                    return self.cache[question]['reponse']
                else:
                    # Suppression de l'entrée expirée
                    del self.cache[question]
                    self._sauver_cache()
        except Exception as e:
            print(f"[INFO] Erreur lecture cache : {e}")
        return None
    
    def stocker(self, question, reponse):
        try:
            self.cache[question] = {
                'reponse': reponse,
                'timestamp': str(time.time())
            }
            self._sauver_cache()
        except Exception as e:
            print(f"[INFO] Erreur stockage cache : {e}")

def get_available_models():
    """Récupère la liste des modèles Ollama disponibles."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return [m["name"].split(":")[0] for m in data]
            return []
    except Exception as e:
        print(f"[INFO] Erreur récupération modèles Ollama : {e}")
        return []

def get_llm_response(prompt, contexte=None, max_retries=3):
    """
    Obtient une réponse du modèle LLM avec contexte optionnel.
    
    Args:
        prompt (str): La question ou prompt
        contexte (dict, optional): Contexte de la question
        max_retries (int): Nombre maximum de tentatives
    """
    try:
        # Préparation du prompt avec contexte
        prompt_final = prompt
        if contexte:
            prompt_final = f"Contexte: {json.dumps(contexte, ensure_ascii=False)}\nQuestion: {prompt}"
            
        # Appel au modèle via l'API Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt_final,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000,
                    "stop": ["<|im_end|>", "."]
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                return data["response"].strip()
        return None
        
    except Exception as e:
        print(f"Erreur LLM : {e}")
        if max_retries > 0:
            time.sleep(1)
            return get_llm_response(prompt, contexte, max_retries - 1)
        return None

def formater_reponse_chat(reponse):
    """Formate la réponse pour un affichage optimal dans le chat."""
    try:
        # Nettoyage initial du texte
        reponse = nettoyer_texte_web(reponse)
        
        # Supprime les balises HTML tout en conservant la structure
        reponse_clean = re.sub(r'<div[^>]*>', '', reponse)
        reponse_clean = re.sub(r'</div>', '', reponse_clean)
        reponse_clean = re.sub(r'<h1[^>]*>', '\n\n# ', reponse_clean)
        reponse_clean = re.sub(r'</h1>', '\n', reponse_clean)
        reponse_clean = re.sub(r'<h2[^>]*>', '\n\n## ', reponse_clean)
        reponse_clean = re.sub(r'</h2>', '\n', reponse_clean)
        reponse_clean = re.sub(r'<h3[^>]*>', '\n\n### ', reponse_clean)
        reponse_clean = re.sub(r'</h3>', '\n', reponse_clean)
        reponse_clean = re.sub(r'<p[^>]*>', '\n', reponse_clean)
        reponse_clean = re.sub(r'</p>', '\n', reponse_clean)
        reponse_clean = re.sub(r'<[^>]+>', '', reponse_clean)
        
        # Supprime les placeholders vides et les caractères spéciaux
        reponse_clean = re.sub(r'\{[^}]*\}', '', reponse_clean)
        reponse_clean = re.sub(r'[^\w\s\n#\-.,;:?!()\[\]{}\'\"€$£¥%@&+=/<>]', '', reponse_clean)
        
        # Nettoie les espaces et sauts de ligne
        lignes = [ligne.strip() for ligne in reponse_clean.split('\n')]
        sections = []
        section_courante = []
        
        for ligne in lignes:
            if ligne.startswith(('#', '---')):
                if section_courante:
                    sections.append('\n'.join(section_courante).strip())
                section_courante = [ligne]
            elif ligne:
                section_courante.append(ligne)
                
        if section_courante:
            sections.append('\n'.join(section_courante).strip())
            
        # Reconstruit la réponse avec un formatage propre
        reponse_clean = '\n\n---\n\n'.join(sections)
        
        # Vérifie la présence de contenu valide
        if not any(c.isalnum() for c in reponse_clean):
            return "Désolé, je n'ai pas pu générer une réponse valide. Veuillez réessayer."
            
        return reponse_clean.strip()
        
    except Exception as e:
        print(f"[ERREUR] Formatage réponse : {str(e)}")
        return "Désolé, une erreur est survenue lors du formatage de la réponse. Veuillez réessayer."

def is_greeting(text: str) -> bool:
    """Détection de salutation, avec fuzzy-match pour typo."""
    GREETINGS = ["bonjour", "salut", "hello", "coucou", "bonsoir"]
    txt = text.lower().strip()
    # 1) Sous-chaîne
    if any(g in txt for g in GREETINGS):
        return True
    # 2) Fuzzy match (couplé à cutoff=0.8)
    close = difflib.get_close_matches(txt, GREETINGS, n=1, cutoff=0.8)
    return bool(close)

def reason(question, chat_history=None):
    """
    Fonction principale de raisonnement de VICTUS.
    """
    if chat_history is None:
        chat_history = []
    
    # 1. Salutations prioritaires (y compris typo comme "bonjou")
    if is_greeting(question):
        return "Bonjour ! Comment puis-je vous aider ?"
        
    # Préparation de la question
    q = question.lower().strip()
    
    # 2. Gestion date/heure locale
    if any(mot in q for mot in ["heure", "date", "jour", "time", "today"]):
        now = datetime.datetime.now()
    
    # 1. Analyse du contexte
    contexte = analyser_contexte_question(question)
    
    # 2. Réponse LLM avec contexte
    reponse_llm = get_llm_response(question, contexte=contexte)
    confiance_llm = evaluer_confiance_reponse(reponse_llm) if reponse_llm else 0.0
    
    # 3. Consultation mémoire
    reponse_memoire = memoire.rechercher(
        type_info="reponse", 
        requete=question
    )
    confiance_memoire = evaluer_confiance_reponse(reponse_memoire[0]) if reponse_memoire else 0.0
    
    # 4. Recherche web si nécessaire
    date_memoire = extraire_date_memoire(reponse_memoire[0]) if reponse_memoire else None
    if needs_web_validation(confiance_llm, confiance_memoire, date_memoire):
        resultats_web = recherche_web(question)
        confiance_web = evaluer_confiance_web(resultats_web)
    else:
        resultats_web = []
        confiance_web = 0.0
    
    # 5. Génération réponse structurée
    reponse_html = generer_reponse_finale(
        question, reponse_llm, confiance_llm,
        reponse_memoire[0] if reponse_memoire else None, confiance_memoire,
        resultats_web, confiance_web,
        contexte=contexte
    )
    
    # 6. Formatage pour le chat
    return formater_reponse_chat(reponse_html)

def analyser_contexte_question(question):
    """Analyse le contexte de la question pour mieux cibler la réponse."""
    contexte = {
        'type': 'general',  # general, technique, culturel, etc.
        'complexite': 'simple',  # simple, moyenne, complexe
        'domaine': None,  # domaine spécifique si identifié
        'filtres': {}  # filtres pour la recherche
    }
    
    # Détection du type de question
    if any(mot in question.lower() for mot in ['comment', 'pourquoi', 'expliquer']):
        contexte['complexite'] = 'moyenne'
    
    # Détection du domaine
    domaines = {
        'technique': ['logiciel', 'programme', 'code', 'bug'],
        'culturel': ['bonjour', 'salut', 'culture', 'français'],
        'commercial': ['prix', 'coût', 'achat', 'vente']
    }
    
    for domaine, mots_cles in domaines.items():
        if any(mot in question.lower() for mot in mots_cles):
            contexte['domaine'] = domaine
            contexte['type'] = domaine
            break
    
    return contexte

def recherche_web_contextuelle(question, contexte):
    """Effectue une recherche web adaptée au contexte."""
    # Adaptation des termes de recherche selon le contexte
    if contexte.get('domaine') == 'culturel':
        question = f"culture française {question}"
    elif contexte.get('domaine') == 'technique':
        question = f"documentation technique {question}"
    
    return recherche_web(question)

def evaluer_confiance_reponse(reponse):
    """Évalue la confiance d'une réponse sur une échelle de 0 à 1."""
    if not reponse:
        return 0
    # Indicateurs de confiance
    mots_confiance = ["certainement", "assurément", "clairement", "évidemment"]
    mots_doute = ["peut-être", "probablement", "possiblement", "je pense"]
    mots_incertitude = ["je ne suis pas sûr", "incertain", "difficile à dire"]
    
    confiance = 0.5  # Base neutre
    for mot in mots_confiance:
        if mot in reponse.lower():
            confiance += 0.1
    for mot in mots_doute:
        if mot in reponse.lower():
            confiance -= 0.05
    for mot in mots_incertitude:
        if mot in reponse.lower():
            confiance -= 0.1
            
    return max(0.1, min(1.0, confiance))

def needs_web_validation(confiance_llm, confiance_memoire, date_memoire):
    """Détermine si une validation web est nécessaire."""
    if confiance_llm >= 0.8:
        return False
        
    if date_memoire:
        age_jours = (datetime.now() - date_memoire).days
        if age_jours < 7 and confiance_memoire >= 0.7:
            return False
            
    return True

def evaluer_confiance_web(resultats):
    """Évalue la confiance des résultats web."""
    if not resultats or not isinstance(resultats, list):
        return 0.0
    
    confiance = 0.5  # Base neutre
    
    # Analyse du nombre de sources
    nb_sources = len(resultats)
    confiance += min(0.3, nb_sources * 0.1)  # Bonus pour chaque source pertinente
    
    # Analyse des sources
    for resultat in resultats:
        if not isinstance(resultat, dict):
            continue
            
        url = resultat.get('source', '').lower()
        texte = resultat.get('texte', '')
        
        # Bonus pour les sources fiables
        sources_fiables = [".gouv.fr", ".edu", ".org", "wikipedia.org", "sante.fr"]
        for source in sources_fiables:
            if source in url:
                confiance += 0.1
                break
        
        # Malus pour les sources moins fiables
        sources_douteuses = ["blog.", "forum.", ".com/forum"]
        for source in sources_douteuses:
            if source in url:
                confiance -= 0.1
                break
        
        # Bonus pour le contenu substantiel
        if len(texte) > 1000:
            confiance += 0.1
        elif len(texte) > 500:
            confiance += 0.05
            
    return max(0.1, min(0.9, confiance))  # Limite entre 0.1 et 0.9

def extraire_date_memoire(reponse_memoire):
    """Extrait la date d'une réponse de la mémoire."""
    try:
        return datetime.strptime(reponse_memoire['date'], '%Y-%m-%d %H:%M:%S')
    except:
        return None

def analyser_et_synthetiser(sources):
    """Analyse approfondie et synthèse intelligente des informations."""
    # Structure pour stocker les informations analysées
    analyse = {
        "nom": None,
        "description": [],
        "historique": [],
        "produits": [],
        "expertise": [],
        "valeurs": [],
        "chiffres": [],
        "avis_clients": [],
        "sources": set()
    }
    
    # Analyse par source
    for source in sources:
        if isinstance(source, dict):  # Nouvelle structure de source
            texte = source['texte']
            analyse["sources"].add(source['source'])
        else:
            texte = source
        
        lignes = texte.split('\n')
        for ligne in lignes:
            ligne = ligne.strip()
            if not ligne or ligne.startswith(("http", "Source", "Titre")):
                continue
            
            # Analyse sémantique de la ligne
            mots = ligne.lower().split()
            
            # Détection intelligente du nom
            if not analyse["nom"]:
                for mot in ligne.split():
                    if mot[0].isupper() and len(mot) > 2 and not any(c.isdigit() for c in mot):
                        if any(indic in ligne.lower() for indic in ["marque", "société", "entreprise", "laboratoire"]):
                            analyse["nom"] = mot
                            break
            
            # Classification intelligente du contenu
            if len(ligne) < 10:  # Ignorer les lignes trop courtes
                continue
                
            # Analyse contextuelle
            if any(mot in mots for mot in ["créé", "fondé", "depuis", "création", "histoire"]):
                if len(ligne) > 30:  # Filtre les mentions trop courtes
                    analyse["historique"].append(ligne)
            
            elif any(mot in mots for mot in ["produit", "gamme", "complément", "offre"]):
                if not any(p in ligne for p in ["SIRET", "NAF", "APE"]):
                    if len(ligne) > 40:  # Description substantielle
                        analyse["produits"].append(ligne)
            
            elif any(mot in mots for mot in ["expert", "spécialisé", "savoir-faire", "compétence"]):
                if len(ligne) > 50:  # Description détaillée
                    analyse["expertise"].append(ligne)
            
            elif any(mot in mots for mot in ["valeur", "engagement", "mission", "éthique", "responsable"]):
                if len(ligne) > 40:
                    analyse["valeurs"].append(ligne)
            
            elif any(mot in mots for mot in ["million", "chiffre", "vente", "réseau", "croissance"]):
                if not any(c in ligne for c in ["SIRET", "NAF", "APE"]):
                    analyse["chiffres"].append(ligne)
            
            elif any(mot in mots for mot in ["avis", "client", "satisfaction", "recommande"]):
                if len(ligne) > 30:
                    analyse["avis_clients"].append(ligne)
            
            elif len(ligne) > 60 and not any(mot in ligne.lower() for mot in ["cookie", "navigation", "javascript"]):
                analyse["description"].append(ligne)
    
    # Dédoublonnage et nettoyage
    for key in analyse:
        if isinstance(analyse[key], list):
            # Supprimer les doublons tout en préservant l'ordre
            seen = set()
            analyse[key] = [x for x in analyse[key] if not (x.lower() in seen or seen.add(x.lower()))]
            
            # Trier par pertinence (longueur comme indicateur simple)
            analyse[key].sort(key=len, reverse=True)
            
            # Limiter le nombre d'éléments
            analyse[key] = analyse[key][:3]
    
    return analyse

def synthetiser_texte(infos, section):
    """Crée une synthèse cohérente pour une section donnée."""
    if not infos.get(section):
        return ""
        
    elements = infos[section]
    if not elements:
        return ""
        
    # Extraction des informations clés
    mots_cles = set()
    for element in elements:
        mots = [mot.lower() for mot in element.split() if len(mot) > 3]
        mots_cles.update(mots)
    
    # Création d'une synthèse
    if section == "historique":
        dates = re.findall(r'\b(19|20)\d{2}\b', ' '.join(elements))
        if dates:
            return f"Fondée en {min(dates)}, " + elements[0].lower()
        return elements[0]
    
    elif section == "produits":
        produits = []
        for element in elements:
            produits.extend(re.findall(r'([A-Z][a-zA-Z-]+(?: [A-Z][a-zA-Z-]+)*)', element))
        if produits:
            return f"La société propose notamment : {', '.join(produits[:3])}"
        return elements[0]
    
    elif section == "expertise":
        return "Spécialisée dans " + elements[0].lower()
    
    elif section == "valeurs":
        valeurs = re.findall(r'([a-zA-Z]+(?:-[a-zA-Z]+)*)', ' '.join(elements))
        valeurs = [v for v in valeurs if len(v) > 5]
        if valeurs:
            return f"Les valeurs fondamentales sont : {', '.join(valeurs[:3])}"
        return elements[0]
    
    return elements[0]

def generer_reponse_finale(question, reponse_llm, confiance_llm, 
                          reponse_memoire, confiance_memoire,
                          reponse_web, confiance_web,
                          contexte=None):
    """Génère une réponse finale structurée et analysée."""
    # Collecte des informations
    sources = []
    if reponse_web: 
        sources.extend(reponse_web if isinstance(reponse_web, list) else [reponse_web])
    if reponse_memoire: 
        sources.append(reponse_memoire)
    if reponse_llm: 
        sources.append(reponse_llm)

    if not sources:
        return "Je ne dispose pas d'informations suffisantes pour répondre à cette question."

    # Analyse approfondie
    infos = analyser_et_synthetiser(sources)
    
    # Construction de la réponse
    reponse = []
    
    # Titre et introduction
    if infos["nom"]:
        reponse.append(f"# {infos['nom']}")
    else:
        reponse.append("# À propos de votre recherche")
        
    if infos["description"]:
        reponse.append(infos["description"][0])
    
    # Sections principales
    sections = {
        "Histoire et Développement": infos["historique"],
        "Expertise et Produits": infos["expertise"] + infos["produits"],
        "Valeurs et Engagements": infos["valeurs"]
    }
    
    for titre, contenu in sections.items():
        if contenu:
            reponse.append(f"# {titre}")
            reponse.append('\n'.join(contenu[:2]))  # Limite à 2 éléments par section
            
    # Chiffres clés si disponibles
    if infos["chiffres"]:
        reponse.append("# Chiffres Clés")
        reponse.append('\n'.join(infos["chiffres"][:2]))
    
    return '\n\n'.join(reponse)

def handle_system_commands(q, orchestrator):
    """Gère les commandes système."""
    if q.startswith("execute "):
        cmd_name = q.split(" ")[1]
        args = q.split(" ")[2:] if len(q.split(" ")) > 2 else None
        try:
            result = orchestrator.execute_command(cmd_name, args)
            return f"Commande {cmd_name} exécutée avec succès:\n{result}"
        except Exception as e:
            return f"Erreur lors de l'exécution de la commande {cmd_name}: {str(e)}"
    elif q.startswith("stop "):
        cmd_name = q.split(" ")[1]
        try:
            orchestrator.stop_command(cmd_name)
            return f"Commande {cmd_name} arrêtée"
        except Exception as e:
            return f"Erreur lors de l'arrêt de la commande {cmd_name}: {str(e)}"
    elif q.startswith("status"):
        cmd_name = q.split(" ")[1] if len(q.split(" ")) > 1 else None
        status = orchestrator.get_status(cmd_name)
        if isinstance(status, dict):
            return "État des commandes:\n" + "\n".join(f"- {k}: {v}" for k, v in status.items())
        return f"État de la commande {cmd_name}: {status}"
    elif q.startswith("register "):
        parts = q.split(" ")
        if len(parts) < 3:
            return "Format: register <nom> <commande>"
        name = parts[1]
        command = " ".join(parts[2:])
        try:
            orchestrator.register_command(name, command)
            return f"Commande {name} enregistrée avec succès"
        except Exception as e:
            return f"Erreur lors de l'enregistrement de la commande: {str(e)}"

# Utilisation de la classe MemoireVectorielle pour retrouver les informations dans la base de données vectorielle
memoire = MemoireVectorielle()

def generer_nouveau_code(fichier, question, correction, feedback, modele_ollama="deepseek-coder:latest"):
    """Génère du nouveau code en utilisant un modèle LLM.
    
    Args:
        fichier (str): Chemin du fichier à modifier
        question (str): Question ou description du problème
        correction (str): Description de la correction attendue
        feedback (str): Retour sur la dernière tentative
        modele_ollama (str): Modèle Ollama à utiliser
        
    Returns:
        str: Code généré et validé
    """
    # Lecture du code existant
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            code_existant = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None
        
    # Construction du prompt
    prompt = f"""Tu es un expert en Python chargé de corriger du code.

FICHIER : {fichier}
PROBLÈME : {question}
CORRECTION ATTENDUE : {correction}
FEEDBACK : {feedback}

CODE ACTUEL :
{code_existant}

INSTRUCTIONS :
1. Analyse le code actuel et le problème
2. Propose une correction qui :
   - Respecte la syntaxe Python
   - Inclut les imports nécessaires
   - Gère les erreurs
   - Valide les types d'entrée
3. Retourne UNIQUEMENT le code corrigé, sans explications

CODE CORRIGÉ :"""

    try:
        # Génération du code avec gestion de l'encodage
        process = subprocess.Popen(
            ["ollama", "run", modele_ollama],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8'
        )
        
        stdout, stderr = process.communicate(input=prompt, timeout=180)
        
        if process.returncode != 0:
            print(f"Erreur Ollama : {stderr}")
            return code_existant
            
        nouveau_code = stdout.strip()
        
        # Validation de la syntaxe
        try:
            compile(nouveau_code, filename=fichier, mode='exec')
        except SyntaxError as e:
            print(f"Erreur de syntaxe dans le code généré : {e}")
            return code_existant
            
        # Vérification des imports nécessaires
        if "Union" in nouveau_code and "from typing import Union" not in nouveau_code:
            nouveau_code = "from typing import Union\n\n" + nouveau_code
            
        return nouveau_code
        
    except subprocess.TimeoutExpired:
        print("Timeout lors de la génération du code")
        return code_existant
    except Exception as e:
        print(f"Erreur lors de la génération du code : {e}")
        return code_existant