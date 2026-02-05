# 📚 RAG Wissensdatenbank

Ein leistungsstarkes Retrieval-Augmented Generation (RAG) System mit intelligenter Dokumentenverarbeitung, semantischer Suche und Confluence-Integration.

---

## 🌟 Features

- **Multi-Format Dokumentenverarbeitung**: PDF, Word, Excel, PowerPoint, E-Mails, Bilder, HTML, Markdown und mehr
- **Hybrid-Suche**: Kombiniert semantische Vektorsuche mit BM25 Keyword-Suche (RRF)
- **Cross-Encoder Reranking**: Schnelles, deutschsprachiges Reranking für bessere Relevanz
- **LangGraph Workflow**: Intelligente Query-Analyse und -Umschreibung
- **Confluence Integration**: Direkte Suche in Atlassian Confluence
- **Dokument-Generator**: Automatische Dokumentenerstellung aus Inhaltsverzeichnis
- **Moderne Web-Oberfläche**: Streamlit UI mit Chat-Interface
- **REST API**: FastAPI Backend für programmatischen Zugriff

---

## 📋 Voraussetzungen

### System-Anforderungen
- **Python**: 3.10 oder höher
- **RAM**: Mindestens 8 GB empfohlen
- **Speicher**: ~2 GB für Abhängigkeiten und Modelle

### Benötigte API-Keys
- **OpenAI API Key**: Für Embeddings und LLM-Generierung
  - Registrierung: https://platform.openai.com/
- **Confluence API Token** (optional): Für Confluence-Integration
  - Erstellung: Atlassian Profil → Security → API tokens

---

## 🚀 Installation

### 1. Repository klonen
```bash
git clone <repository-url>
cd windsurf-project
```

### 2. Virtuelle Umgebung erstellen (empfohlen)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 4. Umgebungsvariablen konfigurieren
```bash
# .env.example kopieren
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac

# .env Datei bearbeiten und API-Key eintragen
```

**Minimale `.env` Konfiguration:**
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Vollständige `.env` Konfiguration:**
```env
# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002

# RAG Einstellungen
CHROMA_PERSIST_DIRECTORY=./chroma_db
MAX_TOKENS=4000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5

# Confluence (optional)
CONFLUENCE_URL=https://your-domain.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_KEY=your-api-token
CONFLUENCE_SPACE_KEY=MYSPACE
CONFLUENCE_IS_CLOUD=true
```

---

## ▶️ Anwendung starten

### Schritt 1: API-Server starten
```bash
python main.py
```
Der API-Server startet auf **http://localhost:8000**

Erfolgreiche Ausgabe:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Schritt 2: Streamlit-Oberfläche starten
In einem **neuen Terminal**:
```bash
streamlit run app.py
```
Die Web-Oberfläche öffnet sich automatisch auf **http://localhost:8501**

### Beide Dienste gleichzeitig (Windows PowerShell)
```powershell
Start-Process python -ArgumentList "main.py"
Start-Sleep -Seconds 5
streamlit run app.py
```

---

## 🖥️ Benutzeroberfläche

### Übersicht
Die Oberfläche ist in zwei Bereiche unterteilt:
- **Linke Seitenleiste**: Dokumentenverwaltung und Einstellungen
- **Hauptbereich**: Chat, Suche und Dokument-Generator

---

## 📁 Seitenleiste - Quellen hinzufügen

### Tab: 📁 Datei
Dokumente per Drag & Drop oder Dateiauswahl hochladen.

**Unterstützte Formate:**
| Kategorie | Formate |
|-----------|---------|
| Dokumente | PDF, DOCX, DOC, TXT, MD |
| Tabellen | XLSX, XLS, CSV |
| Präsentationen | PPTX, PPT |
| Web | HTML, HTM |
| E-Books | EPUB |
| E-Mails | MSG, EML |
| Bilder | JPG, PNG, GIF (OCR) |
| Daten | JSON, XML |
| Archive | ZIP |

### Tab: 📝 Text
Direktes Einfügen von Text in die Wissensdatenbank.

### Tab: 🔗 URL
Webseiten als Quelle hinzufügen. Der Inhalt wird automatisch extrahiert.

### Tab: 🔷 Confluence
Atlassian Confluence als Quelle konfigurieren:
1. **Confluence URL**: `https://domain.atlassian.net/wiki`
2. **Benutzername/E-Mail**: Ihre Atlassian E-Mail
3. **API Token**: Token aus Atlassian Security Settings
4. **Space Key**: Optional, filtert auf bestimmten Space

Buttons:
- **💾 Speichern**: Konfiguration speichern
- **🔍 Testen**: Verbindung prüfen

#### 🔑 Confluence API-Token erstellen

Confluence Cloud nutzt Atlassian API Tokens (kein OAuth nötig).

**Voraussetzungen:**
- Confluence Cloud (nicht Server / Data Center)
- Atlassian-Account mit Zugriff auf das Confluence-Space

**Schritte:**

1. Öffnen Sie die Token-Verwaltung:
   
   👉 https://id.atlassian.com/manage-profile/security/api-tokens

2. Klicken Sie auf **"Create API token"**

3. Vergeben Sie einen Namen (z.B. "RAG Wissensdatenbank")

4. Klicken Sie auf **"Create"**

5. **Kopieren Sie das Token sofort** (wird nur einmal angezeigt!)

6. Tragen Sie das Token in der App unter 🔷 Confluence ein

> ⚠️ **Wichtig**: Das Token hat dieselben Berechtigungen wie Ihr Account. Teilen Sie es nicht!

---

## 📑 Seitenleiste - Meine Quellen

Übersicht aller hochgeladenen Dokumente:
- **Anzahl Quellen** und **Chunks** als Metriken
- Pro Quelle: Name, Chunk-Anzahl, Löschen-Button
- **🗑️ Alle Quellen löschen**: Gesamte Wissensdatenbank leeren

---

## ⚙️ Seitenleiste - Einstellungen

### Antwortlänge
Slider mit drei Optionen:
- **kurz**: 2-3 Sätze, Kernaussagen
- **normal**: Ausgewogene Antwort (Standard)
- **ausführlich**: Detaillierte Erklärungen

### 🔗 Confluence durchsuchen
Checkbox erscheint wenn Confluence konfiguriert ist. Aktivieren um Confluence in die Suche einzubeziehen.

### 🧹 Chat leeren
Löscht den Chatverlauf und startet eine neue Konversation.

---

## 💬 Hauptbereich - Chat

Das Herzstück der Anwendung. Stellen Sie Fragen zu Ihren Dokumenten.

### Funktionsweise
1. Frage eingeben im Chat-Feld
2. System durchsucht alle Quellen (+ optional Confluence)
3. Relevante Dokumente werden reranked
4. LLM generiert Antwort basierend auf Kontext

### Features
- **Konversationsverlauf**: Vorherige Fragen und Antworten bleiben sichtbar
- **Quellenangabe**: Expandierbarer Bereich zeigt verwendete Quellen
- **Workflow-Log**: Details zur Query-Verarbeitung

### Beispiel-Fragen
- "Was sind die wichtigsten Punkte aus dem Jahresbericht?"
- "Vergleiche die Strategien aus Dokument A und B"
- "Fasse alle Informationen zu Thema X zusammen"

---

## 🔍 Hauptbereich - Suche

Direkte Dokumentensuche ohne LLM-Generierung.

- Suchbegriff eingeben
- Anzahl Ergebnisse wählen (1-20)
- Ergebnisse zeigen relevante Textpassagen mit Quellenangabe

Nützlich für:
- Schnelles Finden spezifischer Informationen
- Überprüfen welche Dokumente ein Thema behandeln
- Debuggen der Retrieval-Qualität

---

## 📝 Hauptbereich - Dokument-Generator

Automatische Erstellung von Dokumenten basierend auf Ihrer Wissensdatenbank.

### Verwendung
1. **Dokumenttitel** eingeben
2. **Inhaltsverzeichnis** erstellen (ein Kapitel pro Zeile)
3. **📄 Dokument generieren** klicken
4. Warten bis alle Kapitel generiert sind
5. **📥 Als Markdown herunterladen**

### Beispiel-Inhaltsverzeichnis
```
1. Einleitung
2. Problemstellung
3. Lösungsansatz
4. Implementierung
5. Ergebnisse
6. Fazit
```

---

## 🔌 API-Dokumentation

Die API ist unter **http://localhost:8000/docs** dokumentiert (Swagger UI).

### Wichtige Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/health` | GET | Systemstatus prüfen |
| `/upload` | POST | Dokument hochladen |
| `/add-text` | POST | Text hinzufügen |
| `/add-url` | POST | URL hinzufügen |
| `/query-langgraph` | POST | Frage stellen (empfohlen) |
| `/search` | GET | Dokumente suchen |
| `/sources` | GET | Alle Quellen auflisten |
| `/sources/{name}` | DELETE | Quelle löschen |
| `/confluence/status` | GET | Confluence-Status |
| `/confluence/search` | GET | Confluence durchsuchen |

### Beispiel: Frage stellen
```python
import requests

response = requests.post(
    "http://localhost:8000/query-langgraph",
    json={
        "question": "Was sind die Hauptthemen?",
        "response_length": "normal",
        "include_confluence": False
    }
)

data = response.json()
print(data["answer"])
print(data["sources"])
```

---

## 🏗️ Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────┐
│                  FastAPI Server (main.py)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  LangGraph    │ │   Retriever   │ │  Confluence   │
│  Workflow     │ │  (Hybrid+RRF) │ │    Loader     │
└───────┬───────┘ └───────┬───────┘ └───────────────┘
        │                 │
        ▼                 ▼
┌───────────────┐ ┌───────────────┐
│  OpenAI LLM   │ │   ChromaDB    │
│  (Generation) │ │  (Vectors)    │
└───────────────┘ └───────────────┘
```

### Komponenten

| Komponente | Datei | Funktion |
|------------|-------|----------|
| Document Processor | `document_processor.py` | Dokumente parsen, chunken |
| Embedding Engine | `embedding_engine.py` | Text → Vektoren |
| Vector Database | `vector_database.py` | ChromaDB Wrapper |
| Retriever | `retriever.py` | Hybrid-Suche, Reranking |
| Graph Workflow | `graph_workflow.py` | LangGraph Query-Pipeline |
| Confluence Loader | `confluence_loader.py` | Confluence-Integration |
| RAG Pipeline | `rag_pipeline.py` | Orchestrierung |

---

## 🔧 Fehlerbehebung

### API startet nicht
```bash
# Port bereits belegt?
netstat -ano | findstr :8000

# Prozess beenden (Windows)
taskkill /F /PID <PID>
```

### Streamlit Verbindungsfehler
Stellen Sie sicher, dass die API läuft bevor Streamlit gestartet wird.

### OpenAI Fehler
- API-Key korrekt in `.env`?
- Ausreichend Credits auf dem OpenAI Account?

### Confluence Verbindung fehlgeschlagen
- URL korrekt? (mit `/wiki` am Ende für Cloud)
- API Token aktuell?
- Benutzername = E-Mail Adresse

### Langsame Antworten
- Erste Anfrage lädt Cross-Encoder Modell (~30s)
- Folgende Anfragen sind deutlich schneller

---

## 📄 Lizenz

MIT License

---

## 🤝 Beitragen

Pull Requests sind willkommen! Bitte erstellen Sie zuerst ein Issue für größere Änderungen.
