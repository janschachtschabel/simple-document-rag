import streamlit as st
import requests
import json
from typing import Dict, Any, List
from datetime import datetime
import urllib.parse

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Wissensdatenbank",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better UX (compatible with light and dark mode)
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 0.95rem;
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== API FUNCTIONS ====================

def upload_document(uploaded_file) -> Dict[str, Any]:
    """Upload a document to the API."""
    try:
        files = {"file": uploaded_file}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def add_text_document(text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Add text document to the API."""
    try:
        payload = {"text": text, "metadata": metadata or {}}
        response = requests.post(f"{API_BASE_URL}/add-text", json=payload)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def add_url_document(url: str) -> Dict[str, Any]:
    """Add URL document to the API."""
    try:
        payload = {"url": url}
        response = requests.post(f"{API_BASE_URL}/add-url", json=payload)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def query_system(question: str, response_length: str = "normal", include_confluence: bool = False) -> Dict[str, Any]:
    """Query the RAG system using LangGraph workflow."""
    try:
        payload = {
            "question": question,
            "max_attempts": 2,
            "response_length": response_length,
            "include_confluence": include_confluence
        }
        response = requests.post(f"{API_BASE_URL}/query-langgraph", json=payload, timeout=120)
        result = response.json()
        
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "query_analysis": result.get("metadata", {}).get("query_analysis", {}),
            "documents_retrieved": result.get("metadata", {}).get("grading_summary", {}).get("relevant", 0) + 
                                   result.get("metadata", {}).get("grading_summary", {}).get("ambiguous", 0),
            "workflow_log": result.get("metadata", {}).get("workflow_log", [])
        }
    except requests.exceptions.Timeout:
        return {"error": "Zeitüberschreitung. Bitte erneut versuchen."}
    except Exception as e:
        return {"error": str(e)}

def search_documents(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Search documents."""
    try:
        params = {"q": query, "top_k": top_k}
        response = requests.get(f"{API_BASE_URL}/search", params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def list_sources() -> Dict[str, Any]:
    """List all sources in the database."""
    try:
        response = requests.get(f"{API_BASE_URL}/sources", timeout=10)
        return response.json()
    except Exception as e:
        return {"sources": [], "total_sources": 0, "total_chunks": 0, "error": str(e)}

def delete_source(source_name: str) -> Dict[str, Any]:
    """Delete a source and all its chunks."""
    try:
        encoded_name = urllib.parse.quote(source_name, safe='')
        response = requests.delete(f"{API_BASE_URL}/sources/{encoded_name}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def delete_all_sources() -> Dict[str, Any]:
    """Delete all sources."""
    try:
        response = requests.delete(f"{API_BASE_URL}/sources")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_confluence_status() -> Dict[str, Any]:
    """Get Confluence integration status."""
    try:
        response = requests.get(f"{API_BASE_URL}/confluence/status", timeout=5)
        return response.json()
    except Exception as e:
        return {"configured": False, "available": False, "error": str(e)}

def save_confluence_config(url: str, username: str, api_key: str, space_key: str = "") -> Dict[str, Any]:
    """Save Confluence configuration."""
    try:
        payload = {
            "url": url,
            "username": username,
            "api_key": api_key,
            "space_key": space_key
        }
        response = requests.post(f"{API_BASE_URL}/confluence/config", json=payload, timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        return response.json()
    except Exception as e:
        return {"total_documents": 0, "error": str(e)}

def generate_document(toc_entries: List[str], document_title: str) -> Dict[str, Any]:
    """Generate a document from TOC entries."""
    try:
        payload = {"toc_entries": toc_entries, "document_title": document_title}
        timeout = max(600, len(toc_entries) * 60)
        response = requests.post(f"{API_BASE_URL}/generate-document", json=payload, timeout=timeout)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": f"Zeitüberschreitung nach {timeout}s. Weniger Abschnitte verwenden."}
    except Exception as e:
        return {"error": str(e)}

# ==================== HELPER FUNCTIONS ====================

def get_file_icon(filename: str) -> str:
    """Get an appropriate icon for a file type."""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    icons = {
        'pdf': '📕', 'doc': '📘', 'docx': '📘', 'txt': '📄', 'md': '📝',
        'xlsx': '📊', 'xls': '📊', 'csv': '📊',
        'pptx': '📙', 'ppt': '📙',
        'html': '🌐', 'htm': '🌐',
        'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🖼️',
        'epub': '📖', 'msg': '📧', 'eml': '📧',
        'json': '📋', 'xml': '📋', 'zip': '📦'
    }
    return icons.get(ext, '📄')

def truncate_filename(filename: str, max_length: int = 25) -> str:
    """Truncate long filenames."""
    if len(filename) <= max_length:
        return filename
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    available = max_length - len(ext) - 4  # 4 for "..." and "."
    return f"{name[:available]}...{'.'+ext if ext else ''}"

# ==================== MAIN APPLICATION ====================

def main():
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_sources' not in st.session_state:
        st.session_state.current_sources = []
    if 'response_length' not in st.session_state:
        st.session_state.response_length = "normal"
    if 'delete_confirm' not in st.session_state:
        st.session_state.delete_confirm = None
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        
        # === DOCUMENT UPLOAD SECTION ===
        st.markdown("### ➕ Quellen hinzufügen")
        
        upload_tab, text_tab, url_tab, confluence_tab = st.tabs(["📁 Datei", "📝 Text", "🔗 URL", "🔷 Confluence"])
        
        with upload_tab:
            uploaded_files = st.file_uploader(
                "Dateien hierher ziehen",
                type=['txt', 'md', 'pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls',
                      'html', 'htm', 'epub', 'msg', 'eml', 'jpg', 'jpeg', 'png', 'gif',
                      'csv', 'json', 'xml', 'zip'],
                accept_multiple_files=True,
                help="PDF, Word, Excel, PowerPoint, Bilder, E-Mails und mehr",
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                if st.button("📤 Hochladen", use_container_width=True, type="primary"):
                    progress = st.progress(0)
                    success, errors = 0, 0
                    
                    for i, file in enumerate(uploaded_files):
                        progress.progress((i + 1) / len(uploaded_files))
                        result = upload_document(file)
                        if result.get("success"):
                            success += 1
                        else:
                            errors += 1
                            st.error(f"❌ {file.name}")
                    
                    progress.empty()
                    if success > 0:
                        st.success(f"✅ {success} Datei(en) hinzugefügt")
                        st.rerun()
        
        with text_tab:
            text_content = st.text_area("Text eingeben:", height=100, label_visibility="collapsed", 
                                        placeholder="Text hier einfügen...")
            if text_content and st.button("📝 Text hinzufügen", use_container_width=True):
                result = add_text_document(text_content)
                if result.get("success"):
                    st.success("✅ Text hinzugefügt")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error')}")
        
        with url_tab:
            url_input = st.text_input("URL eingeben:", label_visibility="collapsed",
                                      placeholder="https://...")
            if url_input and st.button("🔗 URL hinzufügen", use_container_width=True):
                with st.spinner("Lade Webseite..."):
                    result = add_url_document(url_input)
                    if result.get("success"):
                        st.success("✅ Webseite hinzugefügt")
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('error')}")
        
        with confluence_tab:
            st.caption("Verbinde mit Atlassian Confluence")
            
            # Load existing config from session or defaults
            conf_url = st.text_input(
                "Confluence URL",
                value=st.session_state.get("confluence_url", ""),
                placeholder="https://domain.atlassian.net/wiki",
                key="conf_url_input"
            )
            conf_user = st.text_input(
                "Benutzername/E-Mail",
                value=st.session_state.get("confluence_user", ""),
                placeholder="user@example.com",
                key="conf_user_input"
            )
            conf_token = st.text_input(
                "API Token",
                value=st.session_state.get("confluence_token", ""),
                type="password",
                placeholder="API Token aus Atlassian",
                key="conf_token_input"
            )
            conf_space = st.text_input(
                "Space Key (optional)",
                value=st.session_state.get("confluence_space", ""),
                placeholder="z.B. WIKI, DOCS",
                key="conf_space_input"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Speichern", use_container_width=True, type="primary"):
                    # Save to session state
                    st.session_state.confluence_url = conf_url
                    st.session_state.confluence_user = conf_user
                    st.session_state.confluence_token = conf_token
                    st.session_state.confluence_space = conf_space
                    
                    # Also save via API
                    result = save_confluence_config(conf_url, conf_user, conf_token, conf_space)
                    if result.get("success"):
                        st.success("✅ Gespeichert")
                    else:
                        st.error(f"❌ {result.get('error', 'Fehler')}")
            
            with col2:
                if st.button("🔍 Testen", use_container_width=True):
                    if conf_url and conf_user and conf_token:
                        status = get_confluence_status()
                        if status.get("configured"):
                            st.success("✅ Verbindung OK")
                        else:
                            st.error("❌ Verbindung fehlgeschlagen")
                    else:
                        st.warning("Bitte alle Felder ausfüllen")
        
        st.divider()
        
        # === SOURCES LIST ===
        st.markdown("### 📑 Meine Quellen")
        
        sources_data = list_sources()
        sources = sources_data.get("sources", [])
        total_chunks = sources_data.get("total_chunks", 0)
        
        if sources:
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quellen", len(sources))
            with col2:
                st.metric("Chunks", total_chunks)
            
            st.markdown("---")
            
            # Source list with delete buttons
            for idx, source in enumerate(sources):
                source_name = source.get("source_name", "Unbekannt")
                chunk_count = source.get("chunk_count", 0)
                original_name = source.get("metadata", {}).get("original_filename", source_name)
                file_type = source.get("metadata", {}).get("file_type", "")
                
                display_name = truncate_filename(original_name)
                icon = get_file_icon(original_name)
                
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.markdown(f"{icon} **{display_name}**")
                    st.caption(f"{chunk_count} Chunks")
                
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"del_{idx}", help=f"'{original_name}' löschen"):
                        st.session_state.delete_confirm = source_name
                
                # Confirm deletion dialog
                if st.session_state.delete_confirm == source_name:
                    st.warning(f"'{display_name}' wirklich löschen?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Ja", key=f"confirm_{idx}", use_container_width=True):
                            result = delete_source(source_name)
                            if result.get("success"):
                                st.session_state.delete_confirm = None
                                st.rerun()
                            else:
                                st.error(result.get("error"))
                    with c2:
                        if st.button("❌ Nein", key=f"cancel_{idx}", use_container_width=True):
                            st.session_state.delete_confirm = None
                            st.rerun()
                
                st.markdown("---")
            
            # Delete all button
            if st.button("🗑️ Alle Quellen löschen", use_container_width=True, type="secondary"):
                st.session_state.delete_confirm = "__ALL__"
            
            if st.session_state.delete_confirm == "__ALL__":
                st.error("⚠️ ALLE Quellen wirklich löschen?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Ja, alle", key="confirm_all", use_container_width=True):
                        delete_all_sources()
                        st.session_state.delete_confirm = None
                        st.rerun()
                with c2:
                    if st.button("❌ Abbrechen", key="cancel_all", use_container_width=True):
                        st.session_state.delete_confirm = None
                        st.rerun()
        else:
            st.info("📭 Noch keine Quellen vorhanden.\n\nLade Dokumente hoch, um zu beginnen.")
        
        st.divider()
        
        # === SETTINGS ===
        st.markdown("### ⚙️ Einstellungen")
        
        response_length = st.select_slider(
            "Antwortlänge",
            options=["kurz", "normal", "ausführlich"],
            value=st.session_state.response_length,
            help="Kurz: 2-3 Sätze | Normal: Standard | Ausführlich: Detailliert"
        )
        st.session_state.response_length = response_length
        
        # Confluence option
        confluence_status = get_confluence_status()
        if confluence_status.get("configured"):
            include_confluence = st.checkbox(
                "🔗 Confluence durchsuchen",
                value=st.session_state.get("include_confluence", False),
                help=f"Confluence: {confluence_status.get('url', 'N/A')}"
            )
            st.session_state.include_confluence = include_confluence
        else:
            st.caption("🔗 Confluence nicht konfiguriert")
            st.session_state.include_confluence = False
        
        # Clear chat button
        if st.button("🧹 Chat leeren", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.current_sources = []
            st.rerun()
    
    # ==================== MAIN CONTENT AREA ====================
    
    # Header
    st.markdown('<p class="main-header">📚 RAG Wissensdatenbank</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Stelle Fragen zu deinen Dokumenten – die KI findet die Antworten.</p>', unsafe_allow_html=True)
    
    # Main tabs
    chat_tab, search_tab, generator_tab = st.tabs(["💬 Chat", "🔍 Suche", "📝 Dokument-Generator"])
    
    # ==================== CHAT TAB ====================
    with chat_tab:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display conversation history
            for i, message in enumerate(st.session_state.conversation_history):
                with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])
                    
                    # Show sources for assistant messages
                    if message["role"] == "assistant" and i // 2 < len(st.session_state.current_sources):
                        sources = st.session_state.current_sources[i // 2]
                        if sources:
                            with st.expander(f"📚 {len(sources)} Quellen verwendet"):
                                for j, src in enumerate(sources):
                                    st.markdown(f"**{j+1}.** {src.get('text', '')[:200]}...")
                                    st.caption(f"Relevanz: {src.get('relevance', 'N/A')}")
        
        # Chat input
        if prompt := st.chat_input("💬 Stelle eine Frage zu deinen Dokumenten..."):
            # Add user message
            st.session_state.conversation_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant", avatar="🤖"):
                search_msg = "🔍 Durchsuche Wissensdatenbank"
                if st.session_state.get("include_confluence"):
                    search_msg += " + Confluence"
                search_msg += "..."
                with st.spinner(search_msg):
                    response = query_system(
                        prompt, 
                        st.session_state.response_length,
                        st.session_state.get("include_confluence", False)
                    )
                    
                    if "answer" in response:
                        st.markdown(response["answer"])
                        
                        # Store sources
                        sources = response.get("sources", [])
                        st.session_state.current_sources.append(sources)
                        
                        # Add to history
                        st.session_state.conversation_history.append({
                            "role": "assistant", 
                            "content": response["answer"]
                        })
                        
                        # Show metadata
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if sources:
                                with st.expander(f"📚 {len(sources)} Quellen verwendet"):
                                    for j, src in enumerate(sources):
                                        st.markdown(f"**{j+1}.** {src.get('text', '')[:200]}...")
                                        st.caption(f"Relevanz: {src.get('relevance', 'N/A')}")
                        with col2:
                            docs = response.get('documents_retrieved', 0)
                            st.caption(f"📊 {docs} Dokumente analysiert")
                    
                    elif "error" in response:
                        st.error(f"❌ {response['error']}")
    
    # ==================== SEARCH TAB ====================
    with search_tab:
        st.markdown("### 🔍 Dokumente durchsuchen")
        st.caption("Finde relevante Passagen in deiner Wissensdatenbank")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input("Suchbegriff", label_visibility="collapsed", 
                                         placeholder="Wonach suchst du?")
        with col2:
            search_limit = st.selectbox("Anzahl", [5, 10, 20, 50], index=1, label_visibility="collapsed")
        
        if search_query:
            with st.spinner("Suche..."):
                results = search_documents(search_query, search_limit)
                
                if "results" in results and results["results"]:
                    st.success(f"✅ {len(results['results'])} Treffer gefunden")
                    
                    for i, result in enumerate(results["results"]):
                        with st.expander(f"**{i+1}.** {result.get('metadata', {}).get('filename', 'Unbekannt')}", expanded=i==0):
                            st.markdown(result.get("text", "")[:500] + "...")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                score = result.get("similarity_score", 0)
                                st.metric("Relevanz", f"{score:.1%}")
                            with col2:
                                chunk = result.get("metadata", {}).get("chunk_index", "?")
                                st.metric("Chunk", chunk)
                            with col3:
                                source = result.get("metadata", {}).get("source_type", "?")
                                st.metric("Typ", source)
                else:
                    st.warning("🔍 Keine Treffer gefunden. Versuche andere Suchbegriffe.")
    
    # ==================== DOCUMENT GENERATOR TAB ====================
    with generator_tab:
        st.markdown("### 📝 Dokument-Generator")
        st.caption("Erstelle ein strukturiertes Dokument basierend auf deiner Wissensdatenbank")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            doc_title = st.text_input("📌 Dokumenttitel", value="Neues Dokument",
                                      placeholder="Titel des zu generierenden Dokuments")
            
            toc_text = st.text_area(
                "📋 Inhaltsverzeichnis",
                height=200,
                placeholder="Ein Eintrag pro Zeile:\n\n1. Einleitung\n2. Hauptteil\n3. Zusammenfassung",
                help="Jede Zeile wird ein Kapitel. Die KI generiert Inhalte für jedes Kapitel."
            )
        
        with col2:
            st.markdown("#### 💡 Tipps")
            st.info("""
            - **Klare Titel** verwenden
            - **3-10 Kapitel** empfohlen
            - **Spezifische Themen** für bessere Ergebnisse
            - Generierung dauert ~30-60s pro Kapitel
            """)
        
        if st.button("🚀 Dokument generieren", type="primary", use_container_width=True):
            if toc_text.strip():
                toc_entries = [e.strip() for e in toc_text.strip().split("\n") if e.strip()]
                
                if len(toc_entries) > 0:
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.info(f"📝 Generiere Dokument mit {len(toc_entries)} Kapiteln...")
                    
                    result = generate_document(toc_entries, doc_title)
                    progress.progress(100)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    elif "full_document" in result:
                        status.success(f"✅ Dokument '{doc_title}' erfolgreich generiert!")
                        
                        # Download button
                        st.download_button(
                            "📥 Als Markdown herunterladen",
                            data=result['full_document'],
                            file_name=f"{doc_title.replace(' ', '_')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                        
                        # Preview
                        with st.expander("📖 Vorschau", expanded=True):
                            st.markdown(result['full_document'])
                        
                        # Sources
                        with st.expander(f"📚 {result.get('total_sources', 0)} Quellen verwendet"):
                            for src in result.get('sources', [])[:20]:
                                st.caption(f"• {src.get('title', src.get('source', 'Unbekannt'))}")
                else:
                    st.warning("⚠️ Bitte mindestens ein Kapitel eingeben.")
            else:
                st.warning("⚠️ Bitte ein Inhaltsverzeichnis eingeben.")

if __name__ == "__main__":
    main()
