import os
import sys
import json
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit, 
    QScrollArea, QFileDialog, QTableWidget, QTableWidgetItem, 
    QHeaderView, QMessageBox, QSplitter, QFrame, QSizePolicy,
    QDialog, QFormLayout, QComboBox
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QColor, QIcon

# Import databases and model logic
from Veritabanı.vector_db import VectorDBManager
from Veritabanı.graph_db import GraphDBManager, Neo4jConnectionError
from Modeller.extractor import DocumentExtractor
from Modeller.retriever import HybridRetriever
from Modeller.synthesizer import AnswerSynthesizer
from Arayüz.graph_canvas import GraphCanvas

CONFIG_FILE = "config.json"

def get_default_config():
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "embed_model": "qwen3-embedding:4b",
        "extract_model": "qwen3:4b",
        "synth_model": "gpt-oss:120b-cloud",
        "num_ctx": 8192
    }

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return get_default_config()

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


# Simple Markdown-to-HTML parser for rich text display in PyQt6
def markdown_to_html(md_text: str) -> str:
    # Escape HTML tags
    html = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold
    html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html)
    # Italic
    html = re.sub(r'\*(.*?)\*', r'<i>\1</i>', html)
    # Bullet points
    lines = html.split("\n")
    in_list = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            content = line.strip()[2:]
            if not in_list:
                new_lines.append("<ul>")
                in_list = True
            new_lines.append(f"<li>{content}</li>")
        else:
            if in_list:
                new_lines.append("</ul>")
                in_list = False
            new_lines.append(line)
    if in_list:
        new_lines.append("</ul>")
    html = "<br>".join(new_lines)
    return html


# Worker thread for document ingestion to keep GUI responsive
class IngestionWorker(QThread):
    progress = pyqtSignal(str, int) # status, percentage
    finished = pyqtSignal(dict)     # result dict
    error = pyqtSignal(str)         # error message

    def __init__(self, extractor: DocumentExtractor, pdf_path: str, doc_metadata: dict = None):
        super().__init__()
        self.extractor = extractor
        self.pdf_path = pdf_path
        self.doc_metadata = doc_metadata

    def run(self):
        try:
            def update_progress(status, pct):
                self.progress.emit(status, pct)
            
            result = self.extractor.ingest_document(self.pdf_path, doc_metadata=self.doc_metadata, progress_callback=update_progress)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# Worker thread for RAG Query
class QueryWorker(QThread):
    finished = pyqtSignal(dict)     # result dict
    error = pyqtSignal(str)         # error message

    def __init__(self, retriever: HybridRetriever, synthesizer: AnswerSynthesizer, query: str, chat_history: list = None):
        super().__init__()
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.query = query
        self.chat_history = chat_history

    def run(self):
        try:
            # 1. Retrieve (with chat history to resolve pronouns)
            context_data = self.retriever.retrieve(self.query, chat_history=self.chat_history)
            
            # Hallucination Guard: Intercept empty context routing immediately
            if context_data.get("routing") == "empty":
                result = {
                    "query": self.query,
                    "thought": "Arama sonucunda veritabanında bu soruya dair hiçbir kavram, ilişki veya anlamsal olarak benzer metin parçası bulunamadı. Boş kaynak bağlamı nedeniyle LLM sentezi atlandı.",
                    "answer": "Verilen kaynaklarda bu bilgi bulunmamaktadır.",
                    "sources": [],
                    "source_paths": {},
                    "concepts": [],
                    "relationships": []
                }
                self.finished.emit(result)
                return

            # 2. Synthesize (with chat history for conversational awareness)
            response = self.synthesizer.synthesize(self.query, context_data["context_text"], chat_history=self.chat_history)
            
            # Query graph_db for source file paths
            source_paths = {}
            source_metadata = {}
            if hasattr(self.retriever, "gdb") and self.retriever.gdb:
                try:
                    with self.retriever.gdb.driver.session() as session:
                        res = session.run("MATCH (d:Document) RETURN d.id AS id, d.name AS name, d.file_path AS file_path")
                        for rec in res:
                            d_name = rec["name"]
                            d_id = rec["id"]
                            if d_name:
                                if rec["file_path"]:
                                    source_paths[d_name] = rec["file_path"]
                                meta = self.retriever.gdb.get_document_metadata(d_id)
                                if meta:
                                    source_metadata[d_name] = meta
                except Exception:
                    pass
            
            # Combine results
            result = {
                "query": self.query,
                "thought": response["thought"],
                "answer": response["answer"],
                "sources": context_data["sources"],
                "source_paths": source_paths,
                "source_metadata": source_metadata,
                "concepts": context_data.get("concepts_found", []),
                "relationships": context_data.get("relationships", [])
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# --- CUSTOM UI WIDGETS ---

class CollapsibleThoughtBox(QWidget):
    """An expandable box for thought process display."""
    def __init__(self, thought_text: str, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 5)
        self.layout.setSpacing(2)

        self.btn_toggle = QPushButton("▼ Düşünme Süreci")
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background-color: #2b2b30;
                color: #b0b0b8;
                border: 1px solid #3c3c43;
                border-radius: 4px;
                padding: 6px;
                text-align: left;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #35353c;
                color: #ffffff;
                border: 1px solid #8b5cf6;
            }
        """)
        self.btn_toggle.clicked.connect(self.toggle)
        self.layout.addWidget(self.btn_toggle)

        self.txt_content = QTextEdit()
        self.txt_content.setReadOnly(True)
        self.txt_content.setText(thought_text)
        self.txt_content.setMinimumHeight(100)
        self.txt_content.setMaximumHeight(250)
        self.txt_content.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1c;
                color: #8b949e;
                border: 1px solid #3c3c43;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.txt_content.hide()
        self.layout.addWidget(self.txt_content)
        self.is_collapsed = True

    def toggle(self):
        if self.is_collapsed:
            self.txt_content.show()
            self.btn_toggle.setText("▲ Düşünme Süreci")
            self.is_collapsed = False
        else:
            self.txt_content.hide()
            self.btn_toggle.setText("▼ Düşünme Süreci")
            self.is_collapsed = True


class DocumentMetadataDialog(QDialog):
    def __init__(self, filename: str, last_user: str = "System", last_project: str = "Default Project", last_company: str = "Default Company", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Doküman İlişkilendirme Detayları")
        self.setFixedWidth(400)
        
        # Dark style QSS
        self.setStyleSheet("""
            QDialog {
                background-color: #161619;
                border: 1px solid #2e2e33;
            }
            QLabel {
                color: #a1a1aa;
                font-size: 11px;
                font-weight: bold;
            }
            QLineEdit, QComboBox {
                background-color: #1c1c1f;
                border: 1px solid #2e2e33;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #06b6d4;
            }
            QPushButton {
                background-color: #1c1c1f;
                border: 1px solid #2e2e33;
                border-radius: 4px;
                padding: 8px 16px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27272a;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title header
        lbl_title = QLabel("<span style='font-size: 14px; font-weight: bold; color: #06b6d4;'>🧠 DOKÜMAN İLİŞKİLENDİRME</span>")
        layout.addWidget(lbl_title)
        
        # File info
        lbl_file = QLabel(f"Dosya: <span style='color: #ffffff;'>{filename}</span>")
        lbl_file.setWordWrap(True)
        layout.addWidget(lbl_file)
        
        # Form
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        
        self.txt_user = QLineEdit()
        self.txt_user.setPlaceholderText("Örn: Ahmet, Can")
        self.txt_user.setText(last_user)
        form_layout.addRow(QLabel("YÜKLEYEN KULLANICI:"), self.txt_user)
        
        self.txt_project = QLineEdit()
        self.txt_project.setPlaceholderText("Örn: DEUS AI, HERMES")
        self.txt_project.setText(last_project)
        form_layout.addRow(QLabel("İLİŞKİLİ PROJE:"), self.txt_project)
        
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Dokümantasyon", "Kod", "Tasarım", "Rapor"])
        form_layout.addRow(QLabel("İÇERİK TÜRÜ:"), self.combo_type)
        
        self.txt_company = QLineEdit()
        self.txt_company.setPlaceholderText("Örn: Aegis Technologies")
        self.txt_company.setText(last_company)
        form_layout.addRow(QLabel("ŞİRKET / ORGANİZASYON:"), self.txt_company)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_container = QHBoxLayout()
        button_container.addStretch()
        
        self.btn_cancel = QPushButton("İptal")
        self.btn_cancel.clicked.connect(self.reject)
        button_container.addWidget(self.btn_cancel)
        
        self.btn_submit = QPushButton("Analiz Et ve İndeksle")
        self.btn_submit.clicked.connect(self.accept)
        button_container.addWidget(self.btn_submit)
        
        layout.addLayout(button_container)
        
    def get_metadata(self) -> dict:
        return {
            "user": self.txt_user.text().strip() or "System",
            "project": self.txt_project.text().strip() or "Default Project",
            "doc_type": self.combo_type.currentText(),
            "company": self.txt_company.text().strip() or "Default Company"
        }


class MessageWidget(QWidget):
    """Custom chat bubble widget with robust resizing and elegant styling."""
    def __init__(self, text: str, is_user: bool, thought: str = None, sources: list = None, source_paths: dict = None, source_metadata: dict = None, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(0)
        
        # Outer bubble frame
        self.frame = QFrame()
        self.frame.setObjectName("MessageFrame")
        self.frame.setMaximumWidth(750)
        self.frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(14, 12, 14, 12)
        frame_layout.setSpacing(8)
        
        # User vs AI styling (Premium dark design)
        if is_user:
            self.frame.setStyleSheet("""
                QFrame#MessageFrame {
                    background-color: #2e1065;
                    border-radius: 12px;
                    border: 1px solid #5b21b6;
                }
            """)
        else:
            self.frame.setStyleSheet("""
                QFrame#MessageFrame {
                    background-color: #18181b;
                    border-radius: 12px;
                    border: 1px solid #27272a;
                }
            """)

        # Header / Author label (rich text styled)
        author_color = '#a78bfa' if is_user else '#06b6d4'
        author_name = 'Kullanıcı' if is_user else 'HERMES AI'
        lbl_author = QLabel(f"<span style='font-weight: bold; color: {author_color}; font-size: 10px; letter-spacing: 0.5px;'>{author_name.upper()}</span>")
        frame_layout.addWidget(lbl_author)

        # If thought exists, add collapsible box
        if thought:
            self.thought_box = CollapsibleThoughtBox(thought)
            frame_layout.addWidget(self.thought_box)

        # Text display (rich text styled)
        self.lbl_text = QLabel()
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_text.setText(f"<div style='color: #f4f4f5; font-size: 13px; line-height: 1.4;'>{markdown_to_html(text) if not is_user else text}</div>")
        self.lbl_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        frame_layout.addWidget(self.lbl_text)

        # If sources exist, add source section with automatic metadata tags
        if sources:
            source_blocks = []
            for src in sources:
                # Extract filename matching ending in .pdf
                match = re.search(r'^(.*?\.pdf)', src)
                if match:
                    base_name = match.group(1).strip()
                else:
                    base_name = src.strip()
                
                link_html = src
                if source_paths and base_name in source_paths:
                    path = source_paths[base_name]
                    link_html = f'<a href="{path}" style="color: #06b6d4; text-decoration: underline; font-weight: bold;">📄 {src}</a>'
                else:
                    link_html = f'<span style="color: #a1a1aa;">📄 {src}</span>'
                    
                meta_html = ""
                if source_metadata and base_name in source_metadata:
                    m = source_metadata[base_name]
                    meta_html = (
                        f'<div style="color: #71717a; font-size: 10px; margin-left: 15px; margin-top: 2px; margin-bottom: 6px;">'
                        f'└ 👤 <b>{m["user_name"]}</b> | 📅 <i>{m["upload_time"]}</i> | 📁 <b>{m["project_name"]}</b> | 🏢 <i>{m["company_name"]}</i>'
                        f'</div>'
                    )
                
                source_blocks.append(f'<div>{link_html}</div>{meta_html}')

            lbl_sources = QLabel(
                f"<div style='margin-top: 8px; border-top: 1px solid #27272a; padding-top: 8px;'>"
                f"<span style='color: #8b5cf6; font-size: 11px; font-weight: bold;'>🔍 Referans Kaynaklar:</span>"
                f"<div style='margin-top: 6px;'>{''.join(source_blocks)}</div>"
                f"</div>"
            )
            lbl_sources.setWordWrap(True)
            lbl_sources.setOpenExternalLinks(False)
            lbl_sources.linkActivated.connect(self.open_source_file)
            lbl_sources.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            frame_layout.addWidget(lbl_sources)

        # Bubble Layout with stretch to enforce proper width wrapping
        bubble_layout = QHBoxLayout()
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        bubble_layout.setSpacing(0)
        
        if is_user:
            bubble_layout.addStretch(1)
            bubble_layout.addWidget(self.frame, 4)
        else:
            bubble_layout.addWidget(self.frame, 4)
            bubble_layout.addStretch(1)
            
        layout.addLayout(bubble_layout)

    def open_source_file(self, file_path: str):
        """Open the clicked source file using system defaults."""
        if file_path and os.path.exists(file_path):
            try:
                os.startfile(file_path)
            except Exception as e:
                print(f"Error opening source file: {e}")


# --- MAIN APPLICATION WINDOW ---

class HermesMainWindow(QMainWindow):
    def __init__(self):
        print("[INIT] super().__init__() calling...")
        super().__init__()
        print("[INIT] setWindowTitle calling...")
        self.setWindowTitle("HERMES - Hybrid Graph-Vector RAG")
        print("[INIT] resize calling...")
        self.resize(1100, 750)
        
        # Load config
        print("[INIT] load_config calling...")
        self.config = load_config()

        # Database managers
        self.vdb = None
        self.gdb = None
        self.extractor = None
        self.retriever = None
        self.synthesizer = None
        
        # Initialize connection
        self.is_connected = False
        self.connection_error_msg = ""
        print("[INIT] setup_databases calling...")
        self.setup_databases()
        
        # Chat history memory
        self.chat_history = []

        # Main widget setup
        print("[INIT] central_widget creating...")
        self.central_widget = QWidget()
        print("[INIT] setCentralWidget calling...")
        self.setCentralWidget(self.central_widget)
        print("[INIT] main_layout creating...")
        self.main_layout = QHBoxLayout(self.central_widget)
        print("[INIT] main_layout created. Setting margins...")
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        print("[INIT] Margins set. Setting spacing...")
        self.main_layout.setSpacing(0)
        print("[INIT] Spacing set. Setting stylesheet...")

        # App QSS styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121214;
            }
            QTabWidget::panel {
                border-top: 1px solid #2e2e33;
                background-color: #121214;
            }
            QTabBar::tab {
                background-color: #1c1c1f;
                border: 1px solid #2e2e33;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 4px;
                color: #a1a1aa;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #242429;
                color: #06b6d4;
                border: 1px solid #3c3c43;
                border-bottom: 2px solid #06b6d4;
            }
            QLineEdit, QTextEdit {
                background-color: #18181b;
                border: 1px solid #2e2e33;
                border-radius: 6px;
                padding: 6px;
                color: #ffffff;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #06b6d4;
            }
            QPushButton {
                background-color: #1c1c1f;
                border: 1px solid #2e2e33;
                border-radius: 6px;
                padding: 8px 14px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27272a;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #18181b;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #2e2e33;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3f3f46;
            }
        """)
        print("[INIT] Stylesheet set successfully!")

        print("[INIT] build_ui calling...")
        self.build_ui()
        print("[INIT] build_ui completed. Updating stats...")
        self.update_stats()
        print("[INIT] update_stats completed. Updating document list...")
        self.update_document_list()
        print("[INIT] update_document_list completed.")

        # If there's connection error on startup, show alert and switch to Settings
        if not self.is_connected:
            self.lbl_connection_banner.show()
            self.lbl_connection_banner.setText(f"HATA: Neo4j'ye bağlanılamadı! Bağlantı detaylarını kontrol edin.")
            QMessageBox.critical(
                self, "Veritabanı Hatası", 
                f"Neo4j veritabanına bağlanılamadı!\n\n{self.connection_error_msg}\n\n"
                "Lütfen Neo4j Desktop / Server uygulamasını çalıştırın ve 'Ayarlar' sekmesinden bilgileri doğrulayın."
            )
            self.tabs.setCurrentIndex(2) # Switch to Settings tab

    def setup_databases(self):
        """Initializes database wrappers and components using config values."""
        try:
            # 1. Initialize Vector Database
            self.vdb = VectorDBManager()
            
            # 2. Try Connecting to Graph Database (Neo4j)
            self.gdb = GraphDBManager(
                uri=self.config["neo4j_uri"],
                user=self.config["neo4j_user"],
                password=self.config["neo4j_password"]
            )
            
            # 3. Ingestion & Retrieval services
            self.extractor = DocumentExtractor(self.vdb, self.gdb, model_name=self.config["extract_model"])
            self.retriever = HybridRetriever(self.vdb, self.gdb, model_name=self.config["extract_model"])
            self.synthesizer = AnswerSynthesizer(model_name=self.config["synth_model"], num_ctx=self.config.get("num_ctx", 8192))
            
            self.is_connected = True
            self.connection_error_msg = ""
        except Neo4jConnectionError as e:
            self.is_connected = False
            self.connection_error_msg = str(e)
        except Exception as e:
            self.is_connected = False
            self.connection_error_msg = f"Genel Veritabanı Hatası: {str(e)}"

    def build_ui(self):
        print("[UI] build_ui starting...")
        # 1. SIDEBAR (Left Panel)
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("QFrame { background-color: #161619; border-right: 1px solid #242429; }")
        print("[UI] Sidebar styled.")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 15)
        sidebar_layout.setSpacing(15)

        # Header Logo (Rich text HTML styled)
        lbl_logo = QLabel("<span style='font-size: 20px; font-weight: bold; color: #06b6d4;'>🧠 HERMES RAG</span>")
        sidebar_layout.addWidget(lbl_logo)

        # PDF Upload Section (Rich text HTML styled)
        lbl_section_upload = QLabel("<span style='font-size: 10px; font-weight: bold; color: #71717a;'>DOKÜMAN YÜKLE</span>")
        sidebar_layout.addWidget(lbl_section_upload)

        self.btn_select_file = QPushButton("📁 PDF Dosyası Seç")
        self.btn_select_file.setStyleSheet("""
            QPushButton {
                background-color: #312e81;
                border: 1px solid #4338ca;
                padding: 10px;
                border-radius: 6px;
                color: #c7d2fe;
            }
            QPushButton:hover {
                background-color: #3730a3;
                border: 1px solid #4f46e5;
            }
        """)
        self.btn_select_file.clicked.connect(self.select_and_upload_pdf)
        sidebar_layout.addWidget(self.btn_select_file)

        # Upload Progress Bar placeholder
        self.lbl_progress = QLabel("")
        self.lbl_progress.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_progress)

        # Ingested Files List Section (Rich text HTML styled)
        lbl_section_files = QLabel("<span style='font-size: 10px; font-weight: bold; color: #71717a;'>DOKÜMAN ENVANTERİ</span>")
        sidebar_layout.addWidget(lbl_section_files)

        self.tbl_files = QTableWidget()
        self.tbl_files.setColumnCount(2)
        self.tbl_files.setHorizontalHeaderLabels(["Doküman Adı", "İşlem"])
        self.tbl_files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tbl_files.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_files.verticalHeader().hide()
        self.tbl_files.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1d;
                border: 1px solid #242429;
                border-radius: 6px;
                gridline-color: #242429;
            }
            QHeaderView::section {
                background-color: #161619;
                color: #71717a;
                padding: 4px;
                font-weight: bold;
                border: none;
                font-size: 10px;
            }
        """)
        self.tbl_files.cellDoubleClicked.connect(self.on_file_table_double_clicked)
        sidebar_layout.addWidget(self.tbl_files)

        # Stats Section (Rich text HTML styled)
        lbl_section_stats = QLabel("<span style='font-size: 10px; font-weight: bold; color: #71717a;'>SİSTEM İSTATİSTİKLERİ</span>")
        sidebar_layout.addWidget(lbl_section_stats)

        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("QFrame { background-color: #1a1a1d; border: 1px solid #242429; border-radius: 6px; }")
        
        stats_layout = QVBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        self.lbl_stat_docs = QLabel("Toplam Doküman: 0")
        self.lbl_stat_chunks = QLabel("Toplam Parça: 0")
        self.lbl_stat_nodes = QLabel("Kavram Düğümü: 0")
        self.lbl_stat_edges = QLabel("İlişki Kenarı: 0")
        stats_layout.addWidget(self.lbl_stat_docs)
        stats_layout.addWidget(self.lbl_stat_chunks)
        stats_layout.addWidget(self.lbl_stat_nodes)
        stats_layout.addWidget(self.lbl_stat_edges)
        sidebar_layout.addWidget(self.stats_frame)

        # Reset Databases button
        self.btn_reset = QPushButton("🗑️ Tüm Verileri Sıfırla")
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #7f1d1d;
                border: 1px solid #991b1b;
                color: #fca5a5;
                font-size: 11px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #991b1b;
                border: 1px solid #b91c1c;
            }
        """)
        self.btn_reset.clicked.connect(self.reset_system)
        sidebar_layout.addWidget(self.btn_reset)

        self.main_layout.addWidget(sidebar)

        # 2. MAIN WORKSPACE (Right Panel)
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        # Error Banner if Neo4j is offline
        self.lbl_connection_banner = QLabel("<div style='background-color: #7f1d1d; color: #fca5a5; font-weight: bold; font-size: 12px; padding: 8px; border-bottom: 1px solid #991b1b; text-align: center;'>UYARI: Veritabanı bağlantısı yok. Bazı özellikler devre dışı.</div>")
        self.lbl_connection_banner.hide()
        workspace_layout.addWidget(self.lbl_connection_banner)

        # Tabs Layout
        self.tabs = QTabWidget()
        workspace_layout.addWidget(self.tabs)
        self.main_layout.addWidget(workspace)

        # TAB 1: Soru-Cevap (Chat)
        tab_chat = QWidget()
        chat_layout = QVBoxLayout(tab_chat)
        chat_layout.setContentsMargins(15, 15, 15, 15)
        chat_layout.setSpacing(10)

        # Header for Chat Tab
        header_layout = QHBoxLayout()
        lbl_chat_title = QLabel("<span style='font-size: 11px; font-weight: bold; color: #a78bfa; letter-spacing: 0.5px;'>💬 AKTİF SOHBET OTURUMU</span>")
        header_layout.addWidget(lbl_chat_title)
        header_layout.addStretch()
        
        self.btn_clear_chat = QPushButton("🧹 Sohbeti Temizle")
        self.btn_clear_chat.setToolTip("Sohbet geçmişini ve arayüzdeki balonları sıfırlar")
        self.btn_clear_chat.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1d;
                border: 1px solid #242429;
                color: #a1a1aa;
                font-size: 11px;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                color: #f4f4f5;
            }
        """)
        self.btn_clear_chat.clicked.connect(self.clear_chat_history)
        header_layout.addWidget(self.btn_clear_chat)
        chat_layout.addLayout(header_layout)

        # Scroll area for messages
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_widget = QWidget()
        self.chat_widget.setStyleSheet("background-color: transparent;")
        self.chat_box_layout = QVBoxLayout(self.chat_widget)
        self.chat_box_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_box_layout.setSpacing(10)
        self.chat_box_layout.addStretch() # Push bubbles to top
        self.chat_scroll.setWidget(self.chat_widget)
        chat_layout.addWidget(self.chat_scroll)

        # Input block
        input_container = QHBoxLayout()
        input_container.setSpacing(10)
        
        self.txt_query = QLineEdit()
        self.txt_query.setPlaceholderText("HERMES'e dokümanlarla veya kavramlarla ilgili soru sorun (örn. 'Sistemde ne var?' veya 'DEUS AI nedir?')")
        self.txt_query.setStyleSheet("font-size: 13px; padding: 10px;")
        self.txt_query.returnPressed.connect(self.submit_query)
        input_container.addWidget(self.txt_query)

        self.btn_send = QPushButton("Gönder")
        self.btn_send.setStyleSheet("""
            QPushButton {
                background-color: #0891b2;
                border: 1px solid #0e7490;
                padding: 10px 20px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0e7490;
                border: 1px solid #06b6d4;
            }
        """)
        self.btn_send.clicked.connect(self.submit_query)
        input_container.addWidget(self.btn_send)

        chat_layout.addLayout(input_container)
        self.tabs.addTab(tab_chat, "💬 Soru - Cevap (RAG)")

        # TAB 2: Kavram Haritası (Mind Map / Visualizer)
        tab_graph = QWidget()
        graph_layout = QHBoxLayout(tab_graph)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Interactive Canvas
        print("[UI] Instantiating GraphCanvas...")
        self.canvas = GraphCanvas()
        print("[UI] GraphCanvas instantiated. Connecting signals...")
        self.canvas.nodeSelected.connect(self.inspect_concept)
        print("[UI] GraphCanvas signals connected. Adding to splitter...")
        splitter.addWidget(self.canvas)

        # Side details pane
        print("[UI] Creating inspector_panel...")
        self.inspector_panel = QFrame()
        self.inspector_panel.setFixedWidth(280)
        print("[UI] Styling inspector_panel...")
        self.inspector_panel.setStyleSheet("QFrame { background-color: #161619; border-left: 1px solid #242429; }")
        print("[UI] inspector_panel styled. Setting layout...")
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(15, 20, 15, 15)
        inspector_layout.setSpacing(15)

        print("[UI] Creating lbl_ins_title...")
        lbl_ins_title = QLabel("<span style='font-size: 10px; font-weight: bold; color: #71717a;'>🔍 KAVRAM DETAYLARI</span>")
        inspector_layout.addWidget(lbl_ins_title)
        print("[UI] lbl_ins_title created and added.")

        self.lbl_node_name = QLabel("<span style='font-size: 15px; font-weight: bold; color: #06b6d4;'>Seçili Düğüm: Yok</span>")
        self.lbl_node_name.setWordWrap(True)
        inspector_layout.addWidget(self.lbl_node_name)

        self.lbl_node_desc = QLabel("<div style='font-size: 12px; color: #a1a1aa;'>Haritadan bir kavram veya doküman düğümüne tıklayarak ilişkilerini ve açıklamasını görün.</div>")
        self.lbl_node_desc.setWordWrap(True)
        inspector_layout.addWidget(self.lbl_node_desc)

        self.btn_ask_about_node = QPushButton("💬 Bu Kavramı Sor")
        self.btn_ask_about_node.setStyleSheet("""
            QPushButton {
                background-color: #0f766e;
                border: 1px solid #0d9488;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0d9488;
            }
        """)
        self.btn_ask_about_node.clicked.connect(self.ask_about_selected_node)
        self.btn_ask_about_node.setEnabled(False)
        inspector_layout.addWidget(self.btn_ask_about_node)

        # Quick layout spacer
        inspector_layout.addStretch()

        self.btn_reset_view = QPushButton("🔄 Görünümü Sıfırla")
        self.btn_reset_view.setStyleSheet("font-size: 11px;")
        self.btn_reset_view.clicked.connect(self.canvas.reset_view)
        inspector_layout.addWidget(self.btn_reset_view)

        splitter.addWidget(self.inspector_panel)
        graph_layout.addWidget(splitter)
        self.tabs.addTab(tab_graph, "🕸️ Kavram Haritası (Mind Map)")

        # TAB 3: Ayarlar
        tab_settings = QWidget()
        settings_layout = QVBoxLayout(tab_settings)
        settings_layout.setContentsMargins(30, 30, 30, 30)
        settings_layout.setSpacing(15)

        lbl_settings_header = QLabel("<span style='font-size: 18px; font-weight: bold; color: #8b5cf6;'>⚙️ Veritabanı ve Model Ayarları</span>")
        settings_layout.addWidget(lbl_settings_header)

        # Connection details forms
        form_layout = QVBoxLayout()
        form_layout.setSpacing(10)

        form_layout.addWidget(QLabel("Neo4j Bolt Adresi (URI):"))
        self.txt_neo_uri = QLineEdit()
        self.txt_neo_uri.setText(self.config["neo4j_uri"])
        form_layout.addWidget(self.txt_neo_uri)

        form_layout.addWidget(QLabel("Neo4j Kullanıcı Adı:"))
        self.txt_neo_user = QLineEdit()
        self.txt_neo_user.setText(self.config["neo4j_user"])
        form_layout.addWidget(self.txt_neo_user)

        form_layout.addWidget(QLabel("Neo4j Şifre:"))
        self.txt_neo_pass = QLineEdit()
        self.txt_neo_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_neo_pass.setText(self.config["neo4j_password"])
        form_layout.addWidget(self.txt_neo_pass)

        form_layout.addWidget(QLabel("Ollama Sentezleme Modeli (Synthesis):"))
        self.txt_synth_model = QLineEdit()
        self.txt_synth_model.setText(self.config["synth_model"])
        form_layout.addWidget(self.txt_synth_model)

        form_layout.addWidget(QLabel("Ollama Bağlam Uzunluğu (Context Length - num_ctx):"))
        self.cmb_num_ctx = QComboBox()
        self.cmb_num_ctx.addItems(["2048", "4096", "8192", "16384", "32768", "65536"])
        current_ctx = str(self.config.get("num_ctx", 8192))
        index = self.cmb_num_ctx.findText(current_ctx)
        if index >= 0:
            self.cmb_num_ctx.setCurrentIndex(index)
        else:
            self.cmb_num_ctx.setCurrentText("8192")
        form_layout.addWidget(self.cmb_num_ctx)

        settings_layout.addLayout(form_layout)

        # Test and Save button
        self.btn_save_config = QPushButton("💾 Bağlantıyı Kaydet ve Test Et")
        self.btn_save_config.setStyleSheet("""
            QPushButton {
                background-color: #6d28d9;
                border: 1px solid #7c3aed;
                padding: 12px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        self.btn_save_config.clicked.connect(self.save_and_reconnect)
        settings_layout.addWidget(self.btn_save_config)

        print("[UI] Adding tab_settings...")
        self.tabs.addTab(tab_settings, "⚙️ Ayarlar")
        print("[UI] build_ui finished.")

    # --- UI EVENT HANDLERS ---

    def select_and_upload_pdf(self):
        """Open file dialog, select PDF and trigger ingestion worker."""
        if not self.is_connected:
            QMessageBox.warning(self, "Bağlantı Yok", "Veritabanı bağlantısı yok. Lütfen Neo4j'yi başlatın ve ayarlardan test edin.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "PDF Dokümanı Seç", "", "PDF Files (*.pdf)")
        if not file_path:
            return

        filename = os.path.basename(file_path)

        # Check if document already exists in SQLite
        exists = False
        try:
            cursor = self.vector_db.conn.cursor()
            cursor.execute("SELECT id FROM chunks WHERE document_name = ? LIMIT 1", (filename,))
            exists = (cursor.fetchone() is not None)
        except Exception as e:
            print("Error checking duplicate document:", e)

        if exists:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setWindowTitle("Mükerrer Dosya")
            msg_box.setText(f"'{filename}' isimli dosya sistemde zaten yüklü.")
            msg_box.setInformativeText("Bu dosyanın üzerine yazmak (eski kopyayı silip yenisini yüklemek) ister misiniz?")
            
            overwrite_btn = msg_box.addButton("Üzerine Yaz", QMessageBox.ButtonRole.YesRole)
            cancel_btn = msg_box.addButton("İptal Et", QMessageBox.ButtonRole.NoRole)
            msg_box.setDefaultButton(cancel_btn)
            
            msg_box.exec()
            
            if msg_box.clickedButton() == cancel_btn:
                return
            else:
                self.delete_document_data(filename)

        last_user = getattr(self, "last_meta_user", "System")
        last_project = getattr(self, "last_meta_project", "Default Project")
        last_company = getattr(self, "last_meta_company", "Default Company")

        dialog = DocumentMetadataDialog(filename, last_user, last_project, last_company, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        metadata = dialog.get_metadata()
        self.last_meta_user = metadata["user"]
        self.last_meta_project = metadata["project"]
        self.last_meta_company = metadata["company"]

        self.btn_select_file.setEnabled(False)
        self.btn_select_file.setText("İşleniyor...")
        self.lbl_progress.setText("İşlem başlatılıyor...")
        
        # Start worker thread
        self.ingest_worker = IngestionWorker(self.extractor, file_path, metadata)
        self.ingest_worker.progress.connect(self.on_ingest_progress)
        self.ingest_worker.finished.connect(self.on_ingest_finished)
        self.ingest_worker.error.connect(self.on_ingest_error)
        self.ingest_worker.start()

    def delete_document_data(self, doc_name: str):
        """Deletes all chunks of a document from SQLite and its nodes/relationships from Neo4j."""
        # 1. Delete chunks from SQLite
        try:
            cursor = self.vector_db.conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE document_name = ?", (doc_name,))
            self.vector_db.conn.commit()
            print(f"Deleted SQLite chunks for document: {doc_name}")
        except Exception as e:
            print(f"Error deleting SQLite chunks for document {doc_name}: {e}")

        # 2. Delete nodes and clean orphans from Neo4j
        query_delete_doc = "MATCH (d:Document {name: $doc_name}) DETACH DELETE d"
        query_delete_orphans = "MATCH (c:Concept) WHERE NOT (c)-[]-() DELETE c"
        try:
            with self.graph_db.driver.session() as session:
                session.run(query_delete_doc, doc_name=doc_name)
                session.run(query_delete_orphans)
            print(f"Deleted Neo4j nodes and clean orphans for document: {doc_name}")
        except Exception as e:
            print(f"Error deleting Neo4j nodes for document {doc_name}: {e}")

    def on_ingest_progress(self, status, pct):
        self.lbl_progress.setText(f"<span style='font-size: 11px; color: #a1a1aa; font-style: italic;'>[%{pct}] {status}</span>")

    def on_ingest_finished(self, result):
        self.btn_select_file.setEnabled(True)
        self.btn_select_file.setText("📁 PDF Dosyası Seç")
        self.lbl_progress.setText(f"<span style='font-size: 11px; color: #4ade80;'>Başarıyla eklendi: {result['document_name']}</span>")
        
        QMessageBox.information(
            self, "İşlem Başarılı",
            f"Doküman başarıyla işlendi ve indekslendi:\n"
            f"- Doküman: {result['document_name']}\n"
            f"- Parça Sayısı: {result['total_chunks']}"
        )
        self.update_stats()
        self.update_document_list()
        self.reload_graph_canvas()

    def on_ingest_error(self, err_msg):
        self.btn_select_file.setEnabled(True)
        self.btn_select_file.setText("📁 PDF Dosyası Seç")
        self.lbl_progress.setText("<span style='font-size: 11px; color: #f87171;'>Hata oluştu!</span>")
        QMessageBox.critical(self, "Yükleme Hatası", f"Doküman işlenirken hata oluştu:\n{err_msg}")

    def update_stats(self):
        """Update system stats labels in sidebar."""
        print("[STATS] Starting update_stats...")
        if not self.is_connected:
            print("[STATS] Not connected.")
            self.lbl_stat_docs.setText("Toplam Doküman: Bağlantı Yok")
            self.lbl_stat_chunks.setText("Toplam Parça: Bağlantı Yok")
            self.lbl_stat_nodes.setText("Kavram Düğümü: Bağlantı Yok")
            self.lbl_stat_edges.setText("İlişki Kenarı: Bağlantı Yok")
            return

        # Fetch stats
        try:
            print("[STATS] Querying Vector DB...")
            v_stats = self.vdb.get_stats()
            print("[STATS] Vector DB stats successfully retrieved:", v_stats)
        except Exception as e:
            print("[STATS] FAILED to get Vector DB stats:", e)
            v_stats = {"total_chunks": 0}

        try:
            print("[STATS] Querying Graph DB...")
            g_stats = self.gdb.get_stats()
            print("[STATS] Graph DB stats successfully retrieved:", g_stats)
        except Exception as e:
            print("[STATS] FAILED to get Graph DB stats:", e)
            g_stats = {
                "total_users": 0, "total_companies": 0, "total_projects": 0, 
                "total_documents": 0, "total_concepts": 0, "total_edges": 0
            }
        
        try:
            print("[STATS] Setting text on QLabels...")
            self.lbl_stat_docs.setText(f"Toplam Doküman: {g_stats['total_documents']} ({g_stats['total_projects']} Proje)")
            self.lbl_stat_chunks.setText(f"Toplam Parça: {v_stats['total_chunks']}")
            self.lbl_stat_nodes.setText(f"Kavram Düğümü: {g_stats['total_concepts']} ({g_stats['total_users']} Yükleyen)")
            self.lbl_stat_edges.setText(f"Toplam İlişki: {g_stats['total_edges']}")
            print("[STATS] Labels set successfully.")
        except Exception as e:
            print("[STATS] FAILED to set text on labels:", e)

    def update_document_list(self):
        """Fetch uploaded documents and populate list view."""
        self.tbl_files.setRowCount(0)
        if not self.is_connected:
            return

        try:
            # Query Neo4j to find document nodes with their file paths
            with self.gdb.driver.session() as session:
                res = session.run("MATCH (n:Document) RETURN n.id AS id, n.name AS name, n.file_path AS file_path")
                
                for idx, record in enumerate(res):
                    doc_id = record["id"]
                    doc_name = record["name"]
                    doc_file_path = record["file_path"] or ""
                    
                    self.tbl_files.insertRow(idx)
                    # File Name
                    name_item = QTableWidgetItem(doc_name)
                    name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    name_item.setData(Qt.ItemDataRole.UserRole, doc_file_path)
                    name_item.setToolTip("Dosyayı varsayılan programla açmak için çift tıklayın.")
                    self.tbl_files.setItem(idx, 0, name_item)
                    
                    # Action Buttons Container Widget
                    actions_widget = QWidget()
                    actions_layout = QHBoxLayout(actions_widget)
                    actions_layout.setContentsMargins(2, 2, 2, 2)
                    actions_layout.setSpacing(6)
                    
                    # Open button
                    btn_open = QPushButton("Aç")
                    btn_open.setStyleSheet("""
                        QPushButton {
                            background-color: #0369a1;
                            border: 1px solid #0284c7;
                            color: #e0f2fe;
                            font-size: 10px;
                            padding: 2px 6px;
                            border-radius: 4px;
                        }
                        QPushButton:hover {
                            background-color: #0284c7;
                        }
                    """)
                    btn_open.setEnabled(bool(doc_file_path))
                    btn_open.clicked.connect(lambda checked, path=doc_file_path: self.open_document_file(path))
                    actions_layout.addWidget(btn_open)
                    
                    # Delete button
                    btn_del = QPushButton("Sil")
                    btn_del.setStyleSheet("""
                        QPushButton {
                            background-color: #450a0a;
                            border: 1px solid #7f1d1d;
                            color: #fca5a5;
                            font-size: 10px;
                            padding: 2px 6px;
                            border-radius: 4px;
                        }
                        QPushButton:hover {
                            background-color: #7f1d1d;
                        }
                    """)
                    btn_del.clicked.connect(lambda checked, d_id=doc_id, d_name=doc_name: self.delete_document(d_id, d_name))
                    actions_layout.addWidget(btn_del)
                    
                    self.tbl_files.setCellWidget(idx, 1, actions_widget)
        except Exception as e:
            print(f"Document list loading failed: {e}")

    def on_file_table_double_clicked(self, row: int, column: int):
        """Handle double-clicking a file in the table to open it."""
        if column == 0:
            item = self.tbl_files.item(row, column)
            if item:
                file_path = item.data(Qt.ItemDataRole.UserRole)
                self.open_document_file(file_path)

    def open_document_file(self, file_path: str):
        """Open the document using the default system program."""
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Dosya Bulunamadı", f"Dosya sistemde bulunamadı veya taşınmış:\n{file_path}")
            return
            
        try:
            os.startfile(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Dosya açılırken hata oluştu:\n{e}")

    def delete_document(self, doc_id: str, doc_name: str):
        """Delete specific document chunks and concept nodes from databases."""
        confirm = QMessageBox.question(
            self, "Doküman Sil",
            f"'{doc_name}' isimli dokümanı ve tüm ilişkili verilerini silmek istediğinize emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                # 1. Delete from Vector DB
                self.vdb.delete_document_chunks(doc_id)
                
                # 2. Delete from Graph DB (Neo4j)
                # We delete the document node and any concepts that were ONLY discussed in this document.
                # If a concept belongs to multiple documents, we just remove the doc_id from its chunk_ids.
                with self.gdb.driver.session() as session:
                    # Detach delete the document node and all its organizational links
                    session.run("MATCH (d:Document {id: $doc_id}) DETACH DELETE d", doc_id=doc_id)
                    
                    # Update concepts: remove references to chunks of this document, and delete concepts belonging to this document
                    session.run("""
                    MATCH (c:Concept)
                    WHERE c.doc_id = $doc_id
                    DETACH DELETE c
                    """, doc_id=doc_id)
                    
                QMessageBox.information(self, "Silindi", f"'{doc_name}' başarıyla silindi.")
                self.update_stats()
                self.update_document_list()
                self.reload_graph_canvas()
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Silme işlemi sırasında hata oluştu:\n{e}")

    def reset_system(self):
        """Completely reset all databases."""
        if not self.is_connected:
            return

        confirm = QMessageBox.question(
            self, "Tüm Sistemi Sıfırla",
            "Tüm vektör dizinini ve Neo4j graf veritabanını temizlemek üzeresiniz.\nBu işlem GERİ ALINAMAZ!\n\nDevam etmek istiyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if confirm == QMessageBox.StandardButton.Yes:
            try:
                self.vdb.reset_db()
                self.gdb.clear_graph()
                QMessageBox.information(self, "Başarılı", "Tüm veritabanı dizinleri başarıyla sıfırlandı.")
                self.update_stats()
                self.update_document_list()
                self.reload_graph_canvas()
            except Exception as e:
                QMessageBox.critical(self, "Sıfırlama Hatası", f"Sıfırlama sırasında hata oluştu:\n{e}")

    def reload_graph_canvas(self):
        """Fetch all graph nodes and edges and load them in the painter canvas."""
        if not self.is_connected:
            return
        try:
            graph_data = self.gdb.get_graph_data()
            self.canvas.set_graph_data(graph_data)
        except Exception as e:
            print(f"Graph reload failed: {e}")

    def showEvent(self, event):
        """Called when window is displayed; loads the initial graph visualization."""
        super().showEvent(event)
        self.reload_graph_canvas()

    def inspect_concept(self, node_properties: dict):
        """Called when canvas node is clicked. Shows details in inspector panel."""
        name = node_properties["label"]
        desc = node_properties["description"]
        node_label = node_properties.get("node_label") or "Concept"
        concept_type = node_properties.get("concept_type") or "Other"

        self.lbl_node_name.setText(f"<div style='font-size: 15px; font-weight: bold; color: #06b6d4;'>{name}</div>")
        
        meta_info = f"<div style='font-size: 10px; color: #71717a; font-weight: bold; margin-bottom: 5px;'>TÜR: {node_label.upper()}"
        if node_label.upper() == "CONCEPT" and concept_type.upper() != "OTHER":
            meta_info += f" ({concept_type.upper()})"
        meta_info += "</div>"
        
        display_desc = desc if desc else "Açıklama girilmemiş."
        self.lbl_node_desc.setText(f"{meta_info}<div style='font-size: 12px; color: #a1a1aa;'>{display_desc}</div>")

        # Enable ask about node button
        self.btn_ask_about_node.setEnabled(True)
        self.btn_ask_about_node.setText(f"💬 '{name}' Hakkında Sor")

    def ask_about_selected_node(self):
        """Inserts a question about the selected concept in chat input."""
        name = self.lbl_node_name.text()
        if name and name != "Seçili Düğüm: Yok":
            self.txt_query.setText(f"{name} nedir ve ne işe yarar?")
            self.tabs.setCurrentIndex(0) # Switch to Chat tab
            self.submit_query()

    def submit_query(self):
        """Submit query to retrieval worker."""
        if not self.is_connected:
            QMessageBox.warning(self, "Bağlantı Yok", "Veritabanı bağlantısı yok. Lütfen Neo4j'yi başlatın.")
            return

        query_text = self.txt_query.text().strip()
        if not query_text:
            return

        self.txt_query.clear()
        self.txt_query.setEnabled(False)
        self.btn_send.setEnabled(False)

        # Add User Message to Chat UI
        user_msg = MessageWidget(query_text, is_user=True)
        # Add to scroll layout before the spacer
        self.chat_box_layout.insertWidget(self.chat_box_layout.count() - 1, user_msg)
        
        # Add a temporary loading indicator bubble for AI
        self.loading_bubble = MessageWidget("HERMES düşünüyor...", is_user=False)
        self.chat_box_layout.insertWidget(self.chat_box_layout.count() - 1, self.loading_bubble)
        
        # Scroll to bottom
        QTimer.singleShot(50, self.scroll_chat_to_bottom)

        # Trigger Query Thread (with history context)
        self.query_worker = QueryWorker(self.retriever, self.synthesizer, query_text, self.chat_history)
        self.query_worker.finished.connect(self.on_query_finished)
        self.query_worker.error.connect(self.on_query_error)
        self.query_worker.start()

    def on_query_finished(self, result):
        self.txt_query.setEnabled(True)
        self.btn_send.setEnabled(True)
        
        # Remove loading indicator
        self.chat_box_layout.removeWidget(self.loading_bubble)
        self.loading_bubble.deleteLater()

        # Add AI response widget
        ai_msg = MessageWidget(
            text=result["answer"],
            is_user=False,
            thought=result["thought"],
            sources=result["sources"],
            source_paths=result.get("source_paths"),
            source_metadata=result.get("source_metadata")
        )
        self.chat_box_layout.insertWidget(self.chat_box_layout.count() - 1, ai_msg)
        
        # Update chat history memory (keep only last 10 messages / 5 turns)
        self.chat_history.append({"role": "user", "content": result["query"]})
        self.chat_history.append({"role": "assistant", "content": result["answer"]})
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        # Scroll to bottom
        QTimer.singleShot(50, self.scroll_chat_to_bottom)
        
        # If the query led to finding any new concepts, we can highlight them in the canvas!
        if result["concepts"]:
            # Highlight first concept in Canvas if we are looking at Mind Map
            first_concept = result["concepts"][0]["id"]
            self.canvas.selected_node_id = first_concept
            self.canvas.update()

    def on_query_error(self, err_msg):
        self.txt_query.setEnabled(True)
        self.btn_send.setEnabled(True)
        
        # Remove loading indicator
        self.chat_box_layout.removeWidget(self.loading_bubble)
        self.loading_bubble.deleteLater()

        ai_msg = MessageWidget(
            text=f"Arama ve sentezleme sırasında bir hata oluştu:\n{err_msg}",
            is_user=False
        )
        self.chat_box_layout.insertWidget(self.chat_box_layout.count() - 1, ai_msg)
        QTimer.singleShot(50, self.scroll_chat_to_bottom)

    def clear_chat_history(self):
        """Clears current chat bubbles from UI and resets memory."""
        self.chat_history = []
        for i in reversed(range(self.chat_box_layout.count())):
            item = self.chat_box_layout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.chat_box_layout.addStretch()

    def scroll_chat_to_bottom(self):
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )

    def save_and_reconnect(self):
        """Save settings config file and test connection to Neo4j."""
        new_config = {
            "neo4j_uri": self.txt_neo_uri.text().strip(),
            "neo4j_user": self.txt_neo_user.text().strip(),
            "neo4j_password": self.txt_neo_pass.text().strip(),
            "embed_model": self.config.get("embed_model", "qwen3-embedding:4b"),
            "extract_model": self.config.get("extract_model", "qwen3:4b"),
            "synth_model": self.txt_synth_model.text().strip(),
            "num_ctx": int(self.cmb_num_ctx.currentText())
        }
        
        # Save locally
        save_config(new_config)
        self.config = new_config

        # Try to setup database again
        self.btn_save_config.setText("Bağlanıyor...")
        self.btn_save_config.setEnabled(False)
        QApplication.processEvents()

        self.setup_databases()

        self.btn_save_config.setText("💾 Bağlantıyı Kaydet ve Test Et")
        self.btn_save_config.setEnabled(True)

        if self.is_connected:
            self.lbl_connection_banner.hide()
            QMessageBox.information(self, "Bağlantı Başarılı", "Neo4j ve model servislerine başarıyla bağlanıldı!")
            self.update_stats()
            self.update_document_list()
            self.reload_graph_canvas()
            self.tabs.setCurrentIndex(0) # Switch back to Chat tab
        else:
            self.lbl_connection_banner.show()
            self.lbl_connection_banner.setText(f"HATA: Bağlantı başarısız!")
            QMessageBox.critical(self, "Bağlantı Hatası", f"Veritabanına bağlanılamadı:\n{self.connection_error_msg}")


# GUI Main launcher
def main():
    print("[HERMES] Uygulama başlatılıyor...")
    try:
        app = QApplication(sys.argv)
        print("[HERMES] QApplication başarıyla başlatıldı.")
        
        window = HermesMainWindow()
        print("[HERMES] HermesMainWindow pencere sınıfı başarıyla oluşturuldu.")
        
        window.show()
        print("[HERMES] Pencere gösterildi. Event loop'a giriliyor...")
        
        exit_code = app.exec()
        print(f"[HERMES] Event loop sonlandı. Çıkış kodu: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"[HERMES] KRİTİK ÇALIŞMA ZAMANI HATASI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
