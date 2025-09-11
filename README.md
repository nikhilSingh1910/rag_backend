# 📄 Document Management & RAG-based Q&A Application

##  Overview
This application provides **document management** and a **Q&A system powered by Retrieval-Augmented Generation (RAG)**.  
Users can upload documents, which are ingested and converted into embeddings for **semantic search**.  
The backend supports **user management, document storage, and retrieval-based Q&A** functionality.  

---

##  Features
-  **User Management**
  - User authentication & authorization  
  - Role-based access (e.g., admin, regular user)  

-  **Document Management**
  - Upload and manage documents  
  - Metadata storage (title, author, tags, upload date)  
  - Support for multiple formats (PDF, TXT, DOCX, etc.)  

-  **Ingestion & Embeddings**
  - Automatic document chunking  
  - Embedding generation using LLM/Vector models  
  - Storage in a vector database for fast retrieval  

-  **RAG-based Q&A**
  - User queries answered using retrieved document context  
  - Combines LLM reasoning with relevant document knowledge  
  - Supports multi-turn conversations  

---
