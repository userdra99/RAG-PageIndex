# PageIndex Web UI Guide

## 🎉 Your Document Analysis & Chat Interface is Live!

Access the web UI at: **http://localhost:8090**

---

## Features

### 📂 Document Management
- **Upload Documents**: Drag & drop or click to upload PDF or Markdown files
- **Auto-Processing**: Documents are automatically analyzed with Qwen3-32B
- **View Structure**: See intelligent document structure extraction
- **Delete Documents**: Remove documents and their processed data

### 💬 Intelligent Chat
- **General Chat**: Ask any question to Qwen3-32B reasoning model
- **Context-Aware**: Select a processed document for context-aware responses
- **Chat History**: All conversations are saved and persistent
- **Reasoning Visibility**: See the model's thinking process in expandable sections

### 🧠 AI Features
- **Qwen3-32B Model**: Powered by advanced reasoning capabilities
- **Dual GPU Acceleration**: Running on RTX 5090 GPUs for fast responses
- **Document Context**: AI understands document structure for better answers

---

## Quick Start Guide

### 1. Upload a Document

1. Click the **"Upload Document"** button in the left sidebar
2. Select a PDF or Markdown file (max 100MB)
3. File uploads automatically
4. Processing starts immediately (watch the status indicator)

### 2. Chat Without Documents

1. Type your question in the chat input at the bottom
2. Press **Enter** or click the **Send** button
3. Watch the AI reason through your question
4. Click **"💭 Reasoning Process"** to see how it thinks

### 3. Chat With Document Context

1. Wait for document processing to complete (status shows "✓ Processed")
2. **Click on a processed document** in the sidebar
3. Document name appears at the top of chat (context indicator)
4. Ask questions about the document
5. AI uses document structure to provide better answers

### 4. Manage Documents

- **View Status**: Each document shows upload size and processing status
- **Process Document**: Click the clock icon to manually process
- **Delete Document**: Click the trash icon to remove
- **Select for Context**: Click document name to use as context

### 5. Manage Chat History

- **Clear History**: Click "Clear History" button at the bottom
- **Persistent**: Chat history survives container restarts
- **Automatic Saving**: Every message is saved automatically

---

## UI Walkthrough

### Header
```
📚 PageIndex
Intelligent Document Analysis with Qwen3-32B
```
Shows application name and current model.

### Left Sidebar - Documents
```
📂 Documents
  [Upload Document Button]

  📄 sample.pdf
     2.3 MB
     ✓ Processed

  📄 research.md
     156 KB
     ⋯ Not Processed
```

### Main Chat Area
```
👋 Welcome to PageIndex
Upload a document to analyze its structure, or start chatting right away!

[Feature highlights shown on first load]
```

### Context Indicator (when document selected)
```
📄 Chatting with context from: sample.pdf [✕]
```
Click ✕ to clear context and chat normally.

### Chat Messages
```
User Message:
  What is quantum computing?

AI Response:
  💭 Reasoning Process (click to expand)
  [AI's thinking process shown here]

  Quantum computing is...
```

### Chat Input
```
[Text area: "Ask a question or chat with the AI..."]  [Send →]
[Clear History]
```

---

## API Endpoints

The web UI uses these REST API endpoints:

### Document Management
- `GET /api/documents` - List all documents
- `POST /api/upload` - Upload new document
- `DELETE /api/documents/<filename>` - Delete document
- `POST /api/process/<filename>` - Process document
- `GET /api/document/<filename>/structure` - Get document structure

### Chat
- `POST /api/chat` - Send chat message
  ```json
  {
    "message": "Your question here",
    "document": "optional-document.pdf"
  }
  ```
- `GET /api/chat/history` - Get chat history
- `DELETE /api/chat/history` - Clear chat history

### System
- `GET /health` - Health check

---

## Example Use Cases

### 1. Analyze a Research Paper

1. Upload your PDF research paper
2. Wait for processing (30-60 seconds)
3. Click the document to select it
4. Ask: "What are the main findings?"
5. Ask: "Summarize the methodology section"

### 2. General Knowledge Questions

1. Don't select any document
2. Ask: "Explain quantum entanglement"
3. Watch the reasoning process
4. Follow up with more questions

### 3. Compare Multiple Documents

1. Upload multiple related documents
2. Process all of them
3. Switch between documents by clicking them
4. Ask comparative questions

### 4. Extract Document Structure

1. Upload a long document
2. After processing, click the document
3. Ask: "What are the main sections?"
4. Ask: "Create an outline of this document"

---

## Tips & Tricks

### Performance
- **Upload Size**: Keep files under 100MB for best performance
- **Processing Time**:
  - Small PDFs (< 10 pages): ~30 seconds
  - Medium PDFs (10-50 pages): 1-3 minutes
  - Large PDFs (50+ pages): 3-5 minutes
  - Markdown: Usually < 30 seconds

### Chat Features
- **Shift+Enter**: Add new line in chat input
- **Enter**: Send message
- **Click Reasoning**: Expand/collapse AI's thinking process
- **Context Switching**: Switch documents mid-conversation

### Document Tips
- **Best Formats**: Well-structured PDFs with clear headings
- **Markdown**: Use proper heading hierarchy (# ## ###)
- **Naming**: Use descriptive filenames (easier to select)

### Chat Best Practices
- **Be Specific**: "Summarize chapter 3" vs "Tell me about the document"
- **Follow Up**: Ask related questions for deeper understanding
- **Context Matters**: Select relevant document for better answers

---

## Troubleshooting

### Document Won't Upload
- Check file size (max 100MB)
- Verify file format (PDF, MD, MARKDOWN only)
- Check browser console for errors

### Processing Takes Too Long
- Large documents need more time
- Check Docker logs: `docker logs pageindex-app`
- Verify vLLM is healthy: `docker ps`

### Chat Not Responding
- Check vLLM is running: `http://localhost:8000/health`
- View logs: `docker logs pageindex-app`
- Verify GPU usage: `nvidia-smi`

### Can't See Document Context
- Document must be processed first (✓ Processed status)
- Click document name to select
- Look for context indicator at top of chat

### Chat History Lost
- Chat history is stored in `/app/data/chat_history.json`
- To persist: mount volume in docker-compose.yml
- Default: survives container restarts but not container removals

---

## Architecture

```
┌─────────────────────────────────────┐
│   Browser (localhost:8090)          │
│   ┌─────────────────────────────┐   │
│   │  HTML/CSS/JS Frontend       │   │
│   │  - Document Manager         │   │
│   │  - Chat Interface           │   │
│   │  - Real-time Updates        │   │
│   └─────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │ HTTP/REST API
               ▼
┌─────────────────────────────────────┐
│  Flask Backend (pageindex-app)      │
│  ┌─────────────────────────────┐   │
│  │  API Endpoints              │   │
│  │  - File Upload/Management   │   │
│  │  - Document Processing      │   │
│  │  - Chat with Context        │   │
│  │  - History Persistence      │   │
│  └─────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
      ┌────────┴─────────┐
      ▼                  ▼
┌──────────────┐   ┌──────────────────┐
│  PageIndex   │   │  vLLM (Port 8000)│
│  Processing  │   │  Qwen3-32B-AWQ   │
│  Library     │   │  Dual RTX 5090   │
└──────────────┘   └──────────────────┘
```

---

## Data Storage

All data is stored in `/app/data/` inside the container:

```
/app/data/
├── uploads/           # Uploaded documents
├── results/           # Processed structures (JSON)
└── chat_history.json  # Chat conversation history
```

**Note**: By default, data is lost when container is removed. To persist data, add volume mounts in `docker-compose.yml`.

---

## Security Notes

⚠️ **Development Server**: This uses Flask's development server
- **Not for production**: Use Gunicorn/uWSGI for production
- **No authentication**: Anyone with access can use it
- **Local only**: Binds to 0.0.0.0 but only accessible via localhost:8090

**Recommendations for Production**:
1. Add authentication (Flask-Login, OAuth)
2. Use HTTPS (nginx reverse proxy with SSL)
3. Use production WSGI server (Gunicorn)
4. Add rate limiting (Flask-Limiter)
5. Implement file size/type validation
6. Add CORS restrictions

---

## Advanced Configuration

### Environment Variables

Set in `.env` or docker-compose.yml:

```bash
VLLM_BASE_URL=http://vllm:8000/v1
DEFAULT_MODEL=Qwen/Qwen3-32B-AWQ
CHATGPT_API_KEY=not-needed
```

### Flask Configuration

Edit `webapp/app.py`:

```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = '/app/data/uploads'
```

### Document Processing Options

Default processing settings in `webapp/app.py`:

```python
# For PDFs
opt = config(
    model=DEFAULT_MODEL,
    toc_check_page_num=20,
    max_page_num_each_node=10,
    max_token_num_each_node=20000,
    if_add_node_summary='yes'
)
```

---

## Browser Support

✅ **Fully Supported**:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

⚠️ **Partial Support**:
- IE 11 (limited CSS features)

---

## Keyboard Shortcuts

- `Enter` - Send message
- `Shift + Enter` - New line in chat
- `Ctrl/Cmd + K` - Focus chat input (coming soon)
- `Escape` - Clear document context (coming soon)

---

## What's Next?

Potential enhancements:
- [ ] Real-time document processing progress
- [ ] Multiple document comparison
- [ ] Export chat to PDF
- [ ] Document annotations
- [ ] Advanced search in documents
- [ ] User authentication
- [ ] Document sharing
- [ ] API key management

---

## Support

- **Logs**: `docker logs pageindex-app -f`
- **vLLM Status**: `docker logs pageindex-vllm -f`
- **Container Status**: `docker ps --filter "name=pageindex"`
- **GPU Usage**: `nvidia-smi -l 1`

**Restart Everything**:
```bash
docker compose -f config/docker-compose.yml restart
```

**Access UI**: http://localhost:8090

Enjoy using PageIndex! 🚀
