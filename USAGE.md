# PageIndex Usage Guide

## Overview
PageIndex uses Qwen3-32B-AWQ reasoning model to intelligently analyze document structures, creating hierarchical representations of PDFs and Markdown files.

## Quick Start

### 1. Process a PDF Document

```bash
# Copy PDF to container
docker cp /path/to/document.pdf pageindex-app:/app/data/

# Process with default settings
docker exec pageindex-app python run_pageindex.py \
  --pdf_path /app/data/document.pdf \
  --model Qwen/Qwen3-32B-AWQ

# Retrieve results
docker cp pageindex-app:/app/results/document_structure.json ./results/
```

### 2. Process a Markdown Document

```bash
docker cp /path/to/document.md pageindex-app:/app/data/

docker exec pageindex-app python run_pageindex.py \
  --md_path /app/data/document.md \
  --model Qwen/Qwen3-32B-AWQ \
  --if-thinning yes \
  --thinning-threshold 5000
```

### 3. Interactive Mode

```bash
# Enter container shell
docker exec -it pageindex-app bash

# Inside container:
cd /app

# Process documents
python run_pageindex.py --pdf_path /app/data/myfile.pdf \
  --model Qwen/Qwen3-32B-AWQ \
  --if-add-node-summary yes \
  --if-add-doc-description yes

# View results
cat results/myfile_structure.json | jq .

# Exit
exit
```

## Configuration Options

### PDF Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pdf_path` | - | Path to PDF file (required) |
| `--model` | Qwen/Qwen3-32B-AWQ | Model to use |
| `--toc-check-pages` | 20 | Pages to scan for table of contents |
| `--max-pages-per-node` | 10 | Maximum pages per structure node |
| `--max-tokens-per-node` | 20000 | Token limit per node |
| `--if-add-node-id` | yes | Add unique IDs to nodes |
| `--if-add-node-summary` | yes | Generate AI summaries |
| `--if-add-doc-description` | no | Add document description |
| `--if-add-node-text` | no | Include full text in nodes |

### Markdown Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--md_path` | - | Path to Markdown file (required) |
| `--if-thinning` | no | Apply tree structure optimization |
| `--thinning-threshold` | 5000 | Minimum tokens for thinning |
| `--summary-token-threshold` | 200 | Token threshold for summaries |

## Advanced Examples

### 1. Detailed PDF Analysis with Summaries

```bash
docker exec pageindex-app python run_pageindex.py \
  --pdf_path /app/data/research_paper.pdf \
  --model Qwen/Qwen3-32B-AWQ \
  --if-add-node-summary yes \
  --if-add-doc-description yes \
  --if-add-node-text yes \
  --max-pages-per-node 5
```

### 2. Optimized Markdown Processing

```bash
docker exec pageindex-app python run_pageindex.py \
  --md_path /app/data/documentation.md \
  --model Qwen/Qwen3-32B-AWQ \
  --if-thinning yes \
  --thinning-threshold 10000 \
  --if-add-node-summary yes
```

### 3. Batch Processing Script

```bash
#!/bin/bash
# Process all PDFs in a directory

for pdf in /path/to/pdfs/*.pdf; do
  filename=$(basename "$pdf")
  docker cp "$pdf" pageindex-app:/app/data/
  docker exec pageindex-app python run_pageindex.py \
    --pdf_path "/app/data/$filename" \
    --model Qwen/Qwen3-32B-AWQ
  docker cp "pageindex-app:/app/results/${filename%.pdf}_structure.json" ./results/
done
```

## Output Format

PageIndex generates JSON files with hierarchical document structure:

```json
{
  "title": "Document Title",
  "structure": [
    {
      "node_id": "1",
      "title": "Chapter 1",
      "summary": "AI-generated summary of this section...",
      "pages": [1, 2, 3],
      "children": [
        {
          "node_id": "1.1",
          "title": "Subsection 1.1",
          "summary": "Summary of subsection...",
          "pages": [1]
        }
      ]
    }
  ]
}
```

## System Architecture

```
┌─────────────────────────────────────────┐
│     PageIndex Container (Port 8090)     │
│  ┌───────────────────────────────────┐  │
│  │  PageIndex Python Application     │  │
│  │  - PDF/MD parsing                 │  │
│  │  - Structure extraction           │  │
│  │  - LLM API client                 │  │
│  └────────────┬──────────────────────┘  │
└───────────────┼─────────────────────────┘
                │ HTTP API calls
                │ (Docker network)
                ▼
┌─────────────────────────────────────────┐
│      vLLM Container (Port 8000)         │
│  ┌───────────────────────────────────┐  │
│  │  Qwen3-32B-AWQ Model              │  │
│  │  - Dual RTX 5090 GPUs             │  │
│  │  - Tensor Parallelism (TP=2)      │  │
│  │  - Reasoning capabilities         │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Monitoring

### Check Service Status
```bash
# View container status
docker ps --filter "name=pageindex"

# Monitor vLLM logs
docker logs pageindex-vllm --tail 50 -f

# Monitor PageIndex logs
docker logs pageindex-app --tail 50 -f
```

### Check GPU Usage
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs pageindex-app

# Restart services
docker compose -f config/docker-compose.yml restart
```

### Out of memory errors
```bash
# Reduce batch size in config
docker exec pageindex-app python run_pageindex.py \
  --max-pages-per-node 3 \
  --max-tokens-per-node 10000
```

### Slow processing
- Large PDFs: Reduce `--max-pages-per-node`
- Complex documents: Increase `--toc-check-pages`
- Enable thinning for markdown: `--if-thinning yes`

## Environment Variables

Current configuration (`pageindex-src/.env`):
```bash
VLLM_BASE_URL=http://vllm:8000/v1
CHATGPT_API_KEY=not-needed
DEFAULT_MODEL=Qwen/Qwen3-32B-AWQ
```

## API Access

Both services expose APIs:

- **vLLM API**: `http://localhost:8000/v1/` (OpenAI-compatible)
- **PageIndex**: Container-only (CLI tool)

### Direct vLLM API Test
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

## Performance Notes

- **GPU Memory**: 60.5GB/65.2GB used (dual RTX 5090)
- **Model Loading**: ~3.2 minutes initial load
- **Inference Speed**: ~30-50 tokens/sec (depending on complexity)
- **Reasoning**: Qwen3 shows thinking process in `<think>` tags

## Next Steps

1. **Test with your documents**: Start with small PDFs to understand output
2. **Optimize settings**: Adjust token limits and page counts for your use case
3. **Integrate results**: Parse JSON output into your application
4. **Batch processing**: Create scripts for multiple documents

## Support

- PageIndex GitHub: https://github.com/edwardzjl/PageIndex
- vLLM Docs: https://docs.vllm.ai/
- Qwen3 Model: https://huggingface.co/Qwen/Qwen3-32B-AWQ
