# vLLM Integration Guide for PageIndex

## Quick Start Integration

### 1. Basic Setup

```javascript
// config/vllm.js
const axios = require('axios');

class VLLMClient {
  constructor(config = {}) {
    this.baseURL = config.baseURL || process.env.VLLM_API_URL || 'http://vllm:8000';
    this.modelName = config.modelName || process.env.VLLM_MODEL_NAME || 'Qwen/Qwen2.5-7B-Instruct';

    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.status === 200;
    } catch (error) {
      console.error('vLLM health check failed:', error.message);
      return false;
    }
  }

  async listModels() {
    const response = await this.client.get('/v1/models');
    return response.data;
  }

  async complete(prompt, options = {}) {
    const response = await this.client.post('/v1/completions', {
      model: this.modelName,
      prompt: prompt,
      max_tokens: options.maxTokens || 512,
      temperature: options.temperature || 0.7,
      top_p: options.topP || 0.9,
      stream: options.stream || false
    });
    return response.data;
  }

  async chat(messages, options = {}) {
    const response = await this.client.post('/v1/chat/completions', {
      model: this.modelName,
      messages: messages,
      max_tokens: options.maxTokens || 512,
      temperature: options.temperature || 0.7,
      top_p: options.topP || 0.9,
      stream: options.stream || false
    });
    return response.data;
  }
}

module.exports = VLLMClient;
```

### 2. Application Integration

```javascript
// app/services/ai.service.js
const VLLMClient = require('../config/vllm');

class AIService {
  constructor() {
    this.vllm = new VLLMClient();
    this.initialized = false;
  }

  async initialize() {
    const healthy = await this.vllm.healthCheck();
    if (!healthy) {
      throw new Error('vLLM service is not available');
    }
    this.initialized = true;
    console.log('AI Service initialized successfully');
  }

  async generateResponse(userMessage, context = []) {
    if (!this.initialized) {
      await this.initialize();
    }

    const messages = [
      { role: 'system', content: 'You are a helpful AI assistant for PageIndex.' },
      ...context,
      { role: 'user', content: userMessage }
    ];

    const response = await this.vllm.chat(messages, {
      maxTokens: 1024,
      temperature: 0.7
    });

    return response.choices[0].message.content;
  }

  async analyzeContent(content) {
    if (!this.initialized) {
      await this.initialize();
    }

    const prompt = `Analyze the following content and provide key insights:\n\n${content}`;
    const response = await this.vllm.complete(prompt, {
      maxTokens: 512,
      temperature: 0.5
    });

    return response.choices[0].text.trim();
  }
}

module.exports = new AIService();
```

### 3. Express API Integration

```javascript
// app/routes/ai.routes.js
const express = require('express');
const router = express.Router();
const aiService = require('../services/ai.service');

// Health check endpoint
router.get('/health', async (req, res) => {
  try {
    const healthy = await aiService.vllm.healthCheck();
    res.json({ status: healthy ? 'healthy' : 'unhealthy' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Chat endpoint
router.post('/chat', async (req, res) => {
  try {
    const { message, context } = req.body;
    const response = await aiService.generateResponse(message, context);
    res.json({ response });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Content analysis endpoint
router.post('/analyze', async (req, res) => {
  try {
    const { content } = req.body;
    const analysis = await aiService.analyzeContent(content);
    res.json({ analysis });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
```

### 4. Main Application Entry

```javascript
// index.js
const express = require('express');
const aiRoutes = require('./app/routes/ai.routes');
const aiService = require('./app/services/ai.service');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// AI routes
app.use('/api/ai', aiRoutes);

// Initialize AI service on startup
async function startServer() {
  try {
    await aiService.initialize();
    app.listen(PORT, () => {
      console.log(`PageIndex running on port ${PORT}`);
      console.log(`vLLM API: ${process.env.VLLM_API_URL}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error.message);
    process.exit(1);
  }
}

startServer();
```

## API Endpoints Reference

### vLLM OpenAI-Compatible Endpoints

#### 1. List Models
```bash
GET http://vllm:8000/v1/models
```

#### 2. Completions
```bash
POST http://vllm:8000/v1/completions
Content-Type: application/json

{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt": "Your prompt here",
  "max_tokens": 512,
  "temperature": 0.7
}
```

#### 3. Chat Completions
```bash
POST http://vllm:8000/v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

#### 4. Health Check
```bash
GET http://vllm:8000/health
```

## Configuration Options

### vLLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 512 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature (0-2) |
| `top_p` | 0.9 | Nucleus sampling parameter |
| `frequency_penalty` | 0 | Penalize frequent tokens |
| `presence_penalty` | 0 | Penalize present tokens |
| `stop` | null | Stop sequences |
| `stream` | false | Enable streaming responses |

### Environment Variables

```bash
# Required
VLLM_API_URL=http://vllm:8000
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# Optional
VLLM_TIMEOUT=30000
VLLM_MAX_RETRIES=3
VLLM_RETRY_DELAY=1000
```

## Error Handling

```javascript
// Robust error handling example
async function safeAICall(operation) {
  const maxRetries = 3;
  let lastError;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;

      if (error.response) {
        // vLLM returned an error
        console.error(`vLLM error (attempt ${attempt}):`, error.response.data);

        if (error.response.status === 503) {
          // Service unavailable, retry
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
          continue;
        }
      } else if (error.request) {
        // Network error
        console.error(`Network error (attempt ${attempt}):`, error.message);
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        continue;
      }

      // Other errors, don't retry
      break;
    }
  }

  throw new Error(`AI operation failed after ${maxRetries} attempts: ${lastError.message}`);
}

// Usage
const response = await safeAICall(() =>
  aiService.generateResponse("Hello")
);
```

## Testing Integration

```javascript
// test/ai.service.test.js
const aiService = require('../app/services/ai.service');

describe('AI Service Integration', () => {
  beforeAll(async () => {
    await aiService.initialize();
  });

  test('should generate response', async () => {
    const response = await aiService.generateResponse('Hello');
    expect(response).toBeTruthy();
    expect(typeof response).toBe('string');
  });

  test('should analyze content', async () => {
    const analysis = await aiService.analyzeContent('Test content');
    expect(analysis).toBeTruthy();
  });

  test('should handle errors gracefully', async () => {
    await expect(
      aiService.generateResponse('')
    ).rejects.toThrow();
  });
});
```

## Performance Optimization

### 1. Connection Pooling
```javascript
// Use keep-alive for persistent connections
const axios = require('axios');
const http = require('http');

const httpAgent = new http.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 50
});

const vllmClient = axios.create({
  baseURL: process.env.VLLM_API_URL,
  httpAgent: httpAgent
});
```

### 2. Caching Responses
```javascript
const NodeCache = require('node-cache');
const cache = new NodeCache({ stdTTL: 300 });

async function getCachedResponse(prompt) {
  const cacheKey = `ai_${Buffer.from(prompt).toString('base64')}`;
  const cached = cache.get(cacheKey);

  if (cached) {
    return cached;
  }

  const response = await aiService.generateResponse(prompt);
  cache.set(cacheKey, response);
  return response;
}
```

### 3. Request Batching
```javascript
const PQueue = require('p-queue');
const queue = new PQueue({ concurrency: 5 });

async function processMultipleRequests(requests) {
  return Promise.all(
    requests.map(req =>
      queue.add(() => aiService.generateResponse(req))
    )
  );
}
```

## Monitoring

```javascript
// middleware/monitor.js
const aiMetrics = {
  requests: 0,
  errors: 0,
  totalLatency: 0
};

function monitorAI(req, res, next) {
  const start = Date.now();

  res.on('finish', () => {
    aiMetrics.requests++;
    aiMetrics.totalLatency += Date.now() - start;

    if (res.statusCode >= 400) {
      aiMetrics.errors++;
    }
  });

  next();
}

app.get('/metrics', (req, res) => {
  res.json({
    ...aiMetrics,
    averageLatency: aiMetrics.totalLatency / aiMetrics.requests,
    errorRate: aiMetrics.errors / aiMetrics.requests
  });
});
```

## Best Practices

1. **Always check health before requests**
2. **Implement retry logic with exponential backoff**
3. **Use appropriate timeouts (30s recommended)**
4. **Cache responses when possible**
5. **Monitor API usage and performance**
6. **Handle streaming responses properly**
7. **Validate input before sending to vLLM**
8. **Use connection pooling for efficiency**
