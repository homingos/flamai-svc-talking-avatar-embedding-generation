# Talking Avatar Embedding Generation Service

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-blue.svg)
![Embeddings](https://img.shields.io/badge/Embeddings-SentenceTransformers-purple.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)
![RunPod](https://img.shields.io/badge/RunPod-Serverless-purple.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

A high-performance text embedding generation service that creates vector representations of text using advanced SentenceTransformer models. Built with FastAPI for production-ready REST APIs, powered by PyTorch with GPU acceleration, and optimized for both standalone deployment and serverless environments like RunPod.

## ✨ Features

- 🔍 **Multi-Model Embedding Generation**: Support for various SentenceTransformer models with automatic device detection
- ⚡ **GPU Acceleration**: CUDA-optimized with half-precision support and cuDNN benchmarking for maximum performance
- 🚀 **Multiple Processing Modes**: Single, batch, and parallel processing for different use cases
- 🎯 **Optimized Model Loading**: Essential models first, optional models in background with parallel loading
- 🧠 **Advanced Configuration**: YAML-based configuration with environment variable overrides and validation
- 🎨 **Clean Architecture**: Modular design with server/process management and clear separation of concerns
- 🌐 **Multiple Deployment Options**: FastAPI server, RunPod serverless, and Docker containers
- 📊 **Comprehensive Monitoring**: Built-in health checks, metrics, and system diagnostics
- ⚡ **Production Ready**: Comprehensive logging, error handling, and graceful shutdown
- 🎯 **Serverless Optimized**: RunPod integration for scalable deployment
- 🔧 **Configurable**: YAML-based configuration with environment variable overrides

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd flamai-svc-talking-avatar-embedding-generation

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Environment Configuration

Create a `.env` file with your configuration:

```bash
# Core Configuration
APP_NAME="Embedding Generation Service"
APP_VERSION="1.0.0"
APP_DEBUG=false
APP_ENVIRONMENT="production"

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_RELOAD=false
SERVER_WORKERS=1

# Embedding Generation Configuration
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
EMBEDDING_DEVICE="auto"
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_SEQ_LENGTH=512
EMBEDDING_NORMALIZE=true

# Performance Configuration
ENABLE_HALF_PRECISION=true
ENABLE_CUDNN_BENCHMARK=true
ENABLE_MIXED_PRECISION=true
LAZY_LOAD_OPTIONAL=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# CORS Configuration
CORS_ALLOW_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
CORS_ALLOW_CREDENTIALS=true
```

### 3. Run the Application

#### Option A: FastAPI Server
```bash
# Start the FastAPI server
python app.py

# The server will start on http://localhost:8000
# Visit http://localhost:8000/docs for interactive API documentation
```

#### Option B: RunPod Serverless
```bash
# Deploy to RunPod (requires RunPod account)
# Upload runpod_app.py to your RunPod endpoint
```

#### Option C: Docker Deployment
```bash
# Build the Docker image
docker build -f .docker/Dockerfile.gpu -t embedding-generation-service .

# Run the container
docker run -p 8000:8000 --gpus all embedding-generation-service
```

## 📁 Project Structure

```
flamai-svc-talking-avatar-embedding-generation/
├── app.py                     # FastAPI server entry point
├── runpod_app.py             # RunPod serverless handler
├── pyproject.toml            # Project dependencies and configuration
├── uv.lock                   # Dependency lock file
├── test_input.json           # Test input data for RunPod
├── src/
│   ├── api/                  # API layer
│   │   ├── routes.py         # REST API endpoints
│   │   ├── handlers.py       # Request handlers
│   │   ├── models.py         # Data models
│   │   └── schema.py         # Pydantic schemas
│   ├── core/                 # Core system components
│   │   ├── server_manager.py # Server lifecycle management
│   │   ├── process_manager.py # Process and file management
│   │   └── managers.py       # Manager utilities
│   ├── services/             # Business logic services
│   │   └── embedding_generator.py # Embedding generation service
│   └── utils/                # Utilities and configuration
│       ├── config/           # Configuration management
│       │   ├── config.yaml   # Default configuration
│       │   ├── settings.py   # Settings loader
│       │   └── modules/      # Configuration modules
│       ├── resources/        # Resource utilities
│       │   ├── logger.py     # Logging configuration
│       │   ├── downloader.py # Model downloader
│       │   └── helper.py     # Helper functions
│       ├── auth/             # Authentication utilities
│       └── io/               # Input/output utilities
├── runtime/                  # Runtime directories
│   ├── assets/              # Static assets
│   ├── logs/                # Application logs
│   ├── models/              # Downloaded models
│   ├── outputs/             # Output files
│   ├── temp/                # Temporary files
│   └── uploads/             # Uploaded files
├── .docker/                 # Docker configuration
│   └── Dockerfile.gpu       # GPU-enabled Dockerfile
└── RUNPOD_README.md         # RunPod deployment guide
```

## ⚙️ Configuration

### Main Configuration

The application uses a comprehensive YAML configuration system with environment variable overrides:

```yaml
# src/utils/config/config.yaml
app:
  name: "Embedding Generation Service"
  version: "1.0.0"
  description: "FastAPI service for generating text embeddings using SentenceTransformers"
  debug: false
  environment: "development"

server:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1

embedding_generation:
  # Model configuration
  model:
    name: "all-MiniLM-L6-v2"  # SentenceTransformer model name
    max_seq_length: 512  # Maximum sequence length for tokenization
    normalize_embeddings: true  # Whether to normalize embeddings to unit vectors
  
  # Device configuration
  device:
    auto_detect: true  # Auto-detect best available device
    preferred_device: "auto"  # auto, cpu, cuda
    gpu_memory_fraction: 0.8  # Fraction of GPU memory to use
    clear_cache_on_shutdown: true  # Clear GPU cache on shutdown
  
  # Processing configuration
  processing:
    default_batch_size: 32  # Default batch size for processing
    max_batch_size: 128  # Maximum allowed batch size
    max_parallel_workers: 16  # Maximum number of parallel workers
    default_parallel_workers: 4  # Default number of parallel workers
    max_texts_per_request: 10000  # Maximum texts per parallel request
    max_texts_per_batch: 1000  # Maximum texts per batch request
    max_text_length: 10000  # Maximum length of individual text
  
  # Performance optimization
  optimization:
    enable_half_precision: true  # Enable half precision for GPU
    enable_cudnn_benchmark: true  # Enable cuDNN benchmark
    enable_mixed_precision: true  # Enable mixed precision training
    lazy_load_optional: true  # Lazy load optional components
```

### Model Configuration

#### Supported Models
- **all-MiniLM-L6-v2**: Fast, general-purpose embeddings (default)
- **all-MiniLM-L12-v2**: Higher quality, larger model
- **multi-qa-MiniLM-L6-cos-v1**: Optimized for Q&A tasks
- **paraphrase-multilingual-MiniLM-L12-v2**: Multilingual support
- **sentence-transformers/all-mpnet-base-v2**: High-quality general embeddings

#### Performance Settings
- **Half Precision**: Enable FP16 for 2x speed improvement on modern GPUs
- **cuDNN Benchmark**: Optimize convolution operations
- **Mixed Precision**: Balance between speed and accuracy
- **Lazy Loading**: Load models in background for faster startup

##  API Endpoints

### Core Endpoints

#### Health Check
```bash
GET /api/v1/health
# Returns: Comprehensive health status and system information

GET /api/v1/status
# Returns: Basic service status information

GET /api/v1/metrics
# Returns: Processing metrics and statistics
```

#### Single Embedding Generation
```bash
POST /api/v1/embedding/single
Content-Type: application/json

{
  "text": "This is a sample text for embedding generation",
  "normalize": true
}

# Returns:
{
  "success": true,
  "text": "This is a sample text for embedding generation",
  "embedding": [0.1, -0.2, 0.3, 0.4, -0.5],
  "embedding_dimension": 384,
  "processing_time": 0.045,
  "normalized": true,
  "message": "Embedding generated successfully"
}
```

#### Batch Embedding Generation
```bash
POST /api/v1/embedding/batch
Content-Type: application/json

{
  "texts": [
    "First sample text",
    "Second sample text", 
    "Third sample text",
    "Fourth sample text"
  ],
  "batch_size": 16,
  "normalize": true
}

# Returns:
{
  "success": true,
  "total_texts": 4,
  "embeddings_generated": 4,
  "embedding_dimension": 384,
  "processing_time": 0.156,
  "batch_size": 16,
  "normalized": true,
  "embeddings": [
    [0.1, -0.2, 0.3, 0.4, -0.5],
    [0.2, -0.1, 0.4, 0.3, -0.6],
    [0.3, -0.3, 0.2, 0.5, -0.4],
    [0.4, -0.4, 0.1, 0.6, -0.3]
  ],
  "message": "Batch embeddings generated successfully"
}
```

#### Parallel Embedding Generation
```bash
POST /api/v1/embedding/parallel
Content-Type: application/json

{
  "texts": [f"Sample text number {i}" for i in range(1, 21)],
  "num_workers": 4,
  "batch_size": 16,
  "normalize": true
}

# Returns:
{
  "success": true,
  "total_texts": 20,
  "embeddings_generated": 20,
  "embedding_dimension": 384,
  "processing_time": 0.234,
  "num_workers": 4,
  "batch_size": 16,
  "normalized": true,
  "embeddings": [[0.1, -0.2, 0.3], [0.2, -0.1, 0.4]],
  "message": "Parallel embeddings generated successfully"
}
```

#### Model Information
```bash
GET /api/v1/model/info

# Returns:
{
  "success": true,
  "model_info": {
    "model_name": "all-MiniLM-L6-v2",
    "device": "cuda",
    "batch_size": 32,
    "max_seq_length": 512,
    "normalize_embeddings": true,
    "embedding_dimension": 384,
    "cuda_available": true,
    "cuda_device_count": 1
  },
  "message": "Model information retrieved successfully"
}
```

#### Cache Management
```bash
POST /api/v1/cache/clear

# Returns:
{
  "success": true,
  "message": "Cache cleared successfully",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🛠️ Advanced Usage

### Custom Model Configuration

1. **Update configuration**:

```python
# In config.yaml, modify the embedding_generation section
embedding_generation:
  model:
    name: "your-custom-model-name"
    max_seq_length: 512
    normalize_embeddings: true
```

2. **Environment variable overrides**:

```bash
export EMBEDDING_MODEL_NAME="your-custom-model-name"
export EMBEDDING_BATCH_SIZE=64
export ENABLE_HALF_PRECISION=true
```

### Batch Processing

```python
# Process multiple texts efficiently
from src.services.embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator(
    model_name="all-MiniLM-L6-v2",
    device="cuda",
    batch_size=32
)

# Generate embeddings for a list of texts
texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
embeddings = generator.generate_embeddings_batch(texts)

# Process with custom batch size
embeddings = generator.generate_embeddings_batch(texts, batch_size=16)
```

### Parallel Processing

```python
# Process large datasets with parallel workers
from src.services.embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator(
    model_name="all-MiniLM-L6-v2",
    device="cuda",
    batch_size=32
)

# Generate embeddings with parallel processing
texts = [f"Sample text {i}" for i in range(1000)]
embeddings = generator.generate_embeddings_parallel(
    texts, 
    num_workers=4, 
    batch_size=32
)
```

### Performance Optimization

```python
# Optimize for inference
generator = EmbeddingGenerator(
    model_name="all-MiniLM-L6-v2",
    device="cuda",
    batch_size=32
)

# Apply optimizations
generator.optimize_for_inference()

# Clear GPU cache when needed
generator.clear_cache()

# Get model information
model_info = generator.get_model_info()
print(f"Model: {model_info['model_name']}")
print(f"Device: {model_info['device']}")
print(f"Embedding dimension: {model_info['embedding_dimension']}")
```

## 📊 Monitoring & Metrics

### Built-in Metrics

The system tracks comprehensive metrics:

- **Processing Performance**: Response times, success rates, throughput
- **Model Performance**: Loading times, GPU utilization, memory usage
- **System Health**: Component status, resource usage, error rates
- **Quality Metrics**: Embedding dimensions, normalization status
- **Cache Management**: Cache hits, memory usage, cleanup efficiency

### Accessing Metrics

```bash
# Get current metrics
curl http://localhost:8000/api/v1/metrics

# Get detailed health status
curl http://localhost:8000/api/v1/health

# Basic service status
curl http://localhost:8000/api/v1/status
```

### Health Check Levels

1. **Basic Health Check** (`/api/v1/status`):
   - Service availability
   - Basic system information
   - Model status

2. **Comprehensive Health Check** (`/api/v1/health`):
   - Detailed system diagnostics
   - Model status and performance
   - Processing metrics and statistics
   - GPU memory usage and optimization status

## 🌍 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_NAME` | Application name | "Embedding Generation Service" | No |
| `APP_VERSION` | Application version | "1.0.0" | No |
| `APP_DEBUG` | Enable debug mode | false | No |
| `SERVER_HOST` | Server host address | 0.0.0.0 | No |
| `SERVER_PORT` | Server port | 8000 | No |
| `EMBEDDING_MODEL_NAME` | Embedding model name | "all-MiniLM-L6-v2" | No |
| `EMBEDDING_DEVICE` | Device to use | "auto" | No |
| `EMBEDDING_BATCH_SIZE` | Default batch size | 32 | No |
| `EMBEDDING_MAX_SEQ_LENGTH` | Max sequence length | 512 | No |
| `EMBEDDING_NORMALIZE` | Normalize embeddings | true | No |
| `ENABLE_HALF_PRECISION` | Enable FP16 precision | true | No |
| `ENABLE_CUDNN_BENCHMARK` | Enable cuDNN benchmark | true | No |
| `LAZY_LOAD_OPTIONAL` | Lazy load optional models | true | No |
| `LOG_LEVEL` | Logging level | INFO | No |

## 🚀 Production Deployment

### Performance Considerations

1. **Hardware Requirements**:
   - **GPU**: NVIDIA GPU with 4GB+ VRAM recommended
   - **CPU**: 2+ cores for parallel processing
   - **RAM**: 8GB+ for model loading and processing
   - **Storage**: SSD recommended for model caching

2. **Model Selection**:
   - Use **all-MiniLM-L6-v2** for fast, general-purpose embeddings
   - Use **all-MiniLM-L12-v2** for higher quality embeddings
   - Use **multi-qa-MiniLM-L6-cos-v1** for Q&A tasks
   - Enable **half precision** for 2x speed improvement on modern GPUs
   - Use **lazy loading** for faster startup times

3. **Scaling Options**:
   - **Horizontal**: Deploy multiple instances behind load balancer
   - **Vertical**: Increase GPU memory and CPU cores
   - **Serverless**: Use RunPod for auto-scaling

### Security Considerations

```bash
# Production environment variables
export APP_DEBUG=false
export LOG_LEVEL=WARNING
export CORS_ALLOW_ORIGINS=["https://yourdomain.com"]

# Configure proper CORS for production
# Update config.yaml CORS settings
```

### Health Monitoring

```bash
# Set up health check monitoring
curl -f http://localhost:8000/api/v1/status || exit 1

# Monitor detailed system status
curl http://localhost:8000/api/v1/health
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Issues**:
   - Check GPU memory availability
   - Verify model names and paths
   - Check internet connectivity for model downloads
   - Use lazy loading for faster startup

2. **Performance Issues**:
   - Enable half precision for faster inference
   - Increase batch size for better GPU utilization
   - Use lazy loading for faster startup
   - Monitor GPU memory usage
   - Enable cuDNN benchmarking

3. **Memory Issues**:
   - Reduce batch size
   - Enable memory optimization
   - Clear GPU cache periodically
   - Use smaller models if needed
   - Use `clear_cache()` method

4. **API Issues**:
   - Check CORS configuration
   - Verify request format and parameters
   - Monitor error logs for details
   - Test with smaller text inputs

### Debug Mode

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
export APP_DEBUG=true

# Run with verbose output
python app.py
```

### Performance Debugging

```bash
# Monitor metrics in real-time
watch -n 5 'curl -s http://localhost:8000/api/v1/metrics | jq .'

# Check detailed health status
curl http://localhost:8000/api/v1/health | jq .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass and code is formatted
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for high-performance web APIs
- Powered by [Sentence Transformers](https://www.sbert.net/) for embedding generation
- GPU acceleration enabled by [PyTorch](https://pytorch.org/) and CUDA
- Serverless deployment supported by [RunPod](https://runpod.io/)
- Configuration management with [Pydantic](https://pydantic.dev/) and [PyYAML](https://pyyaml.org/)

---