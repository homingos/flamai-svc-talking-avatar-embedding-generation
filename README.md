# FastAPI REST Template

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-blue.svg)
![Architecture](https://img.shields.io/badge/Architecture-Clean-orange.svg)
![Process Management](https://img.shields.io/badge/Process-Management-purple.svg)
![Configuration](https://img.shields.io/badge/Configuration-Advanced-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

A production-ready FastAPI REST template with comprehensive server management, process orchestration, and advanced configuration management. Built with clean architecture principles, featuring both stateless and stateful process management modes, comprehensive logging, and enterprise-grade configuration handling.

## ✨ Features

- 🔧 **Advanced Configuration**: YAML-based configuration with environment variable overrides and validation
- 📊 **Process Management**: Both stateful and stateless process management with automatic cleanup
- 🏗️ **Server Management**: Comprehensive server lifecycle with graceful shutdown and signal handling
- 📝 **Structured Logging**: Advanced logging with file rotation and configurable levels
- 🔒 **Security Ready**: JWT authentication, API key management, and CORS configuration
- 📁 **File Management**: Intelligent file tracking and cleanup with orphaned file detection
- 🎯 **Clean Architecture**: Modular design with clear separation of concerns
- ⚡ **Performance Monitoring**: Built-in metrics and health checks
- 🛠️ **Development Tools**: Comprehensive testing setup and development utilities

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd template-fastapi-rest

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Environment Configuration

Create a `.env` file with your configuration:

```bash
# Application Configuration
ENVIRONMENT=development
APP_DEBUG=true
APP_NAME="My FastAPI Application"

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_RELOAD=true

# API Keys (for external services)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
CUSTOM_API_KEY=your_custom_key

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key

# Logging Configuration
LOG_LEVEL=INFO
```

### 3. Configuration Management

The application uses a sophisticated configuration system with YAML files and environment variable overrides:

```yaml
# src/utils/config/config.yaml
app:
  name: "fastapi-rest-template"
  version: "1.0.0"
  description: "FastAPI AI Server with AI Processing"
  debug: false
  environment: "development"

server:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1

# Process Manager Configuration
process_manager:
  type: "stateful"  # stateful or stateless
  cleanup:
    enabled: true
    interval: 300  # 5 minutes
    process_ttl: 1800  # 30 minutes

# Server Manager Configuration
server_manager:
  app:
    name: "FastAPI AI Server"
    version: "1.0.0"
  housekeeping:
    enabled: true
    interval: 600  # 10 minutes
```

### 4. Run the Application

```bash
# Start the FastAPI server
python app.py

# Or using uv
uv run python app.py

# The server will start on http://localhost:8000
# Visit http://localhost:8000/docs for interactive API documentation
```

## 📁 Project Structure

```
template-fastapi-rest/
├── app.py                     # FastAPI application entry point
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── src/
│   ├── api/                  # API layer
│   │   ├── routes.py         # REST API endpoints
│   │   ├── handlers.py       # Request handlers
│   │   ├── models.py         # Data models
│   │   └── schema.py         # Pydantic schemas
│   ├── core/                 # Core application logic
│   │   ├── server_manager.py # Server lifecycle management
│   │   ├── process_manager.py # Process orchestration
│   │   └── managers.py       # Manager utilities
│   ├── services/             # Business logic services
│   │   ├── orchestrator.py   # Service orchestration
│   │   └── pipelines/        # Processing pipelines
│   └── utils/                # Utilities and helpers
│       ├── config/           # Configuration management
│       │   ├── config.yaml   # Default configuration
│       │   ├── settings.py   # Settings manager
│       │   └── modules/      # Configuration modules
│       ├── resources/        # Resource management
│       │   ├── logger.py     # Logging configuration
│       │   ├── downloader.py # Resource downloader
│       │   └── helper.py     # Helper utilities
│       ├── io/               # Input/Output utilities
│       │   ├── reader.py     # File readers
│       │   └── writer.py     # File writers
│       └── auth/             # Authentication utilities
├── runtime/                     # Application data directory
│   ├── temp/                 # Temporary files
│   ├── outputs/              # Output files
│   ├── logs/                 # Log files
│   ├── models/               # AI models
│   ├── uploads/              # Uploaded files
│   └── assets/               # Static assets
└── tests/                    # Test suite
    ├── unit/                 # Unit tests
    ├── integration/          # Integration tests
    └── e2e/                  # End-to-end tests
```

## ⚙️ Configuration

### Main Configuration

The application uses a hybrid configuration approach with YAML files and environment variables:

```yaml
# Application Settings
app:
  name: "fastapi-rest-template"
  version: "1.0.0"
  description: "FastAPI AI Server with AI Processing"
  debug: false
  environment: "development"
  temp_dir: "runtime/temp"
  assets_dir: "runtime/assets"

# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1
  timeout: 30

# Process Manager Configuration
process_manager:
  type: "stateful"  # stateful or stateless
  stateful:
    max_processes: 1000
    enable_memory_optimization: true
  stateless:
    cache_ttl: 300
    enable_persistence: false
  cleanup:
    enabled: true
    interval: 300
    process_ttl: 1800
    orphaned_files_ttl: 3600

# Server Manager Configuration
server_manager:
  app:
    name: "FastAPI AI Server"
    version: "1.0.0"
  directories:
    temp: "./runtime/temp"
    outputs: "./runtime/outputs"
    logs: "./runtime/logs"
  housekeeping:
    enabled: true
    interval: 600
  cleanup:
    force_on_startup: true
    force_on_shutdown: true
```

### Process Management Options

#### Stateful Process Manager
- **Memory-based**: Maintains process state in memory for fast access
- **Performance**: Optimized for single-instance applications
- **Cleanup**: Automatic cleanup of old processes and files

#### Stateless Process Manager
- **Distributed**: Suitable for distributed systems
- **Memory-efficient**: Minimal memory usage
- **Scalable**: Can be deployed across multiple instances

## 🛠️ Advanced Usage

### Custom Service Implementation

1. **Create your service class**:

```python
# src/services/my_service.py
from src.core.server_manager import AIService, ServiceConfig

class MyCustomService(AIService):
    def __init__(self, config: ServiceConfig, device: Optional[str] = None):
        super().__init__(config, device)
    
    async def initialize(self) -> bool:
        # Initialize your service
        self.is_initialized = True
        return True
    
    async def shutdown(self) -> None:
        # Cleanup your service
        self.is_shutting_down = True
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "initialized": self.is_initialized
        }
```

2. **Register your service**:

```python
# In your application startup
from src.core.server_manager import create_server_manager
from src.services.my_service import MyCustomService, ServiceConfig

server_manager = create_server_manager()

# Create and register service
service_config = ServiceConfig(
    name="my_service",
    enabled=True,
    config={"custom_param": "value"}
)
service = MyCustomService(service_config)
server_manager.register_service(service)
```

### Process Management

```python
# Create and manage processes
from src.core.process_manager import create_process_manager, ProcessState

process_manager = create_process_manager()

# Create a new process
process_id = process_manager.create_process(
    "image_processing",
    input_file="image.jpg",
    output_format="png"
)

# Update process status
process_manager.update_process(
    process_id,
    status=ProcessState.RUNNING,
    started_at=time.time()
)

# Track files for cleanup
process_manager.track_file(process_id, "/path/to/output.png")

# Complete the process
process_manager.update_process(
    process_id,
    status=ProcessState.COMPLETED,
    completed_at=time.time()
)
```

### Configuration Management

```python
# Access configuration
from src.utils.config.settings import settings

# Get configuration values
app_name = settings.get("app.name")
server_port = settings.get("server.port", 8000)

# Get section-specific configuration
server_config = settings.get_server_config()
process_config = settings.get_process_manager_config()

# Check environment
if settings.is_development():
    # Development-specific logic
    pass

# Print configuration summary
settings.print_config_summary()
```

## 📊 Monitoring & Metrics

### Built-in Metrics

The system tracks comprehensive metrics:

- **Process Metrics**: Created, completed, failed processes
- **File Management**: Files cleaned, orphaned files detected
- **Server Health**: Component status, resource usage
- **Performance**: Response times, cleanup efficiency

### Accessing Metrics

```python
# Get process manager metrics
from src.core.managers import get_process_manager

process_mgr = get_process_manager(request)
metrics = process_mgr.get_metrics()

# Get server manager status
from src.core.managers import get_server_manager

server_mgr = get_server_manager(request)
status = server_mgr.get_server_status()
```

## 🌍 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Application environment | production | No |
| `APP_DEBUG` | Enable debug mode | false | No |
| `APP_NAME` | Application name | fastapi-rest-template | No |
| `SERVER_HOST` | Server host address | 0.0.0.0 | No |
| `SERVER_PORT` | Server port | 8000 | No |
| `SERVER_RELOAD` | Enable auto-reload | false | No |
| `JWT_SECRET_KEY` | JWT secret key | - | Yes (for auth) |
| `OPENAI_API_KEY` | OpenAI API key | - | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | No |
| `LOG_LEVEL` | Logging level | INFO | No |

## 🚀 Production Deployment

### Performance Considerations

1. **Process Manager Selection**:
   - Use **Stateful** for single-instance deployments
   - Use **Stateless** for distributed/microservice architectures

2. **Resource Allocation**:
   - **CPU**: 2+ cores recommended for concurrent processing
   - **RAM**: 4GB+ for process management and caching
   - **Storage**: SSD recommended for file operations

3. **Scaling Options**:
   - **Horizontal**: Deploy multiple instances behind load balancer
   - **Vertical**: Increase resources for single instance
   - **Process Management**: Configure appropriate TTL and cleanup intervals

### Security Considerations

```bash
# Production environment variables
export ENVIRONMENT=production
export APP_DEBUG=false
export LOG_LEVEL=WARNING
export JWT_SECRET_KEY=your_secure_secret_key

# Configure CORS for production
# Update config.yaml CORS settings
```

### Health Monitoring

```bash
# Set up health check monitoring
curl -f http://localhost:8000/status || exit 1

# Monitor process metrics
curl http://localhost:8000/status | jq '.process_metrics'
```

## 🐛 Troubleshooting

### Common Issues

1. **Configuration Issues**:
   - Verify YAML syntax in config files
   - Check environment variable names and values
   - Ensure all required directories exist

2. **Process Management Issues**:
   - Check process TTL settings
   - Verify cleanup intervals
   - Monitor file tracking configuration

3. **Server Management Issues**:
   - Verify signal handling setup
   - Check housekeeping intervals
   - Monitor service initialization order

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
watch -n 5 'curl -s http://localhost:8000/status | jq .'

# Check process manager status
curl http://localhost:8000/status | jq '.process_metrics'
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src tests/
```

### Test Structure

```
tests/
├── conftest.py           # Test configuration
├── unit/                 # Unit tests
│   ├── test_config.py    # Configuration tests
│   ├── test_process_manager.py
│   └── test_server_manager.py
├── integration/          # Integration tests
│   ├── test_api.py       # API integration tests
│   └── test_services.py  # Service integration tests
└── e2e/                  # End-to-end tests
    └── test_workflows.py # Complete workflow tests
```

## 🤝 Contributing

We welcome contributions from the community! This project follows a comprehensive contribution process to ensure quality and maintainability.

### Quick Start for Contributors

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Setup Environment**: Follow the development setup in [CONTRIBUTING.md](.github/CONTRIBUTING.md)
3. **Create Branch**: Create a feature branch for your changes
4. **Make Changes**: Implement your changes with tests
5. **Submit PR**: Create a pull request following our guidelines

### Key Guidelines

- **Code of Conduct**: Please read and follow our [Code of Conduct](.github/CODE_OF_CONDUCT.md)
- **Security**: Report security vulnerabilities privately - see [Security Policy](.github/SECURITY.md)
- **Testing**: Maintain test coverage and follow testing guidelines
- **Documentation**: Update documentation for any API or configuration changes

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/
```

### Detailed Information

For comprehensive contribution guidelines, development setup, and community standards, please refer to:

- **[CONTRIBUTING.md](.github/CONTRIBUTING.md)** - Complete contribution guide with development setup, code standards, and PR process
- **[CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)** - Community standards and enforcement guidelines
- **[SECURITY.md](.github/SECURITY.md)** - Security policy and vulnerability reporting procedures

## 🔒 Security

### Security Policy

We take security seriously and have established comprehensive security measures:

- **Vulnerability Reporting**: Report security issues privately via email (see [Security Policy](.github/SECURITY.md))
- **Security Updates**: Regular dependency updates and security patches
- **Code Review**: All changes require code review and security assessment
- **Access Control**: Proper authentication and authorization mechanisms

### Security Best Practices

- Keep dependencies updated
- Use strong authentication methods
- Follow secure coding practices
- Never commit secrets or sensitive data
- Report vulnerabilities responsibly

For detailed security information, incident response procedures, and vulnerability reporting, see our [Security Policy](.github/SECURITY.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for high-performance web APIs
- Configuration management inspired by modern microservice patterns
- Process management designed for both monolithic and distributed architectures
- Logging and monitoring following enterprise best practices

---

## 📈 Roadmap

- [ ] Docker containerization for easy deployment
- [ ] Kubernetes deployment manifests
- [ ] Advanced metrics and monitoring dashboard
- [ ] Database integration examples
- [ ] Authentication and authorization examples
- [ ] WebSocket support
- [ ] GraphQL API support
- [ ] Advanced caching strategies
- [ ] Rate limiting implementation
- [ ] API versioning examples