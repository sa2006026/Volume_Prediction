# SAM Segmentation Flask Application

A web-based application for image segmentation using Meta's Segment Anything Model (SAM), with features for droplet analysis, ring width detection, and CSV data export.

## 📁 Project Structure

```
Volume_Prediction/
├── src/                          # Source code
│   ├── web/                      # Web application
│   │   └── sam_website.py       # Main Flask application
│   ├── core/                     # Core analysis modules
│   │   ├── sam_analyzer.py       # SAM integration
│   │   ├── ring_width_analyzer.py
│   │   └── ...
│   ├── analysis/                 # Analysis tools
│   └── utils/                    # Utility functions
│
├── templates/                     # HTML templates
│   └── sam_website.html          # Main web interface
│
├── scripts/                       # Executable scripts
│   ├── docker_start.sh           # Start Docker container
│   ├── start_sam_cpu_mode.sh     # Start Flask in CPU mode
│   ├── start_sam_with_gpu1.sh    # Start Flask on GPU 1
│   ├── cloudflare/                # Cloudflare tunnel scripts
│   │   ├── setup_cloudflare_tunnel.sh
│   │   ├── start_cloudflare_tunnel.sh
│   │   └── find_my_ip.sh
│   ├── analysis/                  # Analysis scripts
│   │   ├── analyze_droplets.py
│   │   ├── match_droplets.py
│   │   └── ...
│   └── utils/                     # Utility scripts
│
├── docs/                          # Documentation
│   ├── features/                  # Feature documentation
│   ├── deployment/                # Deployment guides
│   └── guides/                    # User guides
│
├── tests/                         # Test files
│   ├── test_overlap_filter_modes.py
│   └── ...
│
├── docker/                        # Docker-related files (optional)
├── data/                          # Input data (read-only in Docker)
├── model/                         # SAM model files
├── uploads/                       # User uploads (persistent)
├── results/                       # Segmentation results (persistent)
│
├── Dockerfile                     # GPU Docker image
├── Dockerfile.cpu                 # CPU Docker image
├── docker-compose.yml             # Default Docker Compose
├── docker-compose.gpu.yml         # GPU Docker Compose
├── docker-compose.cpu.yml         # CPU Docker Compose
├── .dockerignore                  # Docker ignore patterns
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Start with GPU support
./scripts/docker_start.sh gpu

# Or start with CPU only
./scripts/docker_start.sh cpu
```

Access the application at: `http://localhost:5013`

### Option 2: Direct Python

```bash
# GPU mode (default)
python3 src/web/sam_website.py

# CPU mode
./scripts/start_sam_cpu_mode.sh

# Use specific GPU
./scripts/start_sam_with_gpu1.sh
```

## 🌐 Public Access (Cloudflare Tunnel)

### Temporary URL (Testing)
```bash
./scripts/cloudflare/start_cloudflare_tunnel.sh
```

### Permanent URL (Production)
```bash
./scripts/cloudflare/setup_cloudflare_tunnel.sh
```

See `docs/deployment/` for detailed setup instructions.

## 📖 Documentation

- **Features**: `docs/features/` - Feature documentation and guides
- **Deployment**: `docs/deployment/` - Docker and Cloudflare setup
- **User Guides**: `docs/guides/` - Usage guides and workflows

## 🔧 Key Features

- **SAM Segmentation**: Interactive image segmentation using Meta's SAM
- **Admin Mode**: Toggle advanced configuration options
- **Ring Width Detection**: Automatic dark edge detection
- **CSV Export**: Export segmentation data with diameter predictions
- **CSV Matching**: Match two CSV files by coordinates
- **Multiple Filters**: Pre-segmentation, intensity, circularity, overlap filters
- **GPU/CPU Support**: Automatic fallback to CPU if GPU unavailable

## 🐳 Docker

The application is containerized for easy deployment:

- **GPU Version**: Uses PyTorch with CUDA support
- **CPU Version**: CPU-only for systems without GPU
- **Volume Mounts**: Data persists outside containers
- **Auto-restart**: Containers restart automatically

See `docs/features/DOCKER_SETUP.md` for detailed Docker documentation.

## 📊 Analysis Scripts

Analysis and utility scripts are located in `scripts/analysis/`:

- `analyze_droplets.py` - Droplet analysis
- `match_droplets.py` - Match droplets between images
- `visualize_results.py` - Visualize segmentation results
- And more...

## 🧪 Testing

Test files are in `tests/`:

```bash
python3 tests/test_overlap_filter_modes.py
python3 tests/test_red_masks.py
```

## 📦 Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- Flask >= 2.0.0
- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- segment-anything >= 1.0

## 🔐 Permissions

If you encounter Docker permission issues:

```bash
sudo usermod -aG docker $USER
# Then log out and back in
```

See `docs/features/DOCKER_PERMISSIONS_FIX.md` for details.

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]
