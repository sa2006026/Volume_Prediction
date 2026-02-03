# Project Folder Structure

This document describes the organized folder structure of the SAM Segmentation Flask Application.

## 📁 Directory Organization

### Root Level
```
Volume_Prediction/
├── Dockerfile*                    # Docker configuration files
├── docker-compose*.yml            # Docker Compose configurations
├── .dockerignore                  # Docker ignore patterns
├── requirements.txt               # Python dependencies
├── README.md                      # Main project documentation
└── FOLDER_STRUCTURE.md            # This file - structure documentation
```

### Source Code (`src/`)
```
src/
├── web/                           # Web application
│   └── sam_website.py            # Main Flask application (3,263 lines)
├── core/                          # Core analysis modules
│   ├── sam_analyzer.py           # SAM integration
│   ├── ring_width_analyzer.py    # Ring width detection
│   └── ...
├── analysis/                      # Analysis tools
├── matching/                      # Matching algorithms
└── utils/                         # Utility functions
```

### Scripts (`scripts/`)
```
scripts/
├── docker_start.sh                # Start Docker container
├── start_sam_cpu_mode.sh          # Start Flask in CPU mode
├── start_sam_with_gpu1.sh        # Start Flask on GPU 1
├── cloudflare/                    # Cloudflare tunnel scripts
│   ├── setup_cloudflare_tunnel.sh # Permanent tunnel setup
│   ├── start_cloudflare_tunnel.sh # Temporary tunnel
│   └── find_my_ip.sh             # IP address finder
├── analysis/                      # Analysis Python scripts
│   ├── analyze_droplets.py
│   ├── match_droplets.py
│   ├── visualize_results.py
│   ├── compare_bf_single_level.py
│   ├── find_unique_max_intensity_droplets.py
│   ├── match_fluorescent_to_bf.py
│   ├── summarize_bf_unique_diameters.py
│   └── ...
└── utils/                         # Utility scripts
    └── example_pre_segmentation_filter.py
```

### Documentation (`docs/`)
```
docs/
├── features/                      # Feature documentation
│   ├── ADMIN_MODE_FEATURE.md
│   ├── DOCKER_SETUP.md
│   ├── CSV_EXPORT_*.md
│   └── ...
├── deployment/                    # Deployment guides
│   ├── CLOUDFLARE_QUICK_START.md
│   ├── PERMANENT_URL_SETUP.md
│   └── ...
├── guides/                        # User guides
│   ├── WORKFLOW_USAGE.txt
│   ├── QUICK_REFERENCE.txt
│   └── ...
├── USER_MANUAL.md                 # User manual
└── README_DROPLET_ANALYSIS.md     # Droplet analysis guide
```

### Tests (`tests/`)
```
tests/
├── test_overlap_filter_modes.py
├── test_red_masks.py
└── ...
```

### Data Directories
```
├── data/                          # Input data (read-only in Docker)
├── model/                         # SAM model files
├── uploads/                       # User uploads (persistent)
├── results/                       # Segmentation results (persistent)
└── csv/                           # CSV data files
```

### Templates (`templates/`)
```
templates/
└── sam_website.html               # Main web interface template
```

## 🔄 Migration Summary

### Files Moved

**Documentation** → `docs/`
- All `.md` feature files → `docs/features/`
- All deployment guides → `docs/deployment/`
- All user guides → `docs/guides/`

**Scripts** → `scripts/`
- Shell scripts → `scripts/` or `scripts/cloudflare/`
- Analysis scripts → `scripts/analysis/`
- Utility scripts → `scripts/utils/`

**Tests** → `tests/`
- All `test_*.py` files → `tests/`

### Updated References

All script references in documentation have been updated to reflect new paths:
- `./docker_start.sh` → `./scripts/docker_start.sh`
- `./setup_cloudflare_tunnel.sh` → `./scripts/cloudflare/setup_cloudflare_tunnel.sh`
- etc.

### Script Path Updates

All scripts now use relative paths to find project root:
- Docker scripts find `docker-compose*.yml` files
- Flask start scripts find `src/web/sam_website.py`
- Cloudflare scripts provide correct paths in error messages

## 📝 Usage Examples

### Starting the Application

```bash
# Using Docker
./scripts/docker_start.sh gpu

# Direct Python (CPU)
./scripts/start_sam_cpu_mode.sh

# Direct Python (GPU 1)
./scripts/start_sam_with_gpu1.sh
```

### Cloudflare Tunnel

```bash
# Temporary URL
./scripts/cloudflare/start_cloudflare_tunnel.sh

# Permanent URL
./scripts/cloudflare/setup_cloudflare_tunnel.sh
```

### Running Analysis Scripts

```bash
# From project root
python3 scripts/analysis/analyze_droplets.py
python3 scripts/analysis/match_droplets.py
```

## 🎯 Benefits of New Structure

1. **Clear Organization**: Related files are grouped together
2. **Easy Navigation**: Find files quickly by category
3. **Scalability**: Easy to add new features without cluttering root
4. **Professional**: Follows standard project structure conventions
5. **Maintainability**: Easier to maintain and update

## 📌 Notes

- Docker files remain at root (standard practice)
- Configuration files remain at root for easy access
- All scripts use relative paths and work from any location
- Documentation is categorized for easy reference
