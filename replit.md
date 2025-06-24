# AI Background Remover - Replit Guide

## Overview

This is a Streamlit-based web application that uses multiple AI algorithms to remove backgrounds from uploaded images. The application now integrates both local AI processing and the official remove.bg API service, providing users with multiple quality options from fast local processing to professional-grade API results.

## System Architecture

The application follows a simple client-server architecture:

- **Frontend**: Streamlit web interface providing image upload and processing capabilities
- **Backend**: Python-based processing using AI models for background removal
- **Runtime**: Python 3.11 with Nix package management
- **Deployment**: Autoscale deployment target on Replit

## Key Components

### 1. Main Application (app.py)
- **Purpose**: Core Streamlit application handling user interface and image processing
- **Key Features**:
  - Image upload with file validation (PNG, JPG, JPEG)
  - File size validation (max 10MB)
  - Two-column layout for before/after comparison
  - Real-time image processing feedback

### 2. Background Removal Engine
- **Technology**: Multiple AI algorithms including official remove.bg API integration
- **Local Processing**: OpenCV-based algorithms with GrabCut, edge detection, and advanced AI models
- **API Integration**: Official remove.bg service for professional-grade results
- **Processing**: Handles various image formats and sizes with multiple quality options

### 3. Configuration Management
- **Streamlit Config**: `.streamlit/config.toml` for server settings
- **Deployment Config**: `.replit` file for Replit-specific deployment
- **Dependencies**: `pyproject.toml` and `uv.lock` for Python package management

## Data Flow

1. **Image Upload**: User uploads image through Streamlit file uploader
2. **Validation**: System validates file type and size constraints
3. **Display**: Original image is displayed in the left column
4. **Processing**: User triggers background removal via button click
5. **AI Processing**: `rembg` library processes the image using AI models
6. **Output**: Processed image is displayed in the right column
7. **Download**: User can download the processed image

## External Dependencies

### Core Libraries
- **Streamlit (>=1.46.0)**: Web application framework
- **rembg**: AI background removal library (added dynamically)
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Numerical operations for image arrays

### System Dependencies (Nix packages)
- Image processing libraries: freetype, lcms2, libjpeg, libtiff, libwebp, openjpeg
- System libraries: libxcrypt, tcl, tk, zlib
- Image optimization: libimagequant

## Deployment Strategy

### Runtime Environment
- **Platform**: Replit with autoscale deployment
- **Python Version**: 3.11
- **Package Manager**: UV for fast dependency management
- **Nix Channel**: stable-24_05 for system packages

### Deployment Process
1. **Package Installation**: UV installs rembg during startup
2. **Server Launch**: Streamlit runs on port 5000
3. **Configuration**: Headless mode with public accessibility
4. **Scaling**: Autoscale deployment target for handling traffic

### Workflow Configuration
- **Run Button**: Triggers parallel workflow execution
- **Dynamic Dependencies**: Rembg is installed at runtime to optimize startup
- **Port Management**: Waits for port 5000 availability before serving

## Changelog
- June 23, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.