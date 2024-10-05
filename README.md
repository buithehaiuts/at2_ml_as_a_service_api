# api repo

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
project-root/
├── app/                      # Main application directory
│
├── backend/                  # Backend services
│   ├── Dockerfile            # Dockerfile for backend
│   ├── api.py                # API definitions
│   └── model.pkl             # Saved model in PKL format
│
├── frontend/                 # Frontend services
│   ├── app.py                # Streamlit application
│   └── Dockerfile            # Dockerfile for frontend
│
├── models/                   # Folder for saved models
│   └── model.pkl             # Additional saved model in PKL format
│
├── .gitignore                # Git ignore file to exclude unnecessary files
├── .gitkeep                  # Keep empty directories in git
├── Makefile                  # Makefile with convenience commands
├── README.md                 # Project documentation
├── __init__.py               # Initializes the package
├── github.txt                # GitHub-related documentation
├── render.txt                # Render-related text file
├── render.yaml               # Render configuration file
├── requirements.txt          # Project dependencies
└── setup.cfg                 # Configuration file for setup tools
