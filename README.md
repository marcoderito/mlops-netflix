
# MLOps Netflix Application

This repository contains a machine learning application for recommending Netflix shows based on user preferences. The application is built using Python and Flet for the frontend.

## Prerequisites

- Python 3.8 or higher
- Poetry for dependency management
- Git (optional, for cloning the repository)

## Installation

### Standard Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-repo/mlops-netflix.git
   cd mlops-netflix
   ```

2. **Install Dependencies with Poetry:**
   ```sh
   poetry install
   ```

3. **Activate the Virtual Environment:**
   ```sh
   poetry shell
   ```

4. **Run the Application:**
   ```sh
   flet --web -p 8080 ui.py
   ```

### Manual Installation (If Poetry Fails)

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-repo/mlops-netflix.git
   cd mlops-netflix
   ```

2. **Create a Virtual Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```

3. **Install Required Python Packages:**
   ```sh
   pip install -r requirements.txt
   ```

   If there is no `requirements.txt`, manually install the packages:
   ```sh
   pip install pandas flet scikit-learn faker cryptography
   ```

4. **Run the Application:**
   ```sh
   flet --web -p 8080 ui.py
   ```

## Files and Directories

- **`env.sh`**: Shell script for setting up environment variables.
- **`gallery.py`**: Handles the gallery view of movie recommendations.
- **`gallery-tests.py`**: Contains tests for the gallery module.
- **`like.py`**: Manages the like/dislike functionality for movie recommendations.
- **`login.py`**: Manages user authentication via OAuth (e.g., AWS Cognito).
- **`machine_learning_utils.py`**: Contains machine learning utilities, including data preprocessing, model training, and evaluation.
- **`movie_db_utils.py`**: Provides utility functions for interacting with the movie database.
- **`poetry.lock`**: Generated by Poetry, locks the project to specific versions of dependencies.
- **`pyproject.toml`**: Contains project metadata and dependencies configuration for Poetry.
- **`ui.py`**: The main entry point of the application, where the Flet app is initialized and run.

## Running Tests

To run tests, use the following command:
```sh
pytest
```

## Troubleshooting

If you encounter issues with missing dependencies or modules not being found, you may need to install them manually using pip, as described in the manual installation section.


