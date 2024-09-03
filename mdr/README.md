
# MLOps Netflix Recommendation Simulation

This project contains code and data for simulating a machine learning-based recommendation system for Netflix shows. The recommendation system utilizes various machine learning techniques, including clustering and classification, to suggest shows based on user preferences.

## Files in the Project

1. **machinelearning_code.py**
   - This is the main Python script that orchestrates the machine learning workflow, including data preprocessing, model training, and user interaction. The script performs the following tasks:
     - Loads and preprocesses show data.
     - Clusters shows into profiles using KMeans.
     - Trains a Random Forest model for predicting user profiles based on their ratings.
     - Provides a user interface for rating shows and receiving recommendations.
     - Generates evaluation reports on model performance.

2. **netflix_titles.csv**
   - This CSV file contains the metadata for Netflix shows, including titles, genres, directors, cast, and more. The data is used for clustering shows and generating text features.

3. **ratings.csv**
   - This file stores user ratings for various shows. It is used to train the Random Forest model and determine user profiles.

4. **profiles.csv**
   - This CSV file associates each user with a specific profile based on their ratings. The profile is determined using a Random Forest model.

5. **predefined_reviews.csv**
   - This file contains predefined reviews for various shows, which are used to initialize the ratings dataset.

6. **clustered_netflix_titles.csv**
   - This CSV file stores the clustered Netflix titles after performing KMeans clustering on the show data. Each show is assigned a profile based on its cluster.

7. **popularity.json**
   - This JSON file keeps track of the popularity of shows based on user ratings. It is used to recommend popular shows to new users.

9. **Report_DeRito.tex**
   - This LaTeX file contains the report related to the project, possibly including the methodology, results, and discussion.

10. **DeRito_Presentation.tex**
    - This LaTeX file contains the presentation material related to the project.

11. **DeRito_Presentation.pdf**
    - This PDF file contains the presentation material related to the project.

12. **Report_DeRito.pdf**
    - This PDF file contains the report related to the project, possibly including the methodology, results, and discussion.

## How to Execute the Script

### Prerequisites

Ensure that you have Python installed on your system. You'll need to manually install the required Python packages for this project. Use the following command to install all the necessary packages:

### Manual Installation

Open your terminal or command prompt and run the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

These libraries are used for data manipulation, machine learning, and data visualization within the `machinelearning_code.py` script.

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **scikit-learn**: For machine learning models and tools.
- **matplotlib**: For creating static, animated, and interactive visualizations.
- **seaborn**: For statistical data visualization (built on top of matplotlib).
- **imbalanced-learn (imblearn)**: For handling imbalanced datasets (provides tools like SMOTE).

### Running the Script

To run the `machinelearning_code.py` script and test the machine learning simulation:

1. **Prepare the Environment:**
   - Ensure all required files (`netflix_titles.csv`, `ratings.csv`, `profiles.csv`, `predefined_reviews.csv`, `clustered_netflix_titles.csv`, `popularity.json`) are in the same directory as the script or update the file paths accordingly.

2. **Execute the Script:**
   - Run the script using Python:

   ```bash
   python machinelearning_code.py
   ```

3. **Follow the Prompts:**
   - The script will prompt you to enter your user ID and age. If you are a new user, it will guide you through rating a few shows. Based on your ratings, it will create or update your user profile and provide show recommendations.

4. **Review the Output:**
   - The script will generate and display recommendations based on your profile.
   - An evaluation report will be generated in PDF format, summarizing the performance of the machine learning models used.

---

This `README` should provide a clear understanding of how to navigate the project files and execute the primary script for the machine learning simulation.
