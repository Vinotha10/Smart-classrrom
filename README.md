# Smart Classroom Application

## Overview
The Smart Classroom application is designed to enhance the learning experience through the use of machine learning and data analytics. This project includes functionalities for data generation, model training, serving predictions, and visualizing results.

## Project Structure
```
smart-classroom
├── data
├── models
├── requirements.txt
├── data_generator.py
├── train_model.py
├── model_server.py
├── control.py
├── simulator_client.py
├── dashboard_app.py
└── utils.py
```

## Requirements
To install the necessary dependencies, run the following command:
```
pip install -r requirements.txt
```

## File Descriptions

- **requirements.txt**: Lists all the dependencies required for the project, including Flask, pandas, numpy, scikit-learn, joblib, tensorflow, streamlit, plotly, requests, and python-dotenv.

- **data_generator.py**: Contains functions for generating synthetic data for the smart classroom application. This includes data preprocessing and saving generated data to CSV files.

- **train_model.py**: Responsible for training machine learning models using the generated data. It includes functions for model selection, training, and evaluation.

- **model_server.py**: Serves the trained models as a web service using Flask to create endpoints for model predictions.

- **control.py**: Contains the logic for controlling the smart classroom application, managing user interactions and system responses.

- **simulator_client.py**: Simulates client interactions with the smart classroom application, sending requests to the model server and handling responses.

- **dashboard_app.py**: Creates a dashboard for visualizing data and model predictions, utilizing Streamlit or Plotly for the user interface.

- **utils.py**: Contains utility functions used across the application, such as data loading, preprocessing, and model evaluation metrics.

## Data and Models
- **data/**: Directory for storing generated CSV files.
- **models/**: Directory for saving trained models and scalers.

## Usage
1. Generate synthetic data using `data_generator.py`.
2. Train models with `train_model.py`.
3. Start the model server using `model_server.py`.
4. Interact with the application through `simulator_client.py` or visualize results using `dashboard_app.py`.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.