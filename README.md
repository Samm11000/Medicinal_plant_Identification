# 🌿 Medicinal Plant Identification App 🌿

This project is a Streamlit web application that identifies medicinal plants from leaf images. It utilizes a pre-trained Keras model to predict the species of the plant based on the uploaded or selected sample image. The application provides a user-friendly interface for image upload, prediction, and result display, making it easy for anyone to identify medicinal plants.

🚀 **Key Features**

*   **Image Upload:** Allows users to upload leaf images directly through the web interface. 📤
*   **Sample Image Selection:** Offers a selection of pre-loaded sample images for testing the application. 🖼️
*   **Pre-trained Keras Model:** Uses a pre-trained Keras model (`medicinal_plant_model.keras`) for accurate plant identification. 🧠
*   **Real-time Prediction:** Provides prediction results with confidence scores in real-time. ⏱️
*   **User-Friendly Interface:** Built with Streamlit for an intuitive and easy-to-use experience. 💻
*   **Error Handling:** Gracefully handles errors such as missing model files or data directories. ⚠️
*   **Session State Management:** Uses Streamlit's session state to persist variables across re-runs. 🔄
*   **Class Label Mapping:** Dynamically loads class labels from the data directory to interpret model predictions. 🏷️

🛠️ **Tech Stack**

| Category      | Technology             | Description                                                                 |
|---------------|------------------------|-----------------------------------------------------------------------------|
| Frontend      | Streamlit              | Python library for creating interactive web applications.                   |
| Backend       | Python                 | Programming language for the application logic.                             |
| AI Model      | TensorFlow/Keras       | Deep learning framework for building and loading the pre-trained model.     |
| Image Processing| OpenCV (cv2)           | Library for image processing tasks like resizing and color space conversion. |
| Data Handling | NumPy                  | Library for numerical operations, especially array manipulation.              |
| Image Handling| PIL (Pillow)           | Library for handling image loading from sample files.                       |
| File System   | os                     | Module for interacting with the file system.                                |

📦 **Getting Started**

### Prerequisites

*   Python 3.6+
*   Pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required packages:**

    ```bash
    pip install streamlit numpy opencv-python tensorflow pillow
    ```

### Running Locally

1.  **Navigate to the project directory:**

    ```bash
    cd <repository_directory>
    ```

2.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.

📂 **Project Structure**

```
├── app.py                      # Main Streamlit application file
├── Models/
│   ├── medicinal_plant_model.keras # Pre-trained Keras model (preferred)
│   └── medicinal_plant_model.h5    # Pre-trained Keras model (older format)
├── Database/
│   └── data/                   # Directory containing class labels
│       └── ...                 # Class label files
└── Sample_Test/                # Directory containing sample images
    └── ...                     # Sample image files
```

📸 **Screenshots**

![Uploading image.png…]()



📬 **Contact**

If you have any questions or suggestions, feel free to contact me at [shashavali8524@gmail.com](mailto:shashavali8524@gmail.com).

💖 **Thanks Message**

Thank you for checking out this project! I hope it's helpful for identifying medicinal plants. Your feedback and contributions are highly appreciated.

