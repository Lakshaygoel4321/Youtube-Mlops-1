# 🎥 YouTube Sentiment Analysis MLOps Project

An end-to-end MLOps pipeline designed to analyze YouTube video sentiment using state-of-the-art tools and a modern Chrome plugin extension. This project demonstrates the lifecycle of a machine learning model with robust MLOps practices, modular architecture, and an interactive front end built for seamless browser integration.

---

## 🌟 Key Features

✅ **Complete MLOps Pipeline:** From data ingestion to model training and prediction.
✅ **Chrome Extension Integration:** Custom extension for YouTube sentiment analysis.
✅ **Interactive Frontend:** Flask-powered web UI embedded in the extension using HTML, CSS, and JavaScript.
✅ **Modular Components:** Clear separation of concerns for data handling and model management.

---

## 🛠️ Tech Stack and Tools

* **Programming Language:** Python
* **Frameworks & Libraries:** Flask, Scikit-learn, Pandas, NumPy
* **Frontend:** HTML, CSS, JavaScript
* **Browser Integration:** Chrome Extension APIs
* **Version Control & Automation:** GitHub

---

## ⚙️ Architecture Overview

The project follows a modular, scalable MLOps workflow:

1. **Data Ingestion:**
   Collects YouTube video data (e.g., comments, metadata) and stores it in a structured format.
2. **Data Validation:**
   Ensures data quality and schema consistency before processing.
3. **Data Transformation:**
   Preprocesses text data (cleaning, tokenization, vectorization) ready for model training.
4. **Model Training:**
   Trains a sentiment classification model to predict positive or negative sentiments.
5. **Frontend Integration:**
   Flask backend serves the prediction API and Chrome Extension communicates with the Flask server.

> **Note:** This project focuses on model development and browser integration; model evaluation and pusher modules are not included.

---

## 📂 Project Structure

```
YouTube-Sentiment-Analysis/
├── src/
│   ├── pipeline/           # Training and prediction pipelines
│   ├── components/         # Data ingestion, validation, transformation
│   ├── entity/             # Configuration and artifact entities
│   └── app.py              # Flask backend API
├── chrome_extension/
│   ├── static/             # CSS, JavaScript assets for the extension UI
│   ├── templates/          # HTML templates
│   └── manifest.json       # Chrome extension configuration
├── notebook/               # Notebooks for EDA and experimentation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Google Chrome (for extension development)
* Git

### Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/YouTube-Sentiment-Analysis.git
cd YouTube-Sentiment-Analysis
```

Create a virtual environment and activate it:

```bash
conda create -n youtube-sentiment python=3.10 -y
conda activate youtube-sentiment
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🌐 Running the Flask Server

Start the backend API:

```bash
python app.py
```

By default, the Flask app runs at:

```
http://localhost:5000
```

---

## 🧩 Chrome Extension Setup

1. Open Chrome and navigate to:

   ```
   chrome://extensions
   ```
2. Enable **Developer mode** (top right).
3. Click **Load unpacked**.
4. Select the `chrome_extension/` folder.
5. The extension will appear in your toolbar.

---

## 🧪 Usage

* Navigate to a YouTube video page.
* Click the extension icon.
* The extension communicates with your Flask API to analyze sentiment and display results in the popup.

---

## 📈 Challenges and Learnings

**Challenges:**

* Integrating a local Flask server with a Chrome extension.
* Handling cross-origin requests securely.
* Designing a clean modular MLOps architecture without cloud deployment.

**Learnings:**

* Improved understanding of Chrome Extension APIs and frontend-backend communication.
* Best practices for structuring MLOps pipelines in production-ready environments.

---

## 🚀 Future Improvements

* Add model evaluation and performance tracking.
* Implement CI/CD for automatic retraining and deployment.
* Deploy the backend API to the cloud for broader access.
* Add model versioning and logging.

---

## 👨‍💻 Author

\[Your Name]

* [LinkedIn](https://www.linkedin.com/in/lakshay-goel-b10878326)
* [GitHub](https://github.com/Lakshaygoel4321)

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🎉 Acknowledgements

Thanks to the open-source community and all contributors to the libraries and frameworks used in this project.
