# GDG Hackathon Project - AI-driven GDPR Compliance Checker

## Overview
This project is an AI-powered GDPR compliance checker that analyzes regulatory documents (PDFs/TXT files) to determine their adherence to GDPR standards. It utilizes a combination of FastAPI, Google Gemini AI, and machine learning techniques to extract, process, and validate compliance rules, providing structured feedback and recommendations.

## Features
- **FastAPI Backend:** Manages API requests, authentication, and processing.
- **Google Gemini AI Integration:** Generates GDPR summaries and compliance analysis.
- **Sentence Transformers:** Uses pre-trained models to encode and compare regulations.
- **Document Processing:** Extracts text from uploaded PDFs or TXT files.
- **Cosine Similarity Matching:** Identifies the closest GDPR rule based on similarity scores.
- **Flutter Frontend:** Displays compliance results in a user-friendly dashboard.

## Tech Stack
- **Backend:** FastAPI, Python, Google Gemini AI, Sentence Transformers, Scikit-learn
- **Frontend:** Flutter (Dart)
- **Storage:** NumPy (for embeddings), Joblib (for labels)
- **Security:** API key authentication

## How It Works
1. **User Uploads a Document:** A PDF or TXT file is uploaded through the Flutter app.
2. **Text Extraction:** The backend extracts and processes the document's text.
3. **Regulation Matching:** Each extracted regulation is compared against a precomputed GDPR dataset using cosine similarity.
4. **GDPR Summary:** Gemini AI provides a concise explanation of the matched GDPR rule.
5. **Compliance Analysis:** The system validates the match and suggests compliance improvements.
6. **Dashboard Display:** Results, scores, and analysis are displayed in the Flutter dashboard.

## Installation & Setup
### Backend (FastAPI)
#### Prerequisites:
- Python 3.8+
- pip
- Virtual environment (recommended)

#### Steps:
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/gdg-hackathon.git
   cd gdg-hackathon/backend
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set environment variables:
   ```sh
   export FAST_API_KEY='your_api_key'
   export GOOGLE_API_KEY='your_google_gemini_key'
   ```
5. Run the FastAPI server:
   ```sh
   uvicorn main:app --reload
   ```

### Frontend (Flutter)
#### Prerequisites:
- Flutter SDK
- Android Studio/Xcode (for mobile development)

#### Steps:
1. Navigate to the Flutter project:
   ```sh
   cd gdg-hackathon/frontend
   ```
2. Install dependencies:
   ```sh
   flutter pub get
   ```
3. Set up environment variables in `.env` file:
   ```sh
   FASTAPI_URL='http://your-fastapi-url'
   FAST_API_KEY='your_api_key'
   ```
4. Run the app:
   ```sh
   flutter run
   ```

## API Endpoints
### Upload and Process Document
- **Endpoint:** `POST /gdpr/upload`
- **Headers:** `access_token: your_api_key`
- **Body:** File (PDF/TXT)
- **Response:**
  ```json
  {
    "file": "uploaded_file.pdf",
    "compliance_results": [
      {
        "Regulation": "Sample Regulation Text",
        "Best Matched GDPR Rule": "GDPR Rule Name",
        "Similarity Score": 0.89,
        "GDPR Summary": "Short GDPR Explanation",
        "Gemini Analysis": "Detailed Compliance Analysis"
      }
    ]
  }
  ```

## Future Scope
- Support for additional regulations (e.g., CCPA, HIPAA).
- Multi-language support for GDPR compliance analysis.
- Advanced AI-driven recommendations for compliance improvements.

## Team
- **Pranav CM, Yuvan M** - Flutter Development, UI/UX Design  
- **Kishore L, Vijay R S** - Backend & AI, Research & Compliance

