# Resume Analyzer & Job Recommender

This project is a web application that analyzes resumes, predicts the candidate's job category and experience level, and recommends relevant job openings.

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI

## Deployment

This project is configured for deployment on Vercel.

### Vercel Deployment

1. **Push to GitHub:** Push your project to a GitHub repository.
2. **Import to Vercel:** Import the repository into your Vercel account.
3. **Configure Project:** Vercel will automatically detect the `vercel.json` file and configure the project.
4. **Deploy:** Click the "Deploy" button.

## Local Development

### Prerequisites

- Python 3.7+
- Node.js and npm

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Open `index.html` in your browser.**