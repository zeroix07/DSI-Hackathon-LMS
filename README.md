# AI-Powered Learning Platform
## Platform Pembelajaran Data Engineer, Data Analyst & Data Scientist dengan AI

Platform pembelajaran interaktif yang menggunakan kecerdasan buatan untuk memberikan pengalaman belajar yang dipersonalisasi dalam bidang Data Engineering, Data Analyst dan Data Science.

## 🎯 Fitur Utama

- **Sistem Login** - Autentikasi pengguna yang aman
- **Pemilihan Peminatan** - Data Engineer atau Data Scientist
- **Assessment AI** - Evaluasi kemampuan otomatis dengan klasifikasi level (Pemula/Menengah/Mahir)
- **Personalized Learning** - Kurikulum yang disesuaikan berdasarkan hasil assessment
- **Materi Pembelajaran** - Konten yang dihasilkan dan disesuaikan oleh AI
- **Chatbot Interaktif** - Asisten AI untuk membantu memahami materi
- **Latihan & Quiz** - Soal yang dibuat otomatis oleh AI
- **Auto-Grading** - Koreksi otomatis dengan feedback yang detail
- **Laporan Pembelajaran** - Raport dan rekomendasi untuk langkah selanjutnya

## 🔄 Alur Pembelajaran

```
Login → Pilih Peminatan → Assessment AI → Personalized Learning Path 
  ↓
Materi Pembelajaran + Chatbot → Latihan & Quiz → Auto-Grading 
  ↓
Laporan & Rekomendasi
```

## 🚀 Instalasi dan Setup

### Prasyarat
- Python 3.8+
- Git

### 1. Clone Repository
```bash
git https://github.com/zeroix07/DSI-Hackathon-LMS.git
cd DSI-Hackathon-LMS
```

### 2. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Buat file `.env` di root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Menjalankan Aplikasi

#### Backend API
```bash
python api_main.py
```
API akan berjalan di: `http://localhost:8000/`


## 🛠️ Teknologi yang Digunakan

- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **AI/LLM**: Groq API
- **Search**: Tavily API
- **Language**: Python

## 📁 Struktur Project

```
ai-learning-platform/
├── api_main.py              # Backend API server
├── index.html               # Main Page
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── .gitignore              # Git ignore file
└── README.md               # Documentation
```

## 🔧 API Endpoints

- `GET /` - Health check
- `POST /assessment` - Melakukan assessment kemampuan
- `POST /generate-learning-path` - Generate personalized learning path
- `POST /chat` - Chatbot untuk bantuan pembelajaran
- `POST /generate-quiz` - Generate quiz dan latihan
- `POST /grade-submission` - Auto-grading jawaban user

## 🎓 Cara Penggunaan

1. **Login** ke platform
2. **Pilih peminatan** (Data Engineer/Data Analyst/Data Scientist)
3. **Ikuti assessment** untuk menentukan level kemampuan
4. **Pelajari materi** yang telah dipersonalisasi
5. **Gunakan chatbot** untuk bertanya tentang materi
6. **Kerjakan latihan dan quiz**
7. **Lihat hasil dan feedback** dari AI
8. **Ikuti rekomendasi** untuk pembelajaran selanjutnya

## 🔐 Keamanan

- Environment variables untuk API keys
- Validasi input pada semua endpoints
- 

## 📄 Lisensi

Project ini menggunakan lisensi MIT. Lihat file `LICENSE` untuk detail lengkap.

---

**Catatan**: Pastikan untuk tidak membagikan API keys Anda dan selalu gunakan environment variables untuk menyimpan credential yang sensitif.