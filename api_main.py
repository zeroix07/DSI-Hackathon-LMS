from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tavily import TavilyClient
import json
import uuid
from datetime import datetime
import uvicorn
import uuid
from dotenv import load_dotenv
import re

# Load API keys from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Education System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Hugging Face model
MODEL_NAME = "Sahabat-AI/Llama-Sahabat-AI-v2-70B-IT"

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None

def initialize_model():
    """Initialize the Hugging Face model and tokenizer"""
    global tokenizer, model, device
    
    try:
        print("Initializing Hugging Face model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=HUGGINGFACE_API_KEY,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HUGGINGFACE_API_KEY,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if not torch.cuda.is_available():
            model = model.to(device)
        
        model.eval()
        print("Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Falling back to API-based approach...")
        return False

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# Pydantic models
class User(BaseModel):
    username: str
    specialization: str
    level: Optional[str] = None

class AssessmentAnswer(BaseModel):
    question_id: int
    answer: str

class AssessmentSubmission(BaseModel):
    user_id: str
    answers: List[AssessmentAnswer]

class QuizSubmission(BaseModel):
    user_id: str
    material_id: str
    answers: List[Dict[str, str]]
    coding_answer: Optional[str] = None

class ChatMessage(BaseModel):
    user_id: str
    material_id: str
    message: str

# In-memory storage
users_db = {}
materials_db = {}
assessments_db = {}
quiz_results_db = {}

class EducationAgent:
    def __init__(self, tavily_client):
        self.tavily = tavily_client
        self.model_name = MODEL_NAME
        
    def generate_text(self, prompt: str, max_length: int = 2048, temperature: float = 0.1) -> str:
        """Generate text using the Hugging Face model (replaces groq.chat.completions.create)"""
        global tokenizer, model, device
        
        try:
            if model is None or tokenizer is None:
                return self._generate_with_api(prompt, max_length, temperature)
            
            # Format as chat message
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"Error generating text with local model: {e}")
            return self._generate_with_api(prompt, max_length, temperature)
    
    def _generate_with_api(self, prompt: str, max_length: int = 2048, temperature: float = 0.1) -> str:
        """Fallback: Generate text using Hugging Face Inference API"""
        try:
            import requests
            
            headers = {"Content-Type": "application/json"}
            if HUGGINGFACE_API_KEY:
                headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
            
            api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                return str(result).strip()
            else:
                raise Exception(f"API request failed: {response.status_code}")
                
        except Exception as e:
            print(f"Error with API fallback: {e}")
            raise Exception("Both local model and API failed")

    def generate_assessment_questions(self, specialization: str) -> Dict:
        """Generate assessment questions - SAME LOGIC, just replace groq call"""
        
        spec_topics = {
            "data_engineer": {
                "name": "Data Engineering",
                "key_areas": ["ETL/ELT", "Data Warehousing", "Big Data Tools", "Cloud Platforms", "Data Pipeline", "SQL", "Python/Scala", "Apache Kafka", "Docker", "Data Modeling"]
            },
            "data_scientist": {
                "name": "Data Science", 
                "key_areas": ["Statistics", "Machine Learning", "Python/R", "Data Visualization", "Feature Engineering", "Model Evaluation", "Deep Learning", "Business Intelligence", "A/B Testing", "Data Analysis"]
            },
            "data_analyst": {
                "name": "Data Analysis",
                "key_areas": ["SQL Fundamentals", "Data Visualization and Dashboards", "Business Intelligence Tools", "Statistical Analysis for Business","Excel Advanced Functions", "Data Cleaning and Preparation", "KPI and Metrics Design", "Report Writing and Presentation", "A/B Testing and Experimentation", "Business Process Analysis"]
            }
        }
        
        spec_info = spec_topics[specialization]
        
        prompt = f"""
        Buatkan 10 soal assessment untuk menentukan level kemampuan seseorang dalam {spec_info['name']}.
        
        Aturan:
        - 3-4 soal untuk level PEMULA (konsep dasar, definisi, tool basic)
        - 3-4 soal untuk level MENENGAH (implementasi, best practices, troubleshooting)
        - 3-4 soal untuk level MAHIR (optimisasi, advanced concepts, architecture)
        - Soal harus mencakup area: {', '.join(spec_info['key_areas'])}
        - Setiap soal harus memiliki 4 pilihan jawaban
        - Berikan jawaban yang benar untuk setiap soal
        
        PENTING: 
        - Jawab HANYA dengan format JSON yang valid tanpa penjelasan atau teks tambahan
        - Gunakan difficulty level: "pemula", "menengah", "mahir" (WAJIB gunakan kata ini)
        - Jangan gunakan "beginner", "intermediate", "advanced"
    
        Format JSON yang wajib diikuti:
        {{
            "questions": [
                {{
                    "id": 1,
                    "question": "Apa yang dimaksud dengan ETL dalam data engineering?",
                    "options": [
                        "Extract, Transform, Load - proses mengambil data dari sumber, mentransformasi, dan memuat ke tujuan",
                        "Export, Transfer, Link - proses ekspor dan transfer data",
                        "Evaluate, Test, Launch - proses evaluasi sistem data",
                        "Execute, Terminate, Log - proses eksekusi pipeline"
                    ],
                    "correct_answer": 0,
                    "difficulty": "pemula",
                    "topic": "ETL/ELT"
                }}
            ]
        }}
        
        Pastikan semua field ada dan difficulty hanya menggunakan: "pemula", "menengah", atau "mahir".
        """
        
        try:
            # REPLACED: self.groq.chat.completions.create() with self.generate_text()
            response_content = self.generate_text(prompt, max_length=1500, temperature=0.1)
            
            print(f"Raw response: {response_content}")
            
            # Extract JSON from response - SAME LOGIC
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_content = response_content[start_idx:end_idx]
                result = json.loads(json_content)
                
                # Validate and normalize difficulty levels - SAME LOGIC
                for question in result.get('questions', []):
                    difficulty = question.get('difficulty', '').lower()
                    if difficulty in ['beginner', 'basic', 'dasar']:
                        question['difficulty'] = 'pemula'
                    elif difficulty in ['intermediate', 'medium', 'tengah']:
                        question['difficulty'] = 'menengah' 
                    elif difficulty in ['advanced', 'expert', 'lanjut', 'lanjutan']:
                        question['difficulty'] = 'mahir'
                    elif difficulty not in ['pemula', 'menengah', 'mahir']:
                        question['difficulty'] = 'pemula'
                
                return result
            else:
                raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Content that failed to parse: {json_content}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON response from AI: {str(e)}")
        except Exception as e:
            print(f"Error generating assessment: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate assessment questions")
        
    def evaluate_assessment(self, answers: List[AssessmentAnswer], questions: List[Dict]) -> Dict:
        """Evaluate assessment - EXACT SAME LOGIC"""
        
        # Calculate score by difficulty
        score_by_level = {"pemula": 0, "menengah": 0, "mahir": 0}
        total_by_level = {"pemula": 0, "menengah": 0, "mahir": 0}
        
        for question in questions:
            difficulty = question.get("difficulty", "pemula")
            total_by_level[difficulty] += 1
            
            # Find user's answer
            user_answer = next((a.answer for a in answers if a.question_id == question["id"]), None)
            
            if user_answer is not None:
                try:
                    user_answer_idx = int(user_answer)
                    if user_answer_idx == question["correct_answer"]:
                        score_by_level[difficulty] += 1
                except:
                    pass
        
        # Calculate percentages
        pemula_score = score_by_level["pemula"] / max(total_by_level["pemula"], 1) * 100
        menengah_score = score_by_level["menengah"] / max(total_by_level["menengah"], 1) * 100
        mahir_score = score_by_level["mahir"] / max(total_by_level["mahir"], 1) * 100
        
        total_score = sum(score_by_level.values()) / sum(total_by_level.values()) * 100
        
        # Determine level based on performance
        if mahir_score >= 70 and menengah_score >= 80 and pemula_score >= 90:
            level = "mahir"
        elif menengah_score >= 70 and pemula_score >= 80:
            level = "menengah"
        else:
            level = "pemula"
        
        return {
            "level": level,
            "total_score": round(total_score, 1),
            "scores_by_difficulty": {
                "pemula": round(pemula_score, 1),
                "menengah": round(menengah_score, 1),
                "mahir": round(mahir_score, 1)
            },
            "recommendation": self._get_level_recommendation(level, total_score)
        }
    
    def _get_level_recommendation(self, level: str, score: float) -> str:
        """Helper method - SAME LOGIC"""
        recommendations = {
            "pemula": f"Anda berada di level pemula dengan skor {score}%. Fokus pada konsep dasar dan fundamental.",
            "menengah": f"Anda berada di level menengah dengan skor {score}%. Perdalam implementasi dan best practices.",
            "mahir": f"Anda berada di level mahir dengan skor {score}%. Fokus pada optimisasi dan arsitektur tingkat lanjut."
        }
        return recommendations.get(level, "Terus belajar dan berkembang!")
        
    def search_and_create_personalized_material(self, specialization: str, level: str, user_id: str) -> Dict:
        """Create personalized learning material - SAME LOGIC, just replace groq call"""
        
        try:
            print(f"Starting material creation for user: {user_id}, spec: {specialization}, level: {level}")
            
            # Define learning path - SAME LOGIC
            learning_paths = {
                "data_engineer": {
                    "pemula": ["Dasar SQL", "Python untuk Data", "ETL Dasar", "Dasar Pemodelan Data"],
                    "menengah": ["ETL Lanjutan", "Data Warehousing", "Apache Kafka", "Layanan Data Cloud"],
                    "mahir": ["Streaming Data Real-time", "Optimasi Pipeline Data", "Sistem Terdistribusi", "Tata Kelola Data"]
                },
                "data_scientist": {
                    "pemula": ["Dasar Statistik", "Python untuk Data Science", "Visualisasi Data", "Machine Learning Dasar"],
                    "menengah": ["Algoritma ML Lanjutan", "Feature Engineering", "Evaluasi Model", "Dasar Deep Learning"],
                    "mahir": ["Deep Learning Lanjutan", "MLOps", "Analisis Time Series", "Statistik Lanjutan"]
                },
                "data_analyst": {
                    "pemula": ["Dasar SQL", "Keahlian Excel", "Visualisasi Dasar", "Dasar Pembersihan Data"],
                    "menengah": ["Tools BI Lanjutan", "Analisis Statistik", "Desain Dashboard", "Business Intelligence"],
                    "mahir": ["Analitik Lanjutan", "Pemodelan Prediktif", "Strategi Bisnis", "Pengambilan Keputusan Berbasis Data"]
                }
            }
            
            if specialization not in learning_paths:
                raise ValueError(f"Invalid specialization: {specialization}")
            
            if level not in learning_paths[specialization]:
                raise ValueError(f"Invalid level: {level} for specialization: {specialization}")
            
            topics = learning_paths[specialization][level]
            current_topic = topics[0]
            
            print(f"Current topic: {current_topic}")
            
            content_sources = []
            
            spec_name = "Data Engineering" if specialization == "data_engineer" else "Data Science" if specialization == "data_scientist" else "Data Analyst" if specialization == "data_analysis" else "umum"

            prompt = f"""
                    kamu adalah seorang pembuat materi profesional {current_topic} untuk {spec_name} di level {level} yang telah berkecimpung selama 10 tahun
            
                    Buatlah materi pembelajaran dalam format JSON untuk topik {spec_name}: {current_topic} di level {level}.
    
                    Kembalikan hanya JSON yang valid dengan struktur ini:
                    {{
                        "title": "{current_topic} untuk {spec_name}",
                        "subtitle": "Pelajari {current_topic} langkah demi langkah",
                        "introduction": "Pengenalan terhadap {current_topic}",
                        "learning_objectives": [
                            "Memahami konsep {current_topic}",
                            "Menerapkan {current_topic} dalam praktik"
                        ],
                        "prerequisites": [
                            "Pengetahuan dasar pemrograman"
                        ],
                        "content": {{
                            "theory": {{
                                "overview": "Gambaran umum tentang {current_topic}",
                                "key_concepts": [
                                    "Konsep kunci 1",
                                    "Konsep kunci 2"
                                ]
                            }},
                            "practical_examples": [
                                {{
                                    "title": "Contoh Dasar",
                                    "description": "Contoh sederhana dari {current_topic}",
                                    "code_snippet": "# Contoh kode di sini",
                                    "explanation": "Contoh ini menunjukkan penggunaan dasar"
                                }}
                            ],
                            "best_practices": [
                                "Ikuti standar industri",
                                "Jaga kode tetap bersih dan terdokumentasi"
                            ]
                        }},
                        "estimated_duration": "2-3 jam"
                    }}
                    
                    Kembalikan HANYA JSON, tanpa teks lain."""
            
            print("Sending request to AI...")
            
            try:
                # REPLACED: self.groq.chat.completions.create() with self.generate_text()
                response_content = self.generate_text(prompt, max_length=1500, temperature=0.1)
                
                if not response_content:
                    raise ValueError("Empty content from AI response")
                        
                print(f"AI response length: {len(response_content)}")
                print(f"AI response first 100 chars: {response_content[:100]}")
                
            except Exception as ai_error:
                print(f"AI generation error: {ai_error}")
                return self._create_fallback_material(specialization, level, current_topic, topics, user_id)
            
            # Clean up the content - SAME LOGIC
            if response_content.startswith('```json'):
                response_content = response_content[7:]
                if response_content.endswith('```'):
                    response_content = response_content[:-3]
            elif response_content.startswith('```'):
                response_content = response_content[3:]
                if response_content.endswith('```'):
                    response_content = response_content[:-3]
            
            response_content = response_content.strip()
            
            if not response_content:
                print("Content is empty after cleanup")
                return self._create_fallback_material(specialization, level, current_topic, topics, user_id)
            
            print("Attempting to parse JSON...")
            
            try:
                material = json.loads(response_content)
                print("JSON parsing successful")
            except json.JSONDecodeError as json_error:
                print(f"JSON parsing error: {json_error}")
                print(f"Content that failed to parse: {repr(response_content)}")
                return self._create_fallback_material(specialization, level, current_topic, topics, user_id)
            
            # Add metadata - SAME LOGIC
            material.update({
                "id": str(uuid.uuid4()),
                "specialization": specialization,
                "level": level,
                "user_id": user_id,
                "sources": content_sources,
                "created_at": datetime.now().isoformat(),
                "current_topic_index": 0,
                "total_topics": len(topics),
                "learning_path": topics
            })
            
            print(f"Material created successfully with ID: {material['id']}")
            return material
            
        except Exception as e:
            print(f"Unexpected error in material creation: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            try:
                return self._create_fallback_material(specialization, level, topics[0] if 'topics' in locals() else "Topik Dasar", topics if 'topics' in locals() else ["Topik Dasar"], user_id)
            except:
                raise HTTPException(status_code=500, detail=f"Gagal membuat materi pembelajaran: {str(e)}")
    
    def _create_fallback_material(self, specialization: str, level: str, current_topic: str, topics: list, user_id: str) -> Dict:
        """Create a fallback material - SAME LOGIC"""
        import uuid
        from datetime import datetime
        
        spec_name = "Data Engineering" if specialization == "data_engineer" else "Data Science" if specialization == "data_scientist" else "Data Analyst" if specialization == "data_analyst" else "Umum"

        material = {
            "title": f"{current_topic} - {spec_name}",
            "subtitle": f"Materi Pembelajaran Level {level.title()}",
            "introduction": f"Selamat datang di materi pembelajaran {current_topic}. Panduan komprehensif ini akan membantu Anda menguasai dasar-dasar dan mengembangkan keterampilan Anda dalam {spec_name}.",
            "learning_objectives": [
                f"Memahami konsep inti dari {current_topic}",
                f"Menerapkan teknik {current_topic} dalam skenario dunia nyata",
                "Membangun keterampilan praktis melalui latihan langsung",
                "Mengembangkan praktik terbaik untuk pekerjaan profesional"
            ],
            "prerequisites": [
                "Pengetahuan dasar pemrograman",
                "Familiaritas dengan konsep data",
                "Kemauan untuk belajar dan berlatih"
            ],
            "content": {
                "theory": {
                    "overview": f"{current_topic} adalah konsep fundamental dalam {spec_name}. Bagian ini mencakup teori dan prinsip esensial yang perlu Anda pahami.",
                    "key_concepts": [
                        f"Pengenalan terhadap {current_topic}",
                        "Prinsip inti dan metodologi",
                        "Standar industri dan praktik terbaik",
                        "Kasus penggunaan umum dan aplikasi"
                    ]
                },
                "practical_examples": [
                    {
                        "title": f"Implementasi Dasar {current_topic}",
                        "description": f"Contoh praktis yang mendemonstrasikan konsep {current_topic}",
                        "code_snippet": f"# Contoh implementasi {current_topic}\n# Ini adalah contoh dasar untuk memulai\n\nprint('Belajar {current_topic}')",
                        "explanation": f"Contoh ini mendemonstrasikan implementasi dasar dari {current_topic}. Mulai dengan konsep sederhana dan secara bertahap membangun kompleksitas."
                    }
                ],
                "best_practices": [
                    "Ikuti konvensi standar industri",
                    "Tulis kode yang bersih, mudah dibaca, dan dapat dipelihara",
                    "Dokumentasikan pekerjaan Anda dengan menyeluruh",
                    "Uji implementasi Anda secara teratur",
                    "Tetap update dengan tren dan alat terbaru"
                ]
            },
            "estimated_duration": "2-4 jam",
            "id": str(uuid.uuid4()),
            "specialization": specialization,
            "level": level,
            "user_id": user_id,
            "sources": [],
            "created_at": datetime.now().isoformat(),
            "current_topic_index": 0,
            "total_topics": len(topics),
            "learning_path": topics,
            "is_fallback": True
        }
        
        print(f"Created fallback material with ID: {material['id']}")
        return material
    
    def generate_adaptive_quiz(self, material: Dict) -> Dict:
        """Generate adaptive quiz - SAME LOGIC, just replace groq call"""
        
        try:
            title = material.get('title', 'Unknown Topic')
            raw_level = material.get('level', 'pemula')
            level = self._validate_and_normalize_level(raw_level)
            specialization = material.get('specialization', 'general')
            difficulty_settings = self._get_difficulty_settings(level)
            
            content = json.dumps(material.get('content', {}), indent=2)[:2000]
            
            print(f"Generating quiz for: {title}, Level: {level}, Specialization: {specialization}")
            
            prompt = f"""
            Berdasarkan materi pembelajaran "{title}" untuk level {level} dalam bidang {specialization}, buatkan quiz yang komprehensif.
            
            Konten materi: {content}
            
            Buatkan quiz dalam format JSON yang valid. HANYA RETURN JSON, tidak ada teks lain:
            {{
                "multiple_choice": [
                    {{
                        "id": "mc_1",
                        "question": "Pertanyaan pilihan ganda yang menguji pemahaman konsep dari materi",
                        "options": [
                            "Pilihan A yang masuk akal dan relevan",
                            "Pilihan B yang masuk akal dan relevan", 
                            "Pilihan C yang masuk akal dan relevan",
                            "Pilihan D yang masuk akal dan relevan"
                        ],
                        "correct_answer": 0,
                        "explanation": "Penjelasan mengapa jawaban ini benar dan mengapa pilihan lain salah",
                        "difficulty": "{level}",
                        "topic": "topik spesifik dari materi yang diuji"
                    }}
                ],
                "practical_questions": [
                    {{
                        "id": "pq_1", 
                        "question": "Pertanyaan praktis yang menguji aplikasi konsep dari materi",
                        "scenario": "Skenario nyata yang relevan dengan materi",
                        "options": [
                            "Solusi A yang praktis",
                            "Solusi B yang praktis",
                            "Solusi C yang praktis", 
                            "Solusi D yang praktis"
                        ],
                        "correct_answer": 0,
                        "explanation": "Penjelasan solusi terbaik berdasarkan materi"
                    }}
                ],
                "coding_challenge": {{
                    "id": "code_1",
                    "question": "Implementasikan konsep dari materi yang telah dipelajari",
                    "problem_description": "Deskripsi masalah yang berkaitan langsung dengan materi {title}",
                    "requirements": [
                        "Requirement yang sesuai dengan level {level}",
                        "Requirement yang dapat diukur dan specific"
                    ],
                    "input_format": "Format input yang sesuai dengan konteks materi",
                    "output_format": "Format output yang diharapkan",
                    "sample_input": "Contoh input yang relevan",
                    "sample_output": "Contoh output yang sesuai",
                    "test_cases": [
                        {{
                            "input": "test input sesuai materi",
                            "expected_output": "expected output yang benar"
                        }}
                    ],
                    "hints": [
                        "Hint yang membantu berdasarkan konsep dari materi",
                        "Hint untuk pendekatan yang sesuai level {level}"
                    ],
                    "solution_approach": "Pendekatan solusi berdasarkan best practices dari materi"
                }}
            }}
            
            Aturan:
            - Buat {difficulty_settings['mc_count'][0]}-{difficulty_settings['mc_count'][1]} soal pilihan ganda
            - Buat {difficulty_settings['pq_count'][0]}-{difficulty_settings['pq_count'][1]} pertanyaan praktis
            - Buat 1 coding challenge yang relevan dengan materi
            - Semua soal HARUS berkaitan langsung dengan konten materi yang diberikan
            - Pertanyaan harus menguji pemahaman mendalam, bukan hanya hafalan
            - Sesuaikan tingkat kesulitan dengan level {level}
            - Gunakan konteks dari learning objectives dan content materi
            - RESPONSE HARUS BERUPA JSON VALID SAJA, TANPA MARKDOWN ATAU TEKS LAIN
            """
            
            print("Sending request to AI for quiz generation...")
            
            try:
                # REPLACED: self.groq.chat.completions.create() with self.generate_text()
                response_content = self.generate_text(prompt, max_length=4000, temperature=0.3)
                
                if not response_content:
                    print("Empty content from AI response")
                    return self._create_fallback_quiz(material, level, difficulty_settings)
                    
                print(f"AI response length: {len(response_content)}")
                print(f"AI response first 200 chars: {response_content[:200]}")
                
            except Exception as ai_error:
                print(f"AI generation error: {ai_error}")
                return self._create_fallback_quiz(material, level, difficulty_settings)
            
            # Clean up the content (similar to material generation) - SAME LOGIC
            response_content = response_content.strip()
            
            if response_content.startswith('```json'):
                response_content = response_content[7:]
                if response_content.endswith('```'):
                    response_content = response_content[:-3]
            elif response_content.startswith('```'):
                response_content = response_content[3:]
                if response_content.endswith('```'):
                    response_content = response_content[:-3]
            
            response_content = response_content.strip()
            
            if not response_content:
                print("Content is empty after cleanup")
                return self._create_fallback_quiz(material, level, difficulty_settings)
            
            print("Attempting to parse JSON...")
            
            # Try to parse JSON - SAME LOGIC
            try:
                quiz = json.loads(response_content)
                print("JSON parsing successful")
            except json.JSONDecodeError as json_error:
                print(f"JSON Parse Error: {json_error}")
                print(f"Content that failed to parse: {repr(response_content[:500])}")
                
                # Try to fix common JSON issues
                response_content = self._fix_json_content(response_content)
                try:
                    quiz = json.loads(response_content)
                    print("JSON parsing successful after fix")
                except json.JSONDecodeError:
                    print("Failed to fix JSON, using fallback")
                    return self._create_fallback_quiz(material, level, difficulty_settings)
            
            # Validate quiz structure - SAME LOGIC
            if not self._validate_quiz_structure(quiz):
                print("Invalid quiz structure, using fallback")
                return self._create_fallback_quiz(material, level, difficulty_settings)
            
            # Add metadata - SAME LOGIC
            quiz.update({
                "id": str(uuid.uuid4()),
                "material_id": material["id"],
                "specialization": material.get("specialization", "general"),
                "level": level,
                "user_id": material.get("user_id"),
                "created_at": datetime.now().isoformat(),
                "max_score": self._calculate_max_score(quiz, difficulty_settings),
                "estimated_duration": "15-30 minutes"
            })
            
            print(f"Quiz generated successfully with ID: {quiz['id']}")
            return quiz
            
        except Exception as e:
            print(f"Unexpected error in quiz generation: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            try:
                level = material.get('level', 'pemula')
                difficulty_settings = self._get_difficulty_settings(level)
                return self._create_fallback_quiz(material, level, difficulty_settings)
            except:
                raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")

    def _validate_and_normalize_level(self, level: str) -> str:
        """Validate and normalize difficulty level - SAME LOGIC"""
        if not level:
            return 'pemula'
            
        valid_levels = ['pemula', 'menengah', 'mahir']
        level = level.lower().strip()
        
        if level in valid_levels:
            return level
        
        level_mapping = {
            'beginner': 'pemula',
            'basic': 'pemula',
            'dasar': 'pemula',
            'intermediate': 'menengah',
            'medium': 'menengah',
            'tengah': 'menengah',
            'advanced': 'mahir',
            'expert': 'mahir',
            'lanjut': 'mahir',
            'lanjutan': 'mahir',
            'tinggi': 'mahir'
        }
        
        return level_mapping.get(level, 'pemula')
    
    def _get_difficulty_settings(self, level: str) -> Dict:
        """Get quiz settings based on difficulty level - SAME LOGIC"""
        settings = {
            'pemula': {
                'mc_count': (3, 5),
                'pq_count': (1, 2),
                'mc_points': 10,
                'pq_points': 15,
                'coding_points': 30
            },
            'menengah': {
                'mc_count': (5, 7),
                'pq_count': (2, 3), 
                'mc_points': 10,
                'pq_points': 15,
                'coding_points': 50
            },
            'mahir': {
                'mc_count': (6, 8),
                'pq_count': (3, 4),
                'mc_points': 10,
                'pq_points': 20,
                'coding_points': 70
            }
        }
        
        return settings.get(level, settings['pemula'])
    
    def _calculate_max_score(self, quiz: Dict, difficulty_settings: Dict) -> int:
        """Calculate maximum possible score for the quiz - SAME LOGIC"""
        mc_count = len(quiz.get("multiple_choice", []))
        pq_count = len(quiz.get("practical_questions", []))
        
        return (mc_count * difficulty_settings['mc_points'] + 
                pq_count * difficulty_settings['pq_points'] + 
                difficulty_settings['coding_points'])
    
    def _fix_json_content(self, content: str) -> str:
        """Try to fix common JSON formatting issues - SAME LOGIC"""
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return content
    
    def _validate_quiz_structure(self, quiz: Dict) -> bool:
        """Validate that quiz has required structure - SAME LOGIC"""
        required_keys = ['multiple_choice', 'practical_questions', 'coding_challenge']
        
        for key in required_keys:
            if key not in quiz:
                print(f"Missing required key: {key}")
                return False
        
        mc = quiz.get('multiple_choice', [])
        if not isinstance(mc, list) or len(mc) == 0:
            print("Invalid multiple_choice structure")
            return False
        
        for q in mc:
            required_fields = ['id', 'question', 'options', 'correct_answer']
            if not all(k in q for k in required_fields):
                print(f"Missing fields in multiple choice question: {q}")
                return False
                
            if not isinstance(q['options'], list) or len(q['options']) < 2:
                print("Invalid options in multiple choice")
                return False
        
        pq = quiz.get('practical_questions', [])
        if not isinstance(pq, list) or len(pq) == 0:
            print("Invalid practical_questions structure")
            return False
        
        cc = quiz.get('coding_challenge', {})
        if not isinstance(cc, dict):
            print("Invalid coding_challenge structure")
            return False
        
        return True
    
    def _create_fallback_quiz(self, material: Dict, level: str, difficulty_settings: Dict) -> Dict:
        """Create a basic fallback quiz when AI generation fails - SAME LOGIC"""
        title = material.get('title', 'Unknown Topic')
        specialization = material.get('specialization', 'general')
        learning_objectives = material.get('learning_objectives', [])
        
        spec_context = {
            'data_engineer': 'data engineering',
            'data_scientist': 'data science',
            'data_analyst': 'data analysis'
        }.get(specialization, 'programming')
        
        print(f"Creating fallback quiz for: {title} ({spec_context}, {level})")
        
        return {
            "id": str(uuid.uuid4()),
            "material_id": material["id"],
            "specialization": specialization,
            "level": level,
            "user_id": material.get("user_id"),
            "created_at": datetime.now().isoformat(),
            "estimated_duration": "15-30 minutes",
            "max_score": self._calculate_fallback_score(difficulty_settings),
            "is_fallback": True,
            "multiple_choice": [
                {
                    "id": "mc_1",
                    "question": f"Apa tujuan utama dari mempelajari {title}?",
                    "options": [
                        f"Memahami konsep fundamental dalam {spec_context}",
                        "Menyelesaikan tugas dengan cepat tanpa pemahaman mendalam",
                        "Mengikuti trend teknologi terbaru saja",
                        "Mendapat sertifikasi tanpa aplikasi praktis"
                    ],
                    "correct_answer": 0,
                    "explanation": f"Tujuan utama mempelajari {title} adalah memahami konsep fundamental yang akan menjadi dasar untuk pengembangan skill lebih lanjut.",
                    "difficulty": level,
                    "topic": "tujuan pembelajaran"
                },
                {
                    "id": "mc_2", 
                    "question": f"Berdasarkan learning objectives, apa yang paling penting dalam menguasai {title}?",
                    "options": [
                        "Membangun pemahaman teoritis yang kuat",
                        "Menghafal semua definisi dan formula",
                        "Fokus hanya pada tools dan software",
                        "Mengabaikan konsep dasar dan langsung ke advanced"
                    ],
                    "correct_answer": 0,
                    "explanation": "Membangun pemahaman teoritis yang kuat adalah fondasi untuk dapat mengaplikasikan knowledge secara efektif.",
                    "difficulty": level,
                    "topic": "strategi pembelajaran"
                }
            ],
            "practical_questions": [
                {
                    "id": "pq_1",
                    "question": f"Dalam konteks {spec_context}, bagaimana Anda akan mengaplikasikan konsep {title} dalam project nyata?",
                    "scenario": f"Anda bekerja dalam project {spec_context} dan perlu mengimplementasikan konsep dari {title}.",
                    "options": [
                        "Mulai dengan requirements analysis dan design yang matang",
                        "Langsung coding tanpa perencanaan detail",
                        "Copy-paste solution dari internet tanpa modifikasi",
                        "Menggunakan tools yang kompleks meskipun tidak diperlukan"
                    ],
                    "correct_answer": 0,
                    "explanation": "Pendekatan terbaik adalah memulai dengan analysis dan design yang matang sesuai dengan principles yang dipelajari dalam materi."
                }
            ],
            "coding_challenge": {
                "id": "code_1",
                "question": f"Implementasikan konsep dasar dari {title}",
                "problem_description": f"Buatlah implementasi sederhana yang mendemonstrasikan pemahaman Anda tentang {title} dalam konteks {spec_context}.",
                "requirements": [
                    "Kode harus clean dan well-documented",
                    f"Implementasi harus sesuai dengan level {level}",
                    "Gunakan best practices yang telah dipelajari"
                ],
                "input_format": "Input sesuai dengan konteks problem",
                "output_format": "Output yang menunjukkan konsep telah diimplementasikan",
                "sample_input": "sample_data",
                "sample_output": "processed_result", 
                "test_cases": [
                    {
                        "input": "test_input_1",
                        "expected_output": "expected_result_1"
                    }
                ],
                "hints": [
                    f"Fokus pada konsep fundamental dari {title}",
                    f"Sesuaikan kompleksitas dengan level {level}",
                    "Pastikan kode dapat dijelaskan step-by-step"
                ],
                "solution_approach": f"Gunakan pendekatan systematic yang sesuai dengan principles dalam {title}. Mulai dari understanding problem, design solution, kemudian implement dengan best practices."
            }
        }
    
    def _calculate_fallback_score(self, difficulty_settings: Dict) -> int:
        """Calculate score for fallback quiz - SAME LOGIC"""
        return (2 * difficulty_settings['mc_points'] + 
                1 * difficulty_settings['pq_points'] + 
                difficulty_settings['coding_points'])
    
    def evaluate_comprehensive_quiz(self, quiz: Dict, user_answers: List[Dict], coding_answer: str = None) -> Dict:
        """Comprehensively evaluate quiz answers - SAME LOGIC"""
        
        try:
            mc_questions = quiz.get("multiple_choice", [])
            practical_questions = quiz.get("practical_questions", [])
            coding_challenge = quiz.get("coding_challenge", {})
            
            # Evaluate multiple choice
            mc_score = 0
            mc_feedback = []
            
            for question in mc_questions:
                user_answer = next((a["answer"] for a in user_answers if a["question_id"] == question["id"]), None)
                is_correct = False
                
                if user_answer is not None:
                    try:
                        user_answer_idx = int(user_answer)
                        is_correct = user_answer_idx == question["correct_answer"]
                        if is_correct:
                            mc_score += 10
                    except:
                        pass
                
                mc_feedback.append({
                    "question_id": question["id"],
                    "question": question["question"],
                    "user_answer": user_answer,
                    "correct_answer": question["correct_answer"],
                    "is_correct": is_correct,
                    "explanation": question.get("explanation", ""),
                    "points": 10 if is_correct else 0
                })
            
            # Evaluate practical questions
            practical_score = 0
            practical_feedback = []
            
            for question in practical_questions:
                user_answer = next((a["answer"] for a in user_answers if a["question_id"] == question["id"]), None)
                is_correct = False
                
                if user_answer is not None:
                    try:
                        user_answer_idx = int(user_answer)
                        is_correct = user_answer_idx == question["correct_answer"]
                        if is_correct:
                            practical_score += 15
                    except:
                        pass
                
                practical_feedback.append({
                    "question_id": question["id"],
                    "question": question["question"],
                    "user_answer": user_answer,
                    "correct_answer": question["correct_answer"],
                    "is_correct": is_correct,
                    "explanation": question.get("explanation", ""),
                    "points": 15 if is_correct else 0
                })
            
            # Evaluate coding challenge using AI
            coding_score = 0
            coding_feedback = {}
            
            if coding_answer and coding_challenge:
                coding_evaluation = self._evaluate_code_with_ai(coding_challenge, coding_answer)
                coding_score = coding_evaluation.get("score", 0)
                coding_feedback = coding_evaluation
            
            # Calculate total score
            max_score = quiz.get("max_score", 100)
            total_score = mc_score + practical_score + coding_score
            percentage = (total_score / max_score) * 100
            
            # Determine performance level
            if percentage >= 85:
                performance = "Excellent"
            elif percentage >= 70:
                performance = "Good"
            elif percentage >= 60:
                performance = "Satisfactory"
            else:
                performance = "Needs Improvement"
            
            return {
                "total_score": total_score,
                "max_score": max_score,
                "percentage": round(percentage, 1),
                "performance": performance,
                "passed": percentage >= 60,
                "breakdown": {
                    "multiple_choice": {
                        "score": mc_score,
                        "max_score": len(mc_questions) * 10,
                        "feedback": mc_feedback
                    },
                    "practical": {
                        "score": practical_score,
                        "max_score": len(practical_questions) * 15,
                        "feedback": practical_feedback
                    },
                    "coding": {
                        "score": coding_score,
                        "max_score": 50,
                        "feedback": coding_feedback
                    }
                },
                "overall_feedback": self._generate_overall_feedback(percentage, performance),
                "submitted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error evaluating quiz: {e}")
            raise HTTPException(status_code=500, detail="Failed to evaluate quiz")
    
    def _evaluate_code_with_ai(self, coding_challenge: Dict, code_answer: str) -> Dict:
        """Enhanced coding answer evaluation - SAME LOGIC, just replace groq call"""
        
        print("🔍 DEBUG: Starting code evaluation...")
        print(f"🔍 DEBUG: Code answer length: {len(code_answer) if code_answer else 0}")
        
        try:
            ideal_solution = self._generate_ideal_solution(coding_challenge)
            print("🔍 DEBUG: Ideal solution generated")
            
            prompt = f"""
            Sebagai expert code reviewer dan educator, evaluasi kode siswa berikut dengan sangat detail:
            
            CODING CHALLENGE:
            Problem: {coding_challenge.get('problem_description', '')}
            Requirements: {coding_challenge.get('requirements', [])}
            Expected Input: {coding_challenge.get('input_format', '')}
            Expected Output: {coding_challenge.get('output_format', '')}
            Sample Input: {coding_challenge.get('sample_input', '')}
            Sample Output: {coding_challenge.get('sample_output', '')}
            Test Cases: {coding_challenge.get('test_cases', [])}
            
            IDEAL SOLUTION (untuk perbandingan):
            {ideal_solution}
            
            STUDENT'S CODE:
            {code_answer}
            
            Berikan evaluasi komprehensif dalam format JSON:
            {{
                "score": 35,
                "detailed_feedback": {{
                    "correctness": {{
                        "score": 20,
                        "status": "good",
                        "explanation": "Kode menyelesaikan masalah dengan benar dan menghasilkan output yang sesuai",
                        "test_results": [
                            {{
                                "test_case": "Basic test -> Expected output",
                                "student_output": "Actual student output",
                                "passed": true,
                                "explanation": "Test case berhasil dijalankan"
                            }}
                        ]
                    }},
                    "code_quality": {{
                        "score": 12,
                        "aspects": {{
                            "readability": {{
                                "score": 4,
                                "feedback": "Kode mudah dibaca dengan struktur yang jelas"
                            }},
                            "efficiency": {{
                                "score": 4,
                                "feedback": "Algoritma cukup efisien untuk masalah ini"
                            }},
                            "best_practices": {{
                                "score": 4,
                                "feedback": "Menggunakan Python best practices dengan baik"
                            }}
                        }}
                    }},
                    "requirements_compliance": {{
                        "score": 3,
                        "checked_requirements": [
                            {{
                                "requirement": "Implement required function",
                                "met": true,
                                "explanation": "Function berhasil diimplementasikan sesuai requirement"
                            }}
                        ]
                    }}
                }},
                "code_comparison": {{
                    "approach_similarity": "Pendekatan siswa mirip dengan solusi optimal, menunjukkan pemahaman yang baik",
                    "alternative_approaches": [
                        "Bisa menggunakan list comprehension untuk lebih concise",
                        "Alternatif dengan built-in functions Python"
                    ],
                    "improvements_needed": [
                        "Tambahkan error handling untuk edge cases",
                        "Optimisasi space complexity"
                    ]
                }},
                "learning_insights": {{
                    "concepts_understood": [
                        "Basic Python syntax dan control structures",
                        "Function definition dan parameter handling",
                        "Data manipulation techniques"
                    ],
                    "concepts_to_review": [
                        "Advanced algorithm optimization",
                        "Error handling best practices",
                        "Memory efficiency techniques"
                    ],
                    "next_practice_suggestions": [
                        "Practice dengan algorithm complexity analysis",
                        "Latihan error handling scenarios",
                        "Study Python built-in functions"
                    ]
                }},
                "ideal_solution_explanation": {{
                    "why_this_approach": "Pendekatan ini optimal karena menggunakan algoritma yang efisien dan readable",
                    "key_techniques": [
                        "Efficient looping techniques",
                        "Proper data structure usage",
                        "Clear variable naming"
                    ],
                    "complexity_analysis": "Time complexity: O(n), Space complexity: O(1)"
                }},
                "personalized_feedback": "Kode Anda menunjukkan pemahaman yang solid tentang konsep programming. Dengan beberapa optimisasi kecil, kode ini bisa menjadi excellent!",
                "actionable_steps": [
                    "Tambahkan comments untuk menjelaskan logic kompleks",
                    "Implement error handling untuk input validation",
                    "Practice algorithm optimization techniques"
                ]
            }}
            """
            
            # REPLACED: self.groq.chat.completions.create() with self.generate_text()
            response_content = self.generate_text(prompt, max_length=6000, temperature=0.2)
            
            print(f"🔍 DEBUG: AI response received, length: {len(response_content) if response_content else 0}")
            
            if not response_content or response_content.strip() == "":
                print("⚠️ WARNING: Empty response from AI, using fallback evaluation")
                return self._create_fallback_code_evaluation(code_answer, coding_challenge)
            
            response_content = response_content.strip()
            
            # Remove code block markers if present
            if response_content.startswith('```json'):
                response_content = response_content[7:-3].strip()
            elif response_content.startswith('```'):
                response_content = response_content[3:-3].strip()
            
            if not response_content:
                print("⚠️ WARNING: Content empty after cleanup, using fallback evaluation")
                return self._create_fallback_code_evaluation(code_answer, coding_challenge)
            
            try:
                evaluation = json.loads(response_content)
                print("✅ DEBUG: JSON parsing successful")
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON parsing error: {json_error}")
                print(f"Content that failed to parse: {response_content[:200]}...")
                return self._create_fallback_code_evaluation(code_answer, coding_challenge)
            
            # Add the ideal solution to the response
            evaluation["ideal_solution"] = {
                "code": ideal_solution,
                "explanation": "Ini adalah salah satu solusi optimal untuk masalah ini. Bandingkan dengan kode Anda!"
            }
            
            result = {
                "score": evaluation.get("score", 35),
                "max_score": 50,
                "enhanced_feedback": evaluation,
                "feedback": {
                    "correctness": evaluation.get("detailed_feedback", {}).get("correctness", {}).get("explanation", "Code evaluated"),
                    "code_quality": evaluation.get("personalized_feedback", "Code quality assessed"),
                    "requirements_met": "Requirements compliance checked",
                    "strengths": evaluation.get("learning_insights", {}).get("concepts_understood", []),
                    "areas_for_improvement": evaluation.get("code_comparison", {}).get("improvements_needed", []),
                    "suggestions": evaluation.get("actionable_steps", [])
                }
            }
            
            print("✅ DEBUG: Enhanced feedback created successfully")
            return result
            
        except Exception as e:
            print(f"❌ ERROR evaluating code: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_code_evaluation(code_answer, coding_challenge)
    
    def _create_fallback_code_evaluation(self, code_answer: str, coding_challenge: Dict) -> Dict:
        """Create fallback evaluation when AI fails - SAME LOGIC"""
        
        print("🔄 DEBUG: Using fallback evaluation")
        
        fallback_enhanced = {
            "score": 35,
            "detailed_feedback": {
                "correctness": {
                    "score": 20,
                    "status": "good",
                    "explanation": "Kode telah dievaluasi dan berfungsi dengan baik untuk test cases dasar",
                    "test_results": [
                        {
                            "test_case": "Sample test case",
                            "student_output": "Output sesuai ekspektasi",
                            "passed": True,
                            "explanation": "Basic test case berhasil dijalankan"
                        }
                    ]
                },
                "code_quality": {
                    "score": 12,
                    "aspects": {
                        "readability": {
                            "score": 4,
                            "feedback": "Kode mudah dibaca dan terstruktur dengan baik"
                        },
                        "efficiency": {
                            "score": 4,
                            "feedback": "Algoritma cukup efisien untuk problem size ini"
                        },
                        "best_practices": {
                            "score": 4,
                            "feedback": "Menggunakan Python conventions dengan baik"
                        }
                    }
                },
                "requirements_compliance": {
                    "score": 3,
                    "checked_requirements": [
                        {
                            "requirement": "Implement core functionality",
                            "met": True,
                            "explanation": "Fungsi utama berhasil diimplementasikan"
                        }
                    ]
                }
            },
            "code_comparison": {
                "approach_similarity": "Pendekatan yang dipilih menunjukkan pemahaman yang baik terhadap masalah",
                "alternative_approaches": [
                    "Bisa menggunakan built-in functions untuk optimisasi",
                    "List comprehension untuk kode yang lebih concise",
                    "Algoritma yang lebih efisien untuk kasus kompleks"
                ],
                "improvements_needed": [
                    "Tambahkan error handling untuk edge cases",
                    "Optimisasi untuk kasus dengan input besar",
                    "Improve code documentation"
                ]
            },
            "learning_insights": {
                "concepts_understood": [
                    "Basic Python programming concepts",
                    "Function implementation dan structure",
                    "Data manipulation techniques",
                    "Basic algorithm logic"
                ],
                "concepts_to_review": [
                    "Algorithm optimization techniques",
                    "Error handling best practices",
                    "Advanced Python features",
                    "Code testing strategies"
                ],
                "next_practice_suggestions": [
                    "Practice dengan problem complexity yang lebih tinggi",
                    "Study advanced algorithm patterns",
                    "Learn about code optimization techniques",
                    "Practice with different data structures"
                ]
            },
            "ideal_solution_explanation": {
                "why_this_approach": "Pendekatan ini balance antara readability, efficiency, dan maintainability",
                "key_techniques": [
                    "Clear logic flow dan structure",
                    "Appropriate data structures usage",
                    "Good variable naming conventions",
                    "Efficient algorithm implementation"
                ],
                "complexity_analysis": "Time complexity analysis akan tersedia setelah detailed review"
            },
            "personalized_feedback": "Kode Anda menunjukkan effort yang baik dan pemahaman konsep yang solid. Dengan beberapa optimisasi dan practice tambahan, kode ini bisa menjadi excellent!",
            "actionable_steps": [
                "Review algoritma untuk optimisasi lebih lanjut",
                "Tambahkan comprehensive error handling",
                "Practice dengan test cases yang lebih bervariasi",
                "Study Python best practices documentation"
            ],
            "ideal_solution": {
                "code": f"""# Ideal solution for the coding challenge
# Problem: {coding_challenge.get('problem_description', 'Coding challenge')}

def solution():
    # Optimized approach for this problem
    # Implementation would be generated based on the specific challenge
    pass

# Note: Detailed ideal solution sedang diproses
# Silakan review feedback di atas untuk improvement""",
                "explanation": "Solusi ideal akan menunjukkan approach yang optimal dengan balance antara readability dan efficiency"
            }
        }
        
        return {
            "score": 35,
            "max_score": 50,
            "enhanced_feedback": fallback_enhanced,
            "feedback": {
                "correctness": "Kode berfungsi dengan baik untuk test cases dasar",
                "code_quality": "Code quality baik dengan room for improvement di beberapa area",
                "requirements_met": "Requirements utama sudah terpenuhi dengan baik",
                "strengths": ["Basic programming concepts", "Function implementation", "Logic structure"],
                "areas_for_improvement": ["Algorithm optimization", "Error handling", "Code documentation"],
                "suggestions": ["Practice advanced problems", "Study optimization techniques", "Learn testing strategies"]
            }
        }

    def _generate_ideal_solution(self, coding_challenge: Dict) -> str:
        """Generate ideal solution for comparison - SAME LOGIC, just replace groq call"""
        try:
            prompt = f"""
            Generate an ideal Python solution for this coding challenge:
            
            Problem: {coding_challenge.get('problem_description', '')}
            Requirements: {coding_challenge.get('requirements', [])}
            Input Format: {coding_challenge.get('input_format', '')}
            Output Format: {coding_challenge.get('output_format', '')}
            Sample Input: {coding_challenge.get('sample_input', '')}
            Sample Output: {coding_challenge.get('sample_output', '')}
            
            Provide clean, efficient Python code with comments.
            """
            
            # REPLACED: groq call with generate_text
            solution = self.generate_text(prompt, max_length=1000, temperature=0.1)
            return solution
            
        except Exception as e:
            print(f"Error generating ideal solution: {e}")
            return f"# Ideal solution for: {coding_challenge.get('problem_description', 'coding challenge')}\n# Solution implementation would be optimized for this specific problem"

    def _generate_overall_feedback(self, percentage: float, performance: str) -> str:
        """Generate overall feedback based on performance - SAME LOGIC"""
        
        feedback_templates = {
            "Excellent": (
                f"Luar biasa! Anda menguasai materi dengan sangat baik ({percentage}%). "
                "Anda siap untuk topik yang lebih advanced."
            ),
            "Good": (
                f"Bagus! Anda memahami sebagian besar materi ({percentage}%). "
                "Dengan sedikit review, Anda akan menguasai sepenuhnya."
            ),
            "Satisfactory": (
                f"Cukup baik ({percentage}%). Anda memahami konsep dasar, "
                "tapi perlu lebih banyak latihan untuk menguasai detail."
            ),
            "Needs Improvement": (
                f"Perlu peningkatan ({percentage}%). Disarankan untuk review materi "
                "dan berlatih lebih banyak sebelum lanjut ke topik berikutnya."
            )
        }

        return feedback_templates.get(performance, "Terus semangat belajar!")

    def generate_comprehensive_report(self, user: Dict, material: Dict, quiz_result: Dict) -> Dict:
        """Generate comprehensive learning report - SAME LOGIC"""
        
        try:
            username = user.get('username', 'Learner')
            level = user.get('level', 'pemula')
            specialization = user.get('specialization', 'data_science')
            material_title = material.get('title', 'Unknown Material')
            score = quiz_result.get('percentage', 0)
            performance = quiz_result.get('performance', 'Needs Improvement')
            
            # Analyze learning progress
            progress_analysis = self._analyze_learning_progress(quiz_result)
            
            # Generate personalized recommendations
            recommendations = self._generate_personalized_recommendations(
                specialization, level, score, quiz_result, material
            )
            
            return {
                "student_info": {
                    "name": username,
                    "specialization": specialization.replace('_', ' ').title(),
                    "current_level": level.title(),
                    "material_completed": material_title
                },
                "performance_summary": {
                    "overall_score": score,
                    "performance_level": performance,
                    "passed": quiz_result.get('passed', False),
                    "completion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "detailed_analysis": progress_analysis,
                "recommendations": recommendations,
                "achievements": self._generate_achievements(score, performance, level),
                "next_steps": self._generate_next_steps(specialization, level, score, material),
                "learning_insights": self._generate_learning_insights(quiz_result),
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating report: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate learning report")
    
    def _analyze_learning_progress(self, quiz_result: Dict) -> Dict:
        """Analyze detailed learning progress - SAME LOGIC"""
        
        breakdown = quiz_result.get('breakdown', {})
        
        return {
            "conceptual_understanding": {
                "score": breakdown.get('multiple_choice', {}).get('score', 0),
                "max_score": breakdown.get('multiple_choice', {}).get('max_score', 0),
                "analysis": "Mengukur pemahaman konsep teoritis"
            },
            "practical_application": {
                "score": breakdown.get('practical', {}).get('score', 0),
                "max_score": breakdown.get('practical', {}).get('max_score', 0),
                "analysis": "Mengukur kemampuan aplikasi praktis"
            },
            "technical_implementation": {
                "score": breakdown.get('coding', {}).get('score', 0),
                "max_score": breakdown.get('coding', {}).get('max_score', 0),
                "analysis": "Mengukur kemampuan implementasi teknis"
            }
        }
    
    def _generate_personalized_recommendations(self, specialization: str, level: str, score: float, quiz_result: Dict, material: Dict) -> Dict:
        """Generate personalized learning recommendations - SAME LOGIC"""
        
        next_topics = material.get('next_topics', [])
        learning_path = material.get('learning_path', [])
        
        if score >= 80:
            return {
                "status": "ready_to_advance",
                "message": "Anda siap untuk materi yang lebih advanced!",
                "recommended_topics": next_topics[:2] if next_topics else [],
                "study_approach": [
                    "Lanjutkan ke topik berikutnya dalam learning path",
                    "Mulai project praktis untuk mengaplikasikan pengetahuan",
                    "Bergabung dengan komunitas untuk berbagi pengalaman"
                ],
                "estimated_timeline": "1-2 minggu untuk topik berikutnya"
            }
        elif score >= 60:
            return {
                "status": "review_and_advance",
                "message": "Pemahaman baik, tapi ada area yang perlu diperkuat",
                "recommended_topics": next_topics[:1] if next_topics else [],
                "review_areas": self._identify_weak_areas(quiz_result),
                "study_approach": [
                    "Review area yang lemah terlebih dahulu",
                    "Lakukan latihan tambahan pada konsep yang belum dikuasai",
                    "Setelah yakin, lanjut ke topik berikutnya"
                ],
                "estimated_timeline": "2-3 minggu termasuk review"
            }
        else:
            return {
                "status": "need_review",
                "message": "Disarankan untuk mempelajari ulang materi ini",
                "recommended_topics": [],
                "focus_areas": [
                    "Pelajari ulang konsep dasar",
                    "Lakukan lebih banyak latihan praktis",
                    "Konsultasi dengan mentor atau bergabung study group"
                ],
                "study_approach": [
                    "Ulangi pembelajaran dari awal dengan pendekatan yang berbeda",
                    "Fokus pada hands-on practice",
                    "Cari sumber pembelajaran tambahan"
                ],
                "estimated_timeline": "3-4 minggu untuk review mendalam"
            }
    
    def _identify_weak_areas(self, quiz_result: Dict) -> List[str]:
        """Identify areas that need improvement - SAME LOGIC"""
        
        weak_areas = []
        breakdown = quiz_result.get('breakdown', {})
        
        mc_score = breakdown.get('multiple_choice', {}).get('score', 0)
        mc_max = breakdown.get('multiple_choice', {}).get('max_score', 1)
        if (mc_score / mc_max) < 0.7:
            weak_areas.append("Pemahaman konsep teoritis")
        
        practical_score = breakdown.get('practical', {}).get('score', 0)
        practical_max = breakdown.get('practical', {}).get('max_score', 1)
        if (practical_score / practical_max) < 0.7:
            weak_areas.append("Aplikasi praktis")
        
        coding_score = breakdown.get('coding', {}).get('score', 0)
        coding_max = breakdown.get('coding', {}).get('max_score', 1)
        if (coding_score / coding_max) < 0.7:
            weak_areas.append("Implementasi teknis")
        
        return weak_areas
    
    def _generate_achievements(self, score: float, performance: str, level: str) -> List[Dict]:
        """Generate achievement badges - SAME LOGIC"""
        
        achievements = []
        
        if score >= 90:
            achievements.append({
                "badge": "Excellence Badge",
                "description": "Menguasai materi dengan sangat baik",
                "icon": "🏆"
            })
        elif score >= 80:
            achievements.append({
                "badge": "Mastery Badge", 
                "description": "Menunjukkan pemahaman yang solid",
                "icon": "🥇"
            })
        elif score >= 70:
            achievements.append({
                "badge": "Competency Badge",
                "description": "Mencapai tingkat kompetensi yang baik",
                "icon": "🥈"
            })
        else:
            achievements.append({
                "badge": "Effort Badge",
                "description": "Menunjukkan usaha dalam pembelajaran",
                "icon": "💪"
            })
        
        achievements.append({
            "badge": f"{level.title()} Learner",
            "description": f"Belajar di level {level}",
            "icon": "📚"
        })
        
        return achievements
    
    def _generate_next_steps(self, specialization: str, level: str, score: float, material: Dict) -> Dict:
        """Generate specific next steps - SAME LOGIC"""
        
        learning_path = material.get('learning_path', [])
        current_index = material.get('current_topic_index', 0)
        
        if score >= 70:
            next_topic = learning_path[current_index + 1] if current_index + 1 < len(learning_path) else "Advanced Topics"
            return {
                "immediate_action": f"Lanjut ke: {next_topic}",
                "short_term_goals": [
                    f"Menyelesaikan {next_topic} dalam 2 minggu",
                    "Mengerjakan project praktis",
                    "Bergabung dengan komunitas"
                ],
                "long_term_vision": f"Menjadi {specialization.replace('_', ' ').title()} yang kompeten",
                "suggested_resources": [
                    "Online courses untuk topik lanjutan",
                    "GitHub repositories untuk referensi",
                    "Kaggle competitions (untuk Data Science)",
                    "Open source projects untuk kontribusi"
                ]
            }
        else:
            return {
                "immediate_action": "Review dan perkuat pemahaman dasar",
                "short_term_goals": [
                    "Mencapai skor minimal 70% pada quiz ulang",
                    "Menyelesaikan latihan tambahan",
                    "Memahami konsep yang masih lemah"
                ],
                "long_term_vision": f"Membangun fondasi yang kuat dalam {specialization.replace('_', ' ').title()}",
                "suggested_resources": [
                    "Tutorial dasar untuk review",
                    "Practice exercises",
                    "Mentoring atau study group",
                    "Alternative learning materials"
                ]
            }
    
    def _generate_learning_insights(self, quiz_result: Dict) -> Dict:
        """Generate insights about learning pattern - SAME LOGIC"""
        
        breakdown = quiz_result.get('breakdown', {})
        
        mc_percentage = (breakdown.get('multiple_choice', {}).get('score', 0) / 
                        max(breakdown.get('multiple_choice', {}).get('max_score', 1), 1)) * 100
        practical_percentage = (breakdown.get('practical', {}).get('score', 0) / 
                               max(breakdown.get('practical', {}).get('max_score', 1), 1)) * 100
        coding_percentage = (breakdown.get('coding', {}).get('score', 0) / 
                            max(breakdown.get('coding', {}).get('max_score', 1), 1)) * 100
        
        learning_style = "balanced"
        if mc_percentage > practical_percentage and mc_percentage > coding_percentage:
            learning_style = "theoretical"
        elif practical_percentage > mc_percentage and practical_percentage > coding_percentage:
            learning_style = "practical"
        elif coding_percentage > mc_percentage and coding_percentage > practical_percentage:
            learning_style = "hands-on"
        
        return {
            "learning_style": learning_style,
            "strength_area": self._identify_strength_area(mc_percentage, practical_percentage, coding_percentage),
            "improvement_area": self._identify_improvement_area(mc_percentage, practical_percentage, coding_percentage),
            "study_recommendations": self._get_study_recommendations(learning_style),
            "progress_trend": "improving" if quiz_result.get('passed', False) else "needs_focus"
        }
    
    def _identify_strength_area(self, mc_pct: float, practical_pct: float, coding_pct: float) -> str:
        """Identify the strongest area - SAME LOGIC"""
        max_score = max(mc_pct, practical_pct, coding_pct)
        if max_score == mc_pct:
            return "Conceptual Understanding"
        elif max_score == practical_pct:
            return "Practical Application"
        else:
            return "Technical Implementation"
    
    def _identify_improvement_area(self, mc_pct: float, practical_pct: float, coding_pct: float) -> str:
        """Identify the area needing most improvement - SAME LOGIC"""
        min_score = min(mc_pct, practical_pct, coding_pct)
        if min_score == mc_pct:
            return "Conceptual Understanding"
        elif min_score == practical_pct:
            return "Practical Application"
        else:
            return "Technical Implementation"
    
    def _get_study_recommendations(self, learning_style: str) -> List[str]:
        """Get study recommendations based on learning style - SAME LOGIC"""
        recommendations = {
            "theoretical": [
                "Fokus pada pemahaman konsep dengan diagram dan visualisasi",
                "Baca dokumentasi dan teori secara mendalam",
                "Diskusi konsep dengan peers atau mentor"
            ],
            "practical": [
                "Lebih banyak hands-on exercises dan case studies",
                "Terapkan konsep dalam project nyata",
                "Belajar melalui problem-solving"
            ],
            "hands-on": [
                "Coding bootcamp dan intensive programming",
                "Contribute to open source projects",
                "Build portfolio melalui personal projects"
            ],
            "balanced": [
                "Kombinasi teori dan praktik yang seimbang",
                "Variasikan metode pembelajaran",
                "Project-based learning dengan konsep yang kuat"
            ]
        }
        return recommendations.get(learning_style, recommendations["balanced"])
    
    def chat_about_material(self, material: Dict, question: str, chat_history: List = None) -> str:
        """Enhanced chatbot for material discussion - SAME LOGIC, just replace groq call"""
        
        try:
            title = material.get('title', 'materi pembelajaran')
            level = material.get('level', 'pemula')
            content = json.dumps(material.get('content', {}), indent=2)[:2500]
            learning_objectives = material.get('learning_objectives', [])
            
            # Build context from chat history
            history_context = ""
            if chat_history:
                recent_history = chat_history[-5:]
                for msg in recent_history:
                    history_context += f"Q: {msg.get('question', '')}\nA: {msg.get('answer', '')}\n"
            
            prompt = f"""
            Anda adalah AI tutor yang ahli dan sabar untuk materi "{title}" (level: {level}).
            
            MATERI PEMBELAJARAN:
            {content}
            
            LEARNING OBJECTIVES:
            {', '.join(learning_objectives)}
            
            RIWAYAT PERCAKAPAN:
            {history_context}
            
            PERTANYAAN STUDENT: {question}
            
            Instruksi:
            1. Jawab pertanyaan dengan jelas dan sesuai level {level}
            2. Berikan penjelasan yang mendidik dan mudah dipahami
            3. Gunakan contoh praktis jika relevan
            4. Jika pertanyaan di luar scope materi, arahkan kembali ke topik
            5. Berikan encouragement untuk belajar
            6. Jika student terlihat kesulitan, berikan step-by-step explanation
            7. Gunakan bahasa Indonesia yang natural dan friendly
            
            Jawab dengan tone supportive dan educational:
            """
            
            # REPLACED: self.groq.chat.completions.create() with self.generate_text()
            response = self.generate_text(prompt, max_length=1000, temperature=0.6)
            
            return response
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return f"Maaf, saya mengalami kesulitan menjawab pertanyaan tentang {material.get('title', 'materi ini')}. Bisa coba tanya dengan cara yang berbeda atau lebih spesifik?"

# Initialize the model when the module loads
model_initialized = initialize_model()

# Initialize agent
agent = EducationAgent(tavily_client)

# API Endpoints - ALL SAME AS ORIGINAL
@app.post("/api/user/login")
async def login_user(user: User):
    """User login and registration"""
    user_id = str(uuid.uuid4())
    user_data = user.model_dump()
    user_data['user_id'] = user_id
    user_data['created_at'] = datetime.now().isoformat()
    
    users_db[user_id] = user_data
    
    return {
        "message": f"Welcome {user.username}! Ready to start your {user.specialization.replace('_', ' ').title()} journey?",
        "user_id": user_id,
        "next_step": "assessment"
    }

@app.get("/api/assessment/{user_id}")
async def get_assessment(user_id: str):
    """Get personalized assessment questions"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    specialization = user['specialization']
    
    questions_data = agent.generate_assessment_questions(specialization)
    
    assessment_id = str(uuid.uuid4())
    assessments_db[assessment_id] = {
        "id": assessment_id,
        "user_id": user_id,
        "specialization": specialization,
        "questions": questions_data.get("questions", []),
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "assessment_id": assessment_id,
        "message": f"Assessment for {specialization.replace('_', ' ').title()}",
        "instructions": "Jawab semua pertanyaan dengan jujur untuk mendapatkan level yang sesuai dengan kemampuan Anda",
        "questions": questions_data.get("questions", []),
        "total_questions": len(questions_data.get("questions", []))
    }

@app.post("/api/assessment/submit")
async def submit_assessment(submission: AssessmentSubmission):
    """Submit assessment and get level classification"""
    if submission.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_assessment = None
    for assessment in assessments_db.values():
        if assessment['user_id'] == submission.user_id:
            user_assessment = assessment
            break
    
    if not user_assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    evaluation = agent.evaluate_assessment(submission.answers, user_assessment['questions'])
    
    users_db[submission.user_id]['level'] = evaluation['level']
    users_db[submission.user_id]['assessment_result'] = evaluation
    
    return {
        "message": f"Assessment completed! Your level: {evaluation['level'].title()}",
        "level": evaluation['level'],
        "score": evaluation['total_score'],
        "scores_breakdown": evaluation['scores_by_difficulty'],
        "recommendation": evaluation['recommendation'],
        "next_step": "personalized_learning"
    }

@app.get("/api/learning/start/{user_id}")
async def start_personalized_learning(user_id: str):
    """Generate personalized learning material"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    
    if not user.get('level'):
        raise HTTPException(status_code=400, detail="Please complete assessment first")
    
    specialization = user['specialization']
    level = user['level']
    
    material = agent.search_and_create_personalized_material(specialization, level, user_id)
    
    material_id = material['id']
    materials_db[material_id] = material
    
    return {
        "message": f"Your personalized {specialization.replace('_', ' ').title()} learning material is ready!",
        "material_id": material_id,
        "material": material,
        "estimated_duration": material.get('estimated_duration', 'Unknown'),
        "next_step": "study_then_quiz"
    }

@app.get("/api/material/{material_id}")
async def get_material(material_id: str):
    """Get specific learning material"""
    if material_id not in materials_db:
        raise HTTPException(status_code=404, detail="Material not found")
    
    return materials_db[material_id]

@app.get("/api/quiz/generate/{material_id}")
async def generate_quiz_for_material(material_id: str):
    """Generate adaptive quiz for completed material"""
    if material_id not in materials_db:
        raise HTTPException(status_code=404, detail="Material not found")
    
    material = materials_db[material_id]
    
    quiz = agent.generate_adaptive_quiz(material)
    
    return {
        "message": "Quiz generated! Test your understanding of the material.",
        "quiz": quiz,
        "instructions": "Jawab semua pertanyaan dengan baik. Ada pilihan ganda, pertanyaan praktis, dan coding challenge.",
        "max_score": quiz.get("max_score", 100)
    }

@app.post("/api/quiz/submit")
async def submit_quiz_answers(submission: QuizSubmission):
    """Submit quiz answers for comprehensive evaluation with enhanced coding feedback"""
    if submission.material_id not in materials_db:
        raise HTTPException(status_code=404, detail="Material not found")
    
    if submission.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    material = materials_db[submission.material_id]
    user = users_db[submission.user_id]
    
    quiz = agent.generate_adaptive_quiz(material)
    
    result = agent.evaluate_comprehensive_quiz(quiz, submission.answers, submission.coding_answer)
    
    result_id = str(uuid.uuid4())
    quiz_results_db[result_id] = {
        "id": result_id,
        "user_id": submission.user_id,
        "material_id": submission.material_id,
        "quiz": quiz,
        "user_answers": submission.answers,
        "coding_answer": submission.coding_answer,
        "result": result,
        "user_level": user.get('level', 'pemula'),
        "user_specialization": user.get('specialization', 'general'),
        "submitted_at": datetime.now().isoformat()
    }
    
    return {
        "message": "Quiz evaluated successfully with enhanced coding feedback!",
        "result_id": result_id,
        "result": result,
        "enhanced_features": [
            "Detailed code analysis with test case results",
            "Comparison with ideal solution",
            "Personalized learning insights",
            "Actionable improvement suggestions"
        ],
        "next_step": "view_report"
    }

@app.get("/api/report/{result_id}")
async def get_comprehensive_learning_report(result_id: str):
    """Get comprehensive learning report (raport)"""
    if result_id not in quiz_results_db:
        raise HTTPException(status_code=404, detail="Result not found")
    
    quiz_result_data = quiz_results_db[result_id]
    user = users_db[quiz_result_data["user_id"]]
    material = materials_db[quiz_result_data["material_id"]]
    quiz_result = quiz_result_data["result"]
    
    report = agent.generate_comprehensive_report(user, material, quiz_result)
    
    return {
        "message": "Your comprehensive learning report is ready!",
        "report": report,
        "material_info": {
            "title": material.get("title"),
            "level": material.get("level"),
            "specialization": material.get("specialization")
        }
    }

@app.post("/api/chat")
async def chat_with_ai_tutor(message: ChatMessage):
    """Chat with AI tutor about the learning material"""
    if message.material_id not in materials_db:
        raise HTTPException(status_code=404, detail="Material not found")
    
    if message.user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    material = materials_db[message.material_id]
    
    chat_history = []
    
    response = agent.chat_about_material(material, message.message, chat_history)
    
    return {
        "response": response,
        "material_title": material.get("title", "Unknown Material"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/user/{user_id}/progress")
async def get_user_progress(user_id: str):
    """Get user's learning progress summary"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    
    completed_materials = []
    quiz_results = []
    
    for material_id, material in materials_db.items():
        if material.get('user_id') == user_id:
            completed_materials.append({
                "id": material_id,
                "title": material.get('title'),
                "level": material.get('level'),
                "created_at": material.get('created_at')
            })
    
    for result_id, result_data in quiz_results_db.items():
        if result_data.get('user_id') == user_id:
            quiz_results.append({
                "id": result_id,
                "material_id": result_data.get('material_id'),
                "score": result_data.get('result', {}).get('percentage', 0),
                "passed": result_data.get('result', {}).get('passed', False),
                "submitted_at": result_data.get('submitted_at')
            })
    
    return {
        "user_info": {
            "username": user.get('username'),
            "specialization": user.get('specialization'),
            "level": user.get('level'),
            "joined_at": user.get('created_at')
        },
        "progress_summary": {
            "materials_completed": len(completed_materials),
            "quizzes_taken": len(quiz_results),
            "average_score": sum(r['score'] for r in quiz_results) / max(len(quiz_results), 1),
            "success_rate": sum(1 for r in quiz_results if r['passed']) / max(len(quiz_results), 1) * 100
        },
        "completed_materials": completed_materials,
        "quiz_history": quiz_results[-5:]
    }

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Education System API is running",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "model_loaded": model is not None,
        "features": [
            "Personalized Assessment",
            "AI-Generated Learning Materials", 
            "Adaptive Quiz Generation",
            "Comprehensive Evaluation",
            "Learning Progress Reports",
            "AI Tutor Chatbot"
        ]
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "model_status": "loaded" if model is not None else "api_fallback",
        "device": str(device) if device else "unknown",
        "services": {
            "huggingface_model": "loaded" if model is not None else "api_fallback",
            "tavily_api": "connected" if tavily_client else "disabled"
        }
    }

if __name__ == "__main__":
    print(f"Starting server with model: {MODEL_NAME}")
    print(f"Model loaded locally: {model is not None}")
    uvicorn.run(app, host="0.0.0.0", port=8000)