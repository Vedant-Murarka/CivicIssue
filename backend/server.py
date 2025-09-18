from fastapi import FastAPI, APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
import bcrypt
import os
import logging
import uuid
import json
from pathlib import Path
import asyncio
from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType
import base64

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Settings
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES', 30))

# Security
security = HTTPBearer()

# FastAPI app
app = FastAPI(title="Smart Civic Issue Reporting Platform", version="1.0.0")
api_router = APIRouter(prefix="/api")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UserCreate(BaseModel):
    phone: str = Field(..., pattern=r'^\+?1?\d{9,15}$')
    full_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    phone: str
    password: str

class OTPVerification(BaseModel):
    phone: str
    otp: str

class IssueCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=255)
    description: str = Field(..., min_length=10, max_length=2000)
    category: str = Field(..., pattern=r'^(pothole|streetlight|garbage|water_leak|traffic|other)$')
    address: Optional[str] = Field(None, max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    priority: Optional[str] = Field('medium', pattern=r'^(low|medium|high|urgent)$')

class IssueResponse(BaseModel):
    id: str
    title: str
    description: str
    category: str
    status: str
    priority: str
    address: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    ai_classification: Optional[str]
    created_at: datetime
    updated_at: datetime
    reporter_name: str
    image_url: Optional[str]

class IssueUpdate(BaseModel):
    status: Optional[str] = Field(None, pattern=r'^(open|in_progress|resolved|closed)$')
    assignee_id: Optional[str] = None
    admin_notes: Optional[str] = None

# Auth functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = await db.users.find_one({"_id": user_id})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# AI Classification Service
class AIClassificationService:
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        
    async def classify_issue(self, title: str, description: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Classify civic issue using Gemini AI"""
        try:
            session_id = str(uuid.uuid4())
            
            chat = LlmChat(
                api_key=self.api_key,
                session_id=session_id,
                system_message="""You are an AI assistant that classifies civic issues reported by citizens. 
                Analyze the title, description, and image (if provided) to:
                1. Determine the main category: pothole, streetlight, garbage, water_leak, traffic, or other
                2. Assess priority level: low, medium, high, urgent
                3. Suggest which department should handle this: Public Works, Water Department, Traffic Management, Sanitation, or Other
                4. Provide a brief analysis of the issue
                
                Respond in JSON format:
                {
                    "category": "category_name",
                    "priority": "priority_level", 
                    "department": "department_name",
                    "analysis": "brief analysis of the issue",
                    "confidence": "high/medium/low"
                }"""
            ).with_model("gemini", "gemini-2.0-flash")
            
            # Prepare message
            prompt = f"Title: {title}\nDescription: {description}\n\nPlease classify this civic issue."
            
            file_contents = []
            if image_path and os.path.exists(image_path):
                file_contents = [FileContentWithMimeType(
                    file_path=image_path,
                    mime_type="image/jpeg"
                )]
            
            user_message = UserMessage(
                text=prompt,
                file_contents=file_contents if file_contents else None
            )
            
            response = await chat.send_message(user_message)
            
            # Parse JSON response
            try:
                classification = json.loads(response)
                return classification
            except json.JSONDecodeError:
                # Fallback classification
                return {
                    "category": "other",
                    "priority": "medium",
                    "department": "Public Works",
                    "analysis": "AI classification unavailable, manual review needed",
                    "confidence": "low"
                }
                
        except Exception as e:
            logger.error(f"AI classification error: {e}")
            return {
                "category": "other",
                "priority": "medium", 
                "department": "Public Works",
                "analysis": "AI classification failed, manual review needed",
                "confidence": "low"
            }

ai_service = AIClassificationService()

# Helper functions
async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    uploads_dir = Path("/app/backend/uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{file_id}.{file_extension}"
    file_path = uploads_dir / filename
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return str(file_path)

def generate_mock_otp() -> str:
    """Generate mock OTP for demo purposes"""
    return "123456"  # In production, use real SMS service

# API Endpoints

@api_router.post("/auth/register")
async def register_user(user_data: UserCreate):
    """Register new user with phone and password"""
    # Check if user already exists
    existing_user = await db.users.find_one({"phone": user_data.phone})
    if existing_user:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    # Hash password
    hashed_password = hash_password(user_data.password)
    
    # Create user document
    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "phone": user_data.phone,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "role": "citizen",
        "is_verified": False,
        "created_at": datetime.utcnow(),
        "otp": generate_mock_otp(),
        "otp_expires": datetime.utcnow() + timedelta(minutes=10)
    }
    
    await db.users.insert_one(user_doc)
    
    return {
        "message": "User registered successfully. Please verify with OTP.",
        "otp": generate_mock_otp(),  # In production, send via SMS
        "user_id": user_id
    }

@api_router.post("/auth/verify-otp")
async def verify_otp(otp_data: OTPVerification):
    """Verify OTP and activate account"""
    user = await db.users.find_one({"phone": otp_data.phone})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # In production, verify actual OTP
    if otp_data.otp == "123456":  # Mock OTP verification
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"is_verified": True, "otp": None, "otp_expires": None}}
        )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user["_id"], "phone": user["phone"], "role": user["role"]}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user["_id"],
                "phone": user["phone"],
                "full_name": user["full_name"],
                "role": user["role"]
            }
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid OTP")

@api_router.post("/auth/login")
async def login_user(login_data: UserLogin):
    """Login with phone and password"""
    user = await db.users.find_one({"phone": login_data.phone})
    if not user or not verify_password(login_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    
    if not user.get("is_verified", False):
        raise HTTPException(status_code=401, detail="Account not verified. Please verify with OTP.")
    
    access_token = create_access_token(
        data={"sub": user["_id"], "phone": user["phone"], "role": user["role"]}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["_id"],
            "phone": user["phone"],
            "full_name": user["full_name"],
            "role": user["role"]
        }
    }

@api_router.post("/issues", response_model=IssueResponse)
async def create_issue(
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    address: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    priority: Optional[str] = Form("medium"),
    image: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    """Create new civic issue with AI classification"""
    
    # Validate image if provided
    if image and not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save image if provided
    image_path = None
    if image:
        image_path = await save_uploaded_file(image)
    
    # AI Classification
    ai_classification = await ai_service.classify_issue(title, description, image_path)
    
    # Create issue document
    issue_id = str(uuid.uuid4())
    issue_doc = {
        "_id": issue_id,
        "title": title,
        "description": description,
        "category": ai_classification.get("category", category),
        "status": "open",
        "priority": ai_classification.get("priority", priority),
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "ai_classification": ai_classification,
        "reporter_id": current_user["_id"],
        "reporter_name": current_user["full_name"],
        "image_path": image_path,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    await db.issues.insert_one(issue_doc)
    
    return IssueResponse(
        id=issue_id,
        title=title,
        description=description,
        category=issue_doc["category"],
        status=issue_doc["status"],
        priority=issue_doc["priority"],
        address=address,
        latitude=latitude,
        longitude=longitude,
        ai_classification=json.dumps(ai_classification),
        created_at=issue_doc["created_at"],
        updated_at=issue_doc["updated_at"],
        reporter_name=current_user["full_name"],
        image_url=f"/api/images/{issue_id}" if image_path else None
    )

@api_router.get("/issues", response_model=List[IssueResponse])
async def get_issues(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """Get issues with filtering"""
    # Build query
    query = {}
    if category:
        query["category"] = category
    if status:
        query["status"] = status
    
    # If regular user, show only their issues
    if current_user.get("role") == "citizen":
        query["reporter_id"] = current_user["_id"]
    
    issues = await db.issues.find(query).sort("created_at", -1).skip(offset).limit(limit).to_list(limit)
    
    return [
        IssueResponse(
            id=issue["_id"],
            title=issue["title"],
            description=issue["description"],
            category=issue["category"],
            status=issue["status"],
            priority=issue["priority"],
            address=issue.get("address"),
            latitude=issue.get("latitude"),
            longitude=issue.get("longitude"),
            ai_classification=json.dumps(issue.get("ai_classification", {})),
            created_at=issue["created_at"],
            updated_at=issue["updated_at"],
            reporter_name=issue["reporter_name"],
            image_url=f"/api/images/{issue['_id']}" if issue.get("image_path") else None
        )
        for issue in issues
    ]

@api_router.put("/issues/{issue_id}")
async def update_issue(
    issue_id: str,
    update_data: IssueUpdate,
    current_user: dict = Depends(get_admin_user)
):
    """Update issue status (admin only)"""
    update_fields = {"updated_at": datetime.utcnow()}
    
    if update_data.status:
        update_fields["status"] = update_data.status
    if update_data.assignee_id:
        update_fields["assignee_id"] = update_data.assignee_id
    if update_data.admin_notes:
        update_fields["admin_notes"] = update_data.admin_notes
    
    result = await db.issues.update_one(
        {"_id": issue_id},
        {"$set": update_fields}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Issue not found")
    
    return {"message": "Issue updated successfully"}

@api_router.get("/admin/dashboard")
async def get_admin_dashboard(current_user: dict = Depends(get_admin_user)):
    """Get admin dashboard statistics"""
    total_issues = await db.issues.count_documents({})
    open_issues = await db.issues.count_documents({"status": "open"})
    in_progress = await db.issues.count_documents({"status": "in_progress"})
    resolved = await db.issues.count_documents({"status": "resolved"})
    
    # Category breakdown
    categories = await db.issues.aggregate([
        {"$group": {"_id": "$category", "count": {"$sum": 1}}}
    ]).to_list(None)
    
    # Priority breakdown
    priorities = await db.issues.aggregate([
        {"$group": {"_id": "$priority", "count": {"$sum": 1}}}
    ]).to_list(None)
    
    return {
        "total_issues": total_issues,
        "open_issues": open_issues,
        "in_progress": in_progress,
        "resolved": resolved,
        "categories": [{"category": cat["_id"], "count": cat["count"]} for cat in categories],
        "priorities": [{"priority": pri["_id"], "count": pri["count"]} for pri in priorities]
    }

@api_router.get("/heatmap-data")
async def get_heatmap_data(
    category: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user)
):
    """Get heatmap data for visualization"""
    # Build query
    query = {"latitude": {"$ne": None}, "longitude": {"$ne": None}}
    if category:
        query["category"] = category
    
    # Date filter
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    query["created_at"] = {"$gte": cutoff_date}
    
    issues = await db.issues.find(query).to_list(None)
    
    heatmap_data = []
    for issue in issues:
        heatmap_data.append({
            "lat": issue["latitude"],
            "lng": issue["longitude"],
            "weight": 1,
            "category": issue["category"],
            "title": issue["title"],
            "status": issue["status"]
        })
    
    return {"data": heatmap_data}

# Add user to admin role (helper endpoint for testing)
@api_router.post("/admin/create")
async def create_admin():
    """Create admin user for testing"""
    admin_phone = "+19999999999"
    existing_admin = await db.users.find_one({"phone": admin_phone})
    
    if not existing_admin:
        admin_id = str(uuid.uuid4())
        admin_doc = {
            "_id": admin_id,
            "phone": admin_phone,
            "full_name": "System Administrator",
            "hashed_password": hash_password("admin123"),
            "role": "admin",
            "is_verified": True,
            "created_at": datetime.utcnow()
        }
        await db.users.insert_one(admin_doc)
        return {"message": "Admin created", "phone": admin_phone, "password": "admin123"}
    else:
        return {"message": "Admin already exists", "phone": admin_phone}

# Include router
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)