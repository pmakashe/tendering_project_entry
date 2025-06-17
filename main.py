from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, cast, Date, and_, ForeignKey
from collections import defaultdict
from decimal import Decimal
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, Integer, String, Date, DateTime, Text, Numeric, select, Boolean, UniqueConstraint
from datetime import date, datetime, timedelta
from typing import Optional, List, Union, AsyncGenerator, Any
import os
from passlib.context import CryptContext

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database Configuration ---
Base = declarative_base()

# Load DATABASE_URL from environment (Render provides it)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:Pune123@localhost:5433/tendering_project_entry_db")

# Ensure asyncpg driver is used
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Add SSL mode for Render's PostgreSQL
if "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)  # echo=True for debug logs

# Create async session factory
async_session = async_sessionmaker(engine, class_=AsyncSession, autocommit=False, autoflush=False, expire_on_commit=False)

# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String(50), unique=True, nullable=False, index=True)
    company_name = Column(String(255))
    tender_entry_date = Column(DateTime, nullable=False)
    tender_status = Column(String(50))
    project_name = Column(String(500), nullable=False)
    name_of_client = Column(String(255))
    state = Column(String(100))
    project_type = Column(String(100))
    tender_id = Column(String(255))
    attch_tender_document_path = Column(String(500))
    tender_estimated_cost_inr = Column(Numeric(18, 2))
    completion_period_month = Column(Integer)
    tender_submission_date = Column(Date)
    tender_opening_date = Column(Date)
    pre_bid_meeting_date = Column(Date)
    clarifications_issued_date = Column(Date)
    corrigendum_received_date = Column(Date)
    emd_required_inr = Column(Numeric(18, 2))
    emd_paid_inr = Column(Numeric(18, 2))
    emd_instrument_type = Column(String(50))
    emd_bg_details_bg_number = Column(String(255))
    emd_bg_details_bank_name = Column(String(255))
    emd_bg_details_bg_expiry_date = Column(Date)
    emd_return_date = Column(Date)
    sd_required_percent = Column(Numeric(5, 2))
    sd_required_inr = Column(Numeric(18, 2))
    sd_instrument_type = Column(String(50))
    pbg_required_percent = Column(Numeric(5, 2))
    pbg_required_inr = Column(Numeric(18, 2))
    pbg_instrument_type = Column(String(50))
    competition_no_of_bidders = Column(Integer)
    final_status_of_tender = Column(String(50))
    remark = Column(Text)
    description = Column(Text)
    status = Column(String(50), default="New")
    tender_fees_paid_inr = Column(Numeric(18, 2))
    work_order_value_inr = Column(Numeric(18, 2))
    work_completion_date = Column(Date)
    contact_person = Column(String(255))
    contact_number = Column(String(50))
    contact_email = Column(String(255))
    __table_args__ = (UniqueConstraint('project_id', name='_project_id_uc'),)

class ExternalTender(Base):
    __tablename__ = "external_tenders"
    id = Column(Integer, primary_key=True, index=True)
    tender_id_external = Column(String(255), unique=True, nullable=False, comment="Tender ID from external portal")
    tender_ref_number = Column(String(255), comment="Tender Reference Number")
    tender_title = Column(String(500), nullable=False)
    department = Column(String(255))
    tender_type = Column(String(100))
    publication_date = Column(Date)
    bid_submission_end_date = Column(DateTime)
    emd_value = Column(Numeric(18, 2))
    estimated_cost = Column(Numeric(18, 2))
    city_location = Column(String(255))
    other_details = Column(Text)
    __table_args__ = (UniqueConstraint('tender_id_external', name='_tender_id_external_uc'),)

class DailyWorkEntry(Base):
    __tablename__ = "daily_work_entry"
    id = Column(Integer, primary_key=True, index=True)
    sr_no = Column(Integer)
    project_id = Column(String(50), ForeignKey('projects.project_id'), nullable=False)
    name_of_project = Column(String(255))
    activity_name = Column(String(255))
    activity_description = Column(Text)
    priority = Column(String(50))
    allocated_to = Column(String(255))
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    helped_by = Column(String(255))
    allocation_date = Column(Date)
    due_date = Column(Date)
    expected_time = Column(String(50))
    work_done_details = Column(Text)
    total_time = Column(String(50))
    completed_on = Column(Date)
    status_summary = Column(String(100))
    is_activity_overdue = Column(Boolean)
    delay_in_days = Column(Integer)
    due_today = Column(Boolean)
    on_time = Column(Boolean)
    open_duration_days = Column(Integer)
    remark = Column(Text)
    __table_args__ = (UniqueConstraint('sr_no', 'project_id', name='_sr_no_project_uc'),)

class Tender(Base):
    __tablename__ = "tenders"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(String)

# --- FastAPI App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIRECTORY = "/tmp/uploaded_documents"  # Use /tmp for Render's ephemeral filesystem
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- Database Session Dependency ---
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
        await session.close()

# --- Utility Functions ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def parse_date_or_none(date_string: Optional[str]) -> Optional[date]:
    if date_string:
        try:
            return datetime.strptime(date_string, '%Y-%m-%d').date()
        except ValueError:
            try:
                return datetime.strptime(date_string, '%d-%m-%Y').date()
            except ValueError:
                print(f"Warning: Could not parse date string '{date_string}'")
                return None
    return None

def parse_datetime_or_none(datetime_string: Optional[str]) -> Optional[datetime]:
    if datetime_string:
        try:
            return datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M')
        except ValueError:
            print(f"Warning: Could not parse datetime string '{datetime_string}'")
            return None
    return None

def parse_numeric_or_none(value_str: Optional[str]) -> Optional[Decimal]:
    if value_str:
        try:
            cleaned_value = value_str.replace(',', '')
            return Decimal(cleaned_value)
        except (ValueError, TypeError):
            print(f"Warning: Could not parse numeric string '{value_str}'")
            return None
    return None

def parse_int_or_none(value_str: Optional[str]) -> Optional[int]:
    if value_str:
        try:
            cleaned_value = value_str.replace(',', '')
            return int(cleaned_value)
        except (ValueError, TypeError):
            print(f"Warning: Could not parse integer string '{value_str}'")
            return None
    return None

def parse_boolean_or_none(value_str: Optional[Union[str, bool]]) -> Optional[bool]:
    if isinstance(value_str, bool):
        return value_str
    if isinstance(value_str, str):
        if value_str.lower() in ('yes', 'true', '1'):
            return True
        if value_str.lower() in ('no', 'false', '0'):
            return False
    return None

# --- Startup Event for Schema Initialization ---
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all)
            print("Database schema initialized successfully")
        except Exception as e:
            print(f"Failed to initialize database schema: {e}")
            raise

    # Test database connection
    async with async_session() as session:
        try:
            await session.execute("SELECT 1")
            print("Database connection successful")
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("signin.html", {"request": request})

@app.get("/signin", response_class=HTMLResponse)
async def signin_page(request: Request):
    return templates.TemplateResponse("signin.html", {"request": request})

@app.post("/signin", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.username == username))
    user = result.scalars().first()

    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            "signin.html",
            {"request": request, "error_message": "Invalid username or password"},
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    response = RedirectResponse(url="/landing_page", status_code=status.HTTP_303_SEE_OTHER)
    return response

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup", response_class=HTMLResponse)
async def register_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    email: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error_message": "Passwords do not match"},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    result = await db.execute(select(User).filter(User.username == username))
    existing_user = result.scalars().first()
    if existing_user:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error_message": "Username already registered"},
            status_code=status.HTTP_409_CONFLICT
        )
    
    if email:
        result_email = await db.execute(select(User).filter(User.email == email))
        existing_email_user = result_email.scalars().first()
        if existing_email_user:
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "error_message": "Email already registered"},
                status_code=status.HTTP_409_CONFLICT
            )

    hashed_password = get_password_hash(password)
    new_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
        return RedirectResponse(url="/signin", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

@app.get("/landing_page", response_class=HTMLResponse)
async def landing_page_route(request: Request):
    current_year = datetime.now().year
    return templates.TemplateResponse("landing_page.html", {"request": request, "current_year": current_year})

@app.get("/project_dashboard", response_class=HTMLResponse)
async def project_dashboard_page(request: Request):
    current_year = datetime.now().year
    return templates.TemplateResponse("project_dashboard.html", {"request": request, "current_year": current_year})

@app.get("/api/dashboard_summary", response_class=JSONResponse)
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    toBeSubmitted_result = await db.execute(
        select(func.count(Project.id), func.sum(Project.tender_estimated_cost_inr))
        .filter(Project.tender_status == "Yet to Submit")
    )
    toBeSubmitted_count, totalToBeSubmittedValue = toBeSubmitted_result.first()
    
    submitted_result = await db.execute(
        select(func.count(Project.id), func.sum(Project.tender_estimated_cost_inr))
        .filter(Project.tender_status == "Submitted")
    )
    submitted_count, totalSubmittedValue = submitted_result.first()

    awarded_result = await db.execute(
        select(func.count(Project.id), func.sum(Project.work_order_value_inr))
        .filter(Project.tender_status == "Awarded")
    )
    awarded_count, totalAwardedValue = awarded_result.first()

    today = date.today()
    in_30_days = today + timedelta(days=30)
    upcomingDeadlines_result = await db.execute(
        select(func.count(Project.id))
        .filter(Project.tender_submission_date.between(today, in_30_days))
    )
    upcomingDeadlines_count = upcomingDeadlines_result.scalar_one_or_none()

    summary_data = {
        "toBeSubmitted": toBeSubmitted_count if toBeSubmitted_count is not None else 0,
        "totalToBeSubmittedValue": float(totalToBeSubmittedValue) if totalToBeSubmittedValue is not None else 0.0,
        "submitted": submitted_count if submitted_count is not None else 0,
        "totalSubmittedValue": float(totalSubmittedValue) if totalSubmittedValue is not None else 0.0,
        "awarded": awarded_count if awarded_count is not None else 0,
        "totalAwardedValue": float(totalAwardedValue) if totalAwardedValue is not None else 0.0,
        "upcomingDeadlines": upcomingDeadlines_count if upcomingDeadlines_count is not None else 0
    }
    return JSONResponse(content=summary_data)

@app.get("/projects_list", response_class=HTMLResponse)
async def projects_page(request: Request, db: AsyncSession = Depends(get_db)):
    return templates.TemplateResponse("projects.html", {"request": request})

@app.get("/api/projects/", response_class=JSONResponse)
async def get_all_projects_api(
    db: AsyncSession = Depends(get_db),
    search_query: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    client: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[str] = Query(None)
):
    print(f"--- API DEBUG (get_all_projects_api) ---")
    print(f"Received filters: status='{status}', client='{client}', search_query='{search_query}', start_date='{start_date}', end_date='{end_date}'")

    query = select(Project)

    if search_query:
        search_pattern = f"%{search_query.lower()}%"
        query = query.filter(
            (func.lower(Project.project_id).like(search_pattern)) |
            (func.lower(Project.project_name).like(search_pattern)) |
            (func.lower(Project.company_name).like(search_pattern)) |
            (func.lower(Project.name_of_client).like(search_pattern))
        )
    if status:
        query = query.filter(func.lower(Project.tender_status) == status.lower().strip())
    if client:
        query = query.filter(func.lower(Project.name_of_client) == client.lower().strip())
    if start_date:
        query = query.filter(Project.tender_entry_date >= start_date)
    if end_date:
        parsed_end_date = parse_date_or_none(end_date)
        if parsed_end_date:
            query = query.filter(Project.tender_entry_date <= parsed_end_date + timedelta(days=1))

    result = await db.execute(query)
    projects = result.scalars().all()
    
    print(f"SQL Query (might not show exact parameters): {query}")
    print(f"Number of projects found by query: {len(projects)}")
    print(f"--- END DEBUG (get_all_projects_api) ---")

    projects_data = []
    for project in projects:
        project_dict = {}
        for col in Project.__table__.columns:
            value = getattr(project, col.name)
            if isinstance(value, Decimal):
                project_dict[col.name] = str(value)
            elif isinstance(value, (date, datetime)):
                if col.name == "tender_entry_date":
                    project_dict[col.name] = value.isoformat(timespec='minutes') if value is not None else None
                else:
                    project_dict[col.name] = value.isoformat() if value is not None else None
            else:
                project_dict[col.name] = value
        
        if project_dict.get("attch_tender_document_path") and os.path.exists(project_dict["attch_tender_document_path"]):
            project_dict["attch_tender_document_path"] = f"/static/{os.path.basename(project_dict['attch_tender_document_path'])}"
        
        project_dict['tender_name'] = project_dict.get('project_name')
        projects_data.append(project_dict)
    
    return JSONResponse(content=projects_data)

@app.get("/user_management", response_class=HTMLResponse)
async def user_management_page(request: Request):
    return templates.TemplateResponse("user_management.html", {"request": request})

@app.get("/api/users/", response_class=JSONResponse)
async def get_all_users_api(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    users_data = []
    for user in users:
        users_data.append({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_active": user.is_active
        })
    return JSONResponse(content=users_data)

@app.post("/api/users/", response_class=JSONResponse)
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    email: Optional[str] = Form(None),
    is_active: bool = Form(True),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).filter(User.username == username))
    if result.scalars().first():
        raise HTTPException(status_code=409, detail="Username already registered")
    
    if email:
        result_email = await db.execute(select(User).filter(User.email == email))
        if result_email.scalars().first():
            raise HTTPException(status_code=409, detail="Email already registered")

    hashed_password = get_password_hash(password)
    new_user = User(username=username, email=email, hashed_password=hashed_password, is_active=is_active)
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
        return JSONResponse(content={"message": f"User '{new_user.username}' created successfully!"})
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {e}")

@app.get("/api/users/{user_id}", response_class=JSONResponse)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse(content={
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active
    })

@app.put("/api/users/{user_id}", response_class=JSONResponse)
async def update_user(
    user_id: int,
    username: str = Form(...),
    email: Optional[str] = Form(None),
    is_active: bool = Form(True),
    password: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.username != username:
        existing_username = await db.execute(select(User).filter(User.username == username))
        if existing_username.scalars().first():
            raise HTTPException(status_code=409, detail="Username already taken.")
    
    if email and user.email != email:
        existing_email = await db.execute(select(User).filter(User.email == email))
        if existing_email.scalars().first():
            raise HTTPException(status_code=409, detail="Email already registered.")

    user.username = username
    user.email = email
    user.is_active = is_active
    if password:
        user.hashed_password = get_password_hash(password)

    try:
        await db.commit()
        await db.refresh(user)
        return JSONResponse(content={"message": f"User '{user.username}' updated successfully!"})
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update user: {e}")

@app.delete("/api/users/{user_id}", response_class=JSONResponse)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await db.delete(user)
    try:
        await db.commit()
        return JSONResponse(content={"message": f"User '{user.username}' deleted successfully!"})
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {e}")

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    return templates.TemplateResponse("reports.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/tender_search", response_class=HTMLResponse)
async def tender_search_page(request: Request):
    return templates.TemplateResponse("tender_search.html", {"request": request})

@app.get("/api/external_tenders", response_class=JSONResponse)
async def get_external_tenders_api(
    search_query: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    tender_type: Optional[str] = Query(None),
    min_emd: Optional[Decimal] = Query(None),
    max_emd: Optional[Decimal] = Query(None),
    min_est_cost: Optional[Decimal] = Query(None),
    max_est_cost: Optional[Decimal] = Query(None),
):
    dummy_tenders = [
        {
            "tender_id_external": "2025_PW_12345",
            "tender_ref_number": "MahaTender/2025/XYZ",
            "tender_title": "Construction of Bridge in Pune",
            "department": "Public Works Department",
            "tender_type": "Open",
            "publication_date": "2025-06-01",
            "bid_submission_end_date": "2025-06-30T17:00:00",
            "emd_value": "50000.00",
            "estimated_cost": "15000000.00",
            "city_location": "Pune",
            "other_details": "Road bridge construction, 2 years completion."
        },
        {
            "tender_id_external": "2025_WS_67890",
            "tender_ref_number": "MahaTender/2025/ABC",
            "tender_title": "Water Supply Pipeline Installation Mumbai",
            "department": "Water Resources Department",
            "tender_type": "Open",
            "publication_date": "2025-05-20",
            "bid_submission_end_date": "2025-06-25T15:30:00",
            "emd_value": "25000.00",
            "estimated_cost": "5000000.00",
            "city_location": "Mumbai",
            "other_details": "Drinking water supply project, Phase I."
        },
        {
            "tender_id_external": "2025_EDU_11223",
            "tender_ref_number": "MahaTender/2025/DEF",
            "tender_title": "IT Equipment Procurement for Schools in Nagpur",
            "department": "Education Department",
            "tender_type": "Limited",
            "publication_date": "2025-06-10",
            "bid_submission_end_date": "2025-07-15T12:00:00",
            "emd_value": "10000.00",
            "estimated_cost": "2000000.00",
            "city_location": "Nagpur",
            "other_details": "Supply of computers and network devices."
        },
        {
            "tender_id_external": "2025_HEALTH_45678",
            "tender_ref_number": "MahaTender/2025/GHI",
            "tender_title": "Medical Supplies for Primary Health Centers, Nashik",
            "department": "Health Department",
            "tender_type": "Open",
            "publication_date": "2025-06-05",
            "bid_submission_end_date": "2025-07-01T10:00:00",
            "emd_value": "15000.00",
            "estimated_cost": "3000000.00",
            "city_location": "Nashik",
            "other_details": "Procurement of essential medical items."
        },
        {
            "tender_id_external": "2025_RURAL_98765",
            "tender_ref_number": "MahaTender/2025/JKL",
            "tender_title": "Rural Road Construction, Satara District",
            "department": "Rural Development Department",
            "tender_type": "Open",
            "publication_date": "2025-05-28",
            "bid_submission_end_date": "2025-06-28T16:00:00",
            "emd_value": "40000.00",
            "estimated_cost": "10000000.00",
            "city_location": "Satara",
            "other_details": "Construction of village link roads."
        }
    ]

    filtered_tenders = []
    for tender in dummy_tenders:
        match = True
        if search_query:
            query_lower = search_query.lower()
            if not (query_lower in tender["tender_id_external"].lower() or \
                    query_lower in tender["tender_ref_number"].lower() or \
                    query_lower in tender["tender_title"].lower() or \
                    query_lower in tender["other_details"].lower()):
                match = False
        if city and tender["city_location"] != city:
            match = False
        if department and tender["department"] != department:
            match = False
        if tender_type and tender["tender_type"] != tender_type:
            match = False
        if min_emd is not None and Decimal(tender["emd_value"]) < min_emd:
            match = False
        if max_emd is not None and Decimal(tender["emd_value"]) > max_emd:
            match = False
        if min_est_cost is not None and Decimal(tender["estimated_cost"]) < min_est_cost:
            match = False
        if max_est_cost is not None and Decimal(tender["estimated_cost"]) > max_est_cost:
            match = False

        if match:
            filtered_tenders.append(tender)

    return JSONResponse(content=filtered_tenders)

@app.post("/api/projects/add_external", response_class=JSONResponse)
async def add_external_tender_to_internal_projects(
    request: Request,
    tender_id_external: str = Form(...),
    tender_ref_number: Optional[str] = Form(None),
    tender_title: str = Form(...),
    department: Optional[str] = Form(None),
    tender_type: Optional[str] = Form(None),
    publication_date: Optional[str] = Form(None),
    bid_submission_end_date: Optional[str] = Form(None),
    emd_value: Optional[str] = Form(None),
    estimated_cost: Optional[str] = Form(None),
    city_location: Optional[str] = Form(None),
    other_details: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    current_year = datetime.now().year
    result = await db.execute(
        select(Project.project_id)
        .filter(Project.project_id.like(f"PID-{current_year}-%"))
        .order_by(Project.project_id.desc())
        .limit(1)
    )
    last_project_id_internal = result.scalars().first()

    next_sequence = 1
    if last_project_id_internal:
        try:
            last_sequence = int(last_project_id_internal.split('-')[-1])
            next_sequence = last_sequence + 1
        except ValueError:
            next_sequence = 1
    else:
        next_sequence = 1

    new_internal_project_id = f"PID-{current_year}-{next_sequence:03d}"

    existing_project = await db.execute(select(Project).filter(Project.tender_id == tender_id_external))
    if existing_project.scalars().first():
        raise HTTPException(status_code=409, detail=f"A project with external ID '{tender_id_external}' already exists in internal projects.")

    new_project = Project(
        project_id=new_internal_project_id,
        project_name=tender_title,
        company_name="External Source (MahaTenders)",
        tender_entry_date=datetime.now(),
        tender_status="Yet to Submit",
        name_of_client=department,
        state="Maharashtra",
        project_type=tender_type,
        tender_id=tender_id_external,
        attch_tender_document_path=None,
        tender_estimated_cost_inr=parse_numeric_or_none(estimated_cost),
        completion_period_month=0,
        tender_submission_date=parse_date_or_none(bid_submission_end_date),
        tender_opening_date=parse_date_or_none(publication_date),
        pre_bid_meeting_date=None,
        clarifications_issued_date=None,
        corrigendum_received_date=None,
        emd_required_inr=parse_numeric_or_none(emd_value),
        emd_paid_inr=parse_numeric_or_none(emd_value),
        emd_instrument_type="Online",
        emd_bg_details_bg_number=None,
        emd_bg_details_bank_name=None,
        emd_bg_details_bg_expiry_date=None,
        emd_return_date=None,
        sd_required_percent=None,
        sd_required_inr=None,
        sd_instrument_type=None,
        pbg_required_percent=None,
        pbg_required_inr=None,
        pbg_instrument_type=None,
        competition_no_of_bidders=None,
        final_status_of_tender="To Be Submitted",
        remark=f"Imported from external source. Original Ref: {tender_ref_number}",
        description=other_details,
        status="New",
        tender_fees_paid_inr=None,
        work_order_value_inr=None,
        work_completion_date=None,
        contact_person=None,
        contact_number=None,
        contact_email=None,
    )

    db.add(new_project)
    try:
        await db.commit()
        await db.refresh(new_project)
        return JSONResponse(content={"message": f"Tender '{tender_title}' (ID: {tender_id_external}) added to internal projects successfully with Project ID: {new_internal_project_id}!"})
    except Exception as e:
        await db.rollback()
        if "UniqueConstraint" in str(e):
            raise HTTPException(status_code=409, detail=f"A project with external ID '{tender_id_external}' or similar Project ID already exists.")
        raise HTTPException(status_code=500, detail=f"Failed to add tender to internal projects: {e}")

@app.get("/new_project", response_class=HTMLResponse)
async def new_project_hub_page(request: Request):
    return templates.TemplateResponse("new_project_hub.html", {"request": request})

@app.get("/new_tender_entry_form", response_class=HTMLResponse)
async def new_tender_entry_form_page(request: Request, db: AsyncSession = Depends(get_db)):
    current_year = datetime.now().year
    result = await db.execute(
        select(Project.project_id)
        .filter(Project.project_id.like(f"PID-{current_year}-%"))
        .order_by(Project.project_id.desc())
        .limit(1)
    )
    last_project_id = result.scalars().first()

    if last_project_id:
        try:
            last_sequence = int(last_project_id.split('-')[-1])
            next_sequence = last_sequence + 1
        except ValueError:
            next_sequence = 1
    else:
        next_sequence = 1

    next_project_id = f"PID-{current_year}-{next_sequence:03d}"

    current_datetime_str = datetime.now().isoformat(timespec='minutes')

    return templates.TemplateResponse(
        "new_tender_entry_form.html",
        {"request": request, "next_project_id": next_project_id, "current_datetime_str": current_datetime_str},
        status_code=status.HTTP_200_OK
    )

@app.get("/daily_work_entry_form", response_class=HTMLResponse)
async def daily_work_entry_form_page(request: Request):
    return templates.TemplateResponse("daily_work_entry_form.html", {"request": request})

@app.post("/projects/", response_class=RedirectResponse)
async def create_project(
    request: Request,
    project_id: str = Form(...),
    project_name: str = Form(..., alias="tender_name"),
    company_name: str = Form(...),
    tender_status: str = Form(...),
    name_of_client: str = Form(...),
    state: str = Form(...),
    project_type: str = Form(...),
    tender_id: Optional[str] = Form(None),
    attch_tender_document: Optional[UploadFile] = File(None),
    tender_estimated_cost_inr_str: str = Form(..., alias="tender_estimated_cost"),
    completion_period_month_str: Optional[str] = Form(None),
    tender_submission_date_str: str = Form(...),
    tender_opening_date_str: Optional[str] = Form(None),
    pre_bid_meeting_date_str: Optional[str] = Form(None),
    clarifications_issued_date_str: Optional[str] = Form(None),
    corrigendum_received_date_str: Optional[str] = Form(None),
    emd_required_inr_str: Optional[str] = Form(None),
    emd_paid_inr_str: Optional[str] = Form(None),
    emd_instrument_type: Optional[str] = Form(None),
    emd_bg_details_bg_number: Optional[str] = Form(None),
    emd_bg_details_bank_name: Optional[str] = Form(None),
    emd_bg_details_bg_expiry_date_str: Optional[str] = Form(None),
    emd_return_date_str: Optional[str] = Form(None),
    sd_required_percent_str: Optional[str] = Form(None),
    sd_required_inr_str: Optional[str] = Form(None),
    sd_instrument_type: Optional[str] = Form(None),
    pbg_required_percent_str: Optional[str] = Form(None),
    pbg_required_inr_str: Optional[str] = Form(None),
    pbg_instrument_type: Optional[str] = Form(None),
    competition_no_of_bidders_str: Optional[str] = Form(None),
    final_status_of_tender: Optional[str] = Form(None),
    remark: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    status: str = Form("New"),
    tender_fees_paid_inr_str: Optional[str] = Form(None),
    work_order_value_inr_str: Optional[str] = Form(None),
    work_completion_date_str: Optional[str] = Form(None),
    contact_person: Optional[str] = Form(None),
    contact_number: Optional[str] = Form(None),
    contact_email: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    document_path = None
    if attch_tender_document and attch_tender_document.filename:
        file_location = os.path.join(UPLOAD_DIRECTORY, attch_tender_document.filename)
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        try:
            with open(file_location, "wb+") as file_object:
                file_object.write(await attch_tender_document.read())
            document_path = file_location
        except Exception as e:
            print(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save document: {e}")

    tender_entry_date = datetime.now()

    new_project = Project(
        project_id=project_id,
        project_name=project_name,
        company_name=company_name,
        tender_entry_date=tender_entry_date,
        tender_status=tender_status,
        name_of_client=name_of_client,
        state=state,
        project_type=project_type,
        tender_id=tender_id,
        attch_tender_document_path=document_path,
        tender_estimated_cost_inr=parse_numeric_or_none(tender_estimated_cost_inr_str),
        completion_period_month=parse_int_or_none(completion_period_month_str),
        tender_submission_date=parse_date_or_none(tender_submission_date_str),
        tender_opening_date=parse_date_or_none(tender_opening_date_str),
        pre_bid_meeting_date=parse_date_or_none(pre_bid_meeting_date_str),
        clarifications_issued_date=parse_date_or_none(clarifications_issued_date_str),
        corrigendum_received_date=parse_date_or_none(corrigendum_received_date_str),
        emd_required_inr=parse_numeric_or_none(emd_required_inr_str),
        emd_paid_inr=parse_numeric_or_none(emd_paid_inr_str),
        emd_instrument_type=emd_instrument_type,
        emd_bg_details_bg_number=emd_bg_details_bg_number,
        emd_bg_details_bank_name=emd_bg_details_bank_name,
        emd_bg_details_bg_expiry_date=parse_date_or_none(emd_bg_details_bg_expiry_date_str),
        emd_return_date=parse_date_or_none(emd_return_date_str),
        sd_required_percent=parse_numeric_or_none(sd_required_percent_str),
        sd_required_inr=parse_numeric_or_none(sd_required_inr_str),
        sd_instrument_type=sd_instrument_type,
        pbg_required_percent=parse_numeric_or_none(pbg_required_percent_str),
        pbg_required_inr=parse_numeric_or_none(pbg_required_inr_str),
        pbg_instrument_type=pbg_instrument_type,
        competition_no_of_bidders=parse_int_or_none(competition_no_of_bidders_str),
        final_status_of_tender=final_status_of_tender,
        remark=remark,
        description=description,
        status=status,
        tender_fees_paid_inr=parse_numeric_or_none(tender_fees_paid_inr_str),
        work_order_value_inr=parse_numeric_or_none(work_order_value_inr_str),
        work_completion_date=parse_date_or_none(work_completion_date_str),
        contact_person=contact_person,
        contact_number=contact_number,
        contact_email=contact_email,
    )

    db.add(new_project)
    try:
        await db.commit()
        await db.refresh(new_project)
        return RedirectResponse(url="/project_success", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        await db.rollback()
        print(f"Error creating project: {e}")
        current_year = datetime.now().year
        result = await db.execute(
            select(Project.project_id)
            .filter(Project.project_id.like(f"PID-{current_year}-%"))
            .order_by(Project.project_id.desc())
            .limit(1)
        )
        last_project_id = result.scalars().first()
        next_sequence = 1
        if last_project_id:
            try:
                last_sequence = int(last_project_id.split('-')[-1])
                next_sequence = last_sequence + 1
            except ValueError:
                pass
        next_project_id = f"PID-{current_year}-{next_sequence:03d}"
        current_datetime_str = datetime.now().isoformat(timespec='minutes')

        return templates.TemplateResponse(
            "new_tender_entry_form.html",
            {"request": request, "next_project_id": next_project_id, "current_datetime_str": current_datetime_str, "error_message": f"Failed to create project: {e}"},
            status_code=status.HTTP_400_BAD_REQUEST
        )

@app.get("/project_success", response_class=HTMLResponse)
async def project_success_page(request: Request):
    return templates.TemplateResponse("project_success.html", {"request": request, "current_year": datetime.now().year})

@app.get("/edit_project/{project_id}", response_class=HTMLResponse)
async def edit_project_page(project_id: str, request: Request):
    return templates.TemplateResponse("edit_project.html", {"request": request, "project_id": project_id})

@app.get("/api/projects/{project_id}", response_class=JSONResponse)
async def get_project_details_api(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).filter(Project.project_id == project_id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_dict = {}
    for col in Project.__table__.columns:
        value = getattr(project, col.name)
        if isinstance(value, Decimal):
            project_dict[col.name] = str(value)
        elif isinstance(value, (date, datetime)):
            project_dict[col.name] = value.isoformat() if value is not None else None
        else:
            project_dict[col.name] = value
    
    if project_dict.get("attch_tender_document_path") and os.path.exists(project_dict["attch_tender_document_path"]):
        project_dict["attch_tender_document_path"] = f"/static/{os.path.basename(project_dict['attch_tender_document_path'])}"

    project_dict['tender_name'] = project_dict.get('project_name')
    project_dict['ref_tender_id'] = project_dict.get('tender_id')
    project_dict['notes'] = project_dict.get('description')
    project_dict['tender_type'] = project_dict.get('project_type')

    return JSONResponse(content=project_dict)

@app.put("/api/projects/edit/{project_id}", response_class=JSONResponse)
async def update_project(
    project_id: str,
    project_name: str = Form(..., alias="tender_name"),
    company_name: str = Form(...),
    tender_status: str = Form(...),
    name_of_client: str = Form(...),
    state: str = Form(...),
    project_type: str = Form(..., alias="tender_type"),
    tender_id: Optional[str] = Form(None, alias="ref_tender_id"),
    attch_tender_document_path: Optional[UploadFile] = File(None),
    tender_estimated_cost_inr_str: Optional[str] = Form(None, alias="tender_estimated_cost_inr"),
    completion_period_month_str: Optional[str] = Form(None),
    tender_submission_date_str: Optional[str] = Form(None),
    tender_opening_date_str: Optional[str] = Form(None),
    pre_bid_meeting_date_str: Optional[str] = Form(None),
    clarifications_issued_date_str: Optional[str] = Form(None),
    corrigendum_received_date_str: Optional[str] = Form(None),
    emd_required_inr_str: Optional[str] = Form(None),
    emd_paid_inr_str: Optional[str] = Form(None),
    emd_instrument_type: Optional[str] = Form(None),
    emd_bg_details_bg_number: Optional[str] = Form(None),
    emd_bg_details_bank_name: Optional[str] = Form(None),
    emd_bg_details_bg_expiry_date_str: Optional[str] = Form(None),
    emd_return_date_str: Optional[str] = Form(None),
    sd_required_percent_str: Optional[str] = Form(None),
    sd_required_inr_str: Optional[str] = Form(None),
    sd_instrument_type: Optional[str] = Form(None),
    pbg_required_percent_str: Optional[str] = Form(None),
    pbg_required_inr_str: Optional[str] = Form(None),
    pbg_instrument_type: Optional[str] = Form(None),
    competition_no_of_bidders_str: Optional[str] = Form(None),
    final_status_of_tender: Optional[str] = Form(None),
    remark: Optional[str] = Form(None),
    description: Optional[str] = Form(None, alias="notes"),
    status: Optional[str] = Form(None),
    tender_fees_paid_inr_str: Optional[str] = Form(None),
    work_order_value_inr_str: Optional[str] = Form(None),
    work_completion_date_str: Optional[str] = Form(None),
    contact_person: Optional[str] = Form(None),
    contact_number: Optional[str] = Form(None),
    contact_email: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Project).filter(Project.project_id == project_id))
    project = result.scalars().first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if attch_tender_document_path and attch_tender_document_path.filename:
        file_location = os.path.join(UPLOAD_DIRECTORY, attch_tender_document_path.filename)
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        try:
            with open(file_location, "wb+") as file_object:
                file_object.write(await attch_tender_document_path.read())
            project.attch_tender_document_path = file_location
        except Exception as e:
            print(f"Error saving uploaded file for update: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save new document: {e}")
    
    project.project_name = project_name
    project.company_name = company_name
    project.tender_status = tender_status
    project.name_of_client = name_of_client
    if state is not None and state != "": project.state = state
    project.project_type = project_type
    if tender_id is not None: project.tender_id = tender_id
    
    project.tender_estimated_cost_inr = parse_numeric_or_none(tender_estimated_cost_inr_str)
    project.completion_period_month = parse_int_or_none(completion_period_month_str)
    project.tender_submission_date = parse_date_or_none(tender_submission_date_str)
    project.tender_opening_date = parse_date_or_none(tender_opening_date_str)
    project.pre_bid_meeting_date = parse_date_or_none(pre_bid_meeting_date_str)
    project.clarifications_issued_date = parse_date_or_none(clarifications_issued_date_str)
    project.corrigendum_received_date = parse_date_or_none(corrigendum_received_date_str)
    project.emd_required_inr = parse_numeric_or_none(emd_required_inr_str)
    project.emd_paid_inr = parse_numeric_or_none(emd_paid_inr_str)
    if emd_instrument_type is not None: project.emd_instrument_type = emd_instrument_type
    if emd_bg_details_bg_number is not None: project.emd_bg_details_bg_number = emd_bg_details_bg_number
    if emd_bg_details_bank_name is not None: project.emd_bg_details_bank_name = emd_bg_details_bank_name
    project.emd_bg_details_bg_expiry_date = parse_date_or_none(emd_bg_details_bg_expiry_date_str)
    project.emd_return_date = parse_date_or_none(emd_return_date_str)
    project.sd_required_percent = parse_numeric_or_none(sd_required_percent_str)
    project.sd_required_inr = parse_numeric_or_none(sd_required_inr_str)
    if sd_instrument_type is not None: project.sd_instrument_type = sd_instrument_type
    project.pbg_required_percent = parse_numeric_or_none(pbg_required_percent_str)
    project.pbg_required_inr = parse_numeric_or_none(pbg_required_inr_str)
    if pbg_instrument_type is not None: project.pbg_instrument_type = pbg_instrument_type
    project.competition_no_of_bidders = parse_int_or_none(competition_no_of_bidders_str)
    if final_status_of_tender is not None: project.final_status_of_tender = final_status_of_tender
    if remark is not None: project.remark = remark
    if description is not None: project.description = description
    if status is not None: project.status = status

    project.tender_fees_paid_inr = parse_numeric_or_none(tender_fees_paid_inr_str)
    project.work_order_value_inr = parse_numeric_or_none(work_order_value_inr_str)
    project.work_completion_date = parse_date_or_none(work_completion_date_str)
    if contact_person is not None: project.contact_person = contact_person
    if contact_number is not None: project.contact_number = contact_number
    if contact_email is not None: project.contact_email = contact_email
    
    try:
        await db.commit()
        await db.refresh(project)
        return JSONResponse(content={"message": "Project updated successfully!"})
    except Exception as e:
        await db.rollback()
        print(f"Error updating project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update project: {e}")

@app.put("/api/projects/{project_id}", response_class=JSONResponse)
async def update_project_status(
    project_id: str,
    tender_status: str,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Project).filter(Project.project_id == project_id))
    project = result.scalars().first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.tender_status = tender_status
    try:
        await db.commit()
        await db.refresh(project)
        return JSONResponse(content={"message": f"Project status for {project_id} updated to {tender_status}"})
    except Exception as e:
        await db.rollback()
        print(f"Error updating project status {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update project status: {e}")

@app.get("/daily_work_status_dashboard", response_class=HTMLResponse)
async def daily_work_status_dashboard_page(request: Request):
    return templates.TemplateResponse("daily_work_status_dashboard.html", {"request": request})

@app.get("/api/daily_work_summary", response_class=JSONResponse)
async def get_daily_work_summary(
    request: Request,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    try:
        today = date.today()
        query = select(
            DailyWorkEntry.project_id,
            DailyWorkEntry.name_of_project,
            DailyWorkEntry.status_summary,
            DailyWorkEntry.due_date,
            DailyWorkEntry.completed_on,
            DailyWorkEntry.is_activity_overdue,
            DailyWorkEntry.delay_in_days,
            DailyWorkEntry.due_today,
            DailyWorkEntry.on_time,
            DailyWorkEntry.open_duration_days,
            DailyWorkEntry.allocated_to,
            User.username
        ).outerjoin(User, DailyWorkEntry.user_id == User.id)

        if start_date:
            query = query.filter(DailyWorkEntry.allocation_date >= start_date)
        if end_date:
            query = query.filter(DailyWorkEntry.allocation_date <= end_date)

        result = await db.execute(query)
        daily_entries = result.fetchall()

        total_completed = 0
        total_not_started = 0
        total_overdue = 0
        total_due_today = 0
        total_delay_days = 0
        total_open_duration = 0
        total_on_time = 0
        total_tasks = len(daily_entries) if daily_entries else 0
        overdue_by_person = defaultdict(int)
        overdue_by_project = defaultdict(int)
        status_by_person = defaultdict(lambda: {"Completed": 0, "In Progress": 0, "Not Started": 0})
        tasks_by_project = defaultdict(lambda: {"Completed": 0, "In Progress": 0, "Not Started": 0})

        for entry in daily_entries:
            entry_data = entry._mapping if hasattr(entry, '_mapping') else entry
            if entry_data.status_summary == "Completed":
                total_completed += 1
            elif entry_data.status_summary == "Not Started":
                total_not_started += 1
            if entry_data.is_activity_overdue:
                total_overdue += 1
                overdue_by_person[entry_data.allocated_to or 'Unassigned'] += 1
                overdue_by_project[entry_data.name_of_project or 'N/A'] += 1
            if entry_data.due_today:
                total_due_today += 1
            if entry_data.on_time:
                total_on_time += 1
            if entry_data.delay_in_days:
                total_delay_days += entry_data.delay_in_days or 0
            if entry_data.open_duration_days:
                total_open_duration += entry_data.open_duration_days or 0
            status_by_person[entry_data.allocated_to or 'Unassigned'][entry_data.status_summary or 'Not Started'] += 1
            tasks_by_project[entry_data.name_of_project or 'N/A'][entry_data.status_summary or 'Not Started'] += 1

        avg_delay = total_delay_days / total_tasks if total_tasks > 0 else 0
        avg_open_duration = total_open_duration / total_tasks if total_tasks > 0 else 0
        on_time_percentage = (total_on_time / total_tasks * 100) if total_tasks > 0 else 0

        response_data = {
            "total_completed": total_completed,
            "total_not_started": total_not_started,
            "total_overdue": total_overdue,
            "total_due_today": total_due_today,
            "average_delay_days": float(avg_delay),
            "average_open_duration_days": float(avg_open_duration),
            "on_time_percentage": float(on_time_percentage),
            "overdue_by_person": dict(overdue_by_person),
            "overdue_by_project": dict(overdue_by_project),
            "status_by_person": {k: dict(v) for k, v in status_by_person.items()},
            "tasks_by_project": {k: dict(v) for k, v in tasks_by_project.items()},
            "raw_entries": [
                {
                    "project_id": entry_data.project_id,
                    "name_of_project": entry_data.name_of_project or 'N/A',
                    "status_summary": entry_data.status_summary or 'Not Started',
                    "due_date": entry_data.due_date.isoformat() if entry_data.due_date else None,
                    "completed_on": entry_data.completed_on.isoformat() if entry_data.completed_on else None,
                    "is_activity_overdue": entry_data.is_activity_overdue or False,
                    "delay_in_days": entry_data.delay_in_days or 0,
                    "due_today": entry_data.due_today or False,
                    "on_time": entry_data.on_time or False,
                    "open_duration_days": entry_data.open_duration_days or 0,
                    "allocated_to": entry_data.allocated_to or 'Unassigned'
                }
                for entry in daily_entries
            ]
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        print(f"Error fetching daily work summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch daily work summary: {e}")

@app.delete("/api/projects/{project_id}", response_class=JSONResponse)
async def delete_project_api(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).filter(Project.project_id == project_id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    await db.delete(project)
    try:
        await db.commit()
        return JSONResponse(content={"message": f"Project '{project.project_id}' deleted successfully!"})
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {e}")

@app.get("/home", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request):
    return templates.TemplateResponse("project_list.html", {"request": request})

@app.get("/all_tasks", response_class=HTMLResponse)
async def all_tasks_page(request: Request):
    return templates.TemplateResponse("all_tasks.html", {"request": request})