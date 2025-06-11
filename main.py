from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, Integer, String, Date, DateTime, Text, Numeric, select, Boolean, UniqueConstraint, func
from datetime import date, datetime
from typing import Optional, List, Union, AsyncGenerator, Any
import os # Make sure this import is present
from decimal import Decimal

from fastapi.staticfiles import StaticFiles

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database Configuration ---
# Fetch DATABASE_URL from environment variable for deployment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:Pune123@localhost:5433/tendering_project_entry_db")
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

Base = declarative_base()

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
    project_id = Column(String, unique=True, index=True, nullable=False) # Auto-generated like "PID-2024-001"
    project_name = Column(String, nullable=False)
    company_name = Column(String, nullable=False) # UNIVASTU, UBILLP, Autofina Robotics
    tender_entry_date = Column(DateTime, nullable=False) # Changed to DateTime
    tender_status = Column(String, nullable=False) # To Be Submitted, Awarded, Study, Submitted, Financial Opened, Technical Opened, Not Submitted, Withdrawn, Cancelled
    name_of_client = Column(String, nullable=False)
    state = Column(String, nullable=False) # Gujrat, Karnataka, Madhya Pradesh, Maharashtra, Rajasthan, Uttar Pradesh
    project_type = Column(String, nullable=False) # Road, Building, Water Supply, Sports Complex, Metro, Commercial
    tender_id = Column(String, nullable=True)
    attch_tender_document = Column(String, nullable=True) # File path/URL
    tender_estimated_cost = Column(Numeric(18, 2), nullable=False) # Decimal for currency
    completion_period_month = Column(Integer, nullable=True) # <--- CHANGED TO nullable=True
    tender_submission_date = Column(Date, nullable=False)
    tender_opening_date = Column(Date, nullable=True)
    pre_bid_meeting_date = Column(Date, nullable=True)
    clarifications_issued_date = Column(Date, nullable=True)
    corrigendum_received_date = Column(Date, nullable=True)
    emd_required_inr = Column(Numeric(18, 2), nullable=False)
    emd_paid_inr = Column(Numeric(18, 2), nullable=True)
    emd_instrument_type = Column(String, nullable=True) # Cash, BG, FDR, DD
    emd_bg_details_bg_number = Column(String, nullable=True)
    emd_bg_details_bank_name = Column(String, nullable=True)
    emd_bg_details_bg_expiry_date = Column(Date, nullable=True)
    emd_return_date = Column(Date, nullable=True)
    sd_required_percent = Column(Numeric(5, 2), nullable=True) # Percentage (e.g., 5.00 for 5%)
    sd_required_inr = Column(Numeric(18, 2), nullable=True)
    sd_instrument_type = Column(String, nullable=True) # Cash, BG, FDR, DD
    pbg_required_percent = Column(Numeric(5, 2), nullable=True)
    pbg_required_inr = Column(Numeric(18, 2), nullable=True)
    pbg_instrument_type = Column(String, nullable=True) # Cash, BG, FDR, DD
    competition_no_of_bidders = Column(Integer, nullable=True)
    final_status_of_tender = Column(String, nullable=True) # Same as tender_status options
    remark = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String, default="New", nullable=False) # New, In Progress, Completed, etc.

    __table_args__ = (UniqueConstraint('project_id', name='_project_id_uc'),)


# --- FastAPI App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directory for uploaded documents
UPLOAD_DIRECTORY = "uploaded_documents"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- Database Session Dependency ---
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
        await session.close() # Ensure session is closed after use

# --- Utility Functions ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def parse_date_or_none(date_str: Optional[str]) -> Optional[date]:
    if date_str:
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return None
    return None

def parse_numeric_or_none(value_str: Optional[str]) -> Optional[Decimal]:
    if value_str:
        try:
            return Decimal(value_str)
        except (ValueError, TypeError):
            return None
    return None

def parse_int_or_none(value_str: Optional[str]) -> Optional[int]:
    if value_str:
        try:
            return int(value_str)
        except (ValueError, TypeError):
            return None
    return None

# --- Routes ---

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        # Create tables if they don't exist
        # NOTE: If you change nullable properties on existing columns,
        # you might need to drop and recreate your table or use Alembic for migrations.
        # For development, dropping and recreating is often simplest.
        await conn.run_sync(Base.metadata.create_all)

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
        # Pass a specific detail to the template for error display
        return templates.TemplateResponse(
            "signin.html",
            {"request": request, "error_message": "Invalid username or password"},
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    # On successful login, redirect to the dashboard
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    summary_data = {
        "toBeSubmitted": 5,
        "awarded": 3,
        "submitted": 10,
        "upcomingDeadlines": 2
    }
    return templates.TemplateResponse("dashboard.html", {"request": request, "summary": summary_data})

@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project))
    projects = result.scalars().all()
    projects_data = []
    for project in projects:
        project_dict = project.__dict__
        clean_project_dict = {k: v for k, v in project_dict.items() if not k.startswith('_')}
        for key, value in clean_project_dict.items():
            if isinstance(value, Decimal):
                clean_project_dict[key] = str(value)
            # Add a check for None before calling isoformat() for date/datetime fields
            elif isinstance(value, (date, datetime)):
                if value is not None:
                    clean_project_dict[key] = value.isoformat()
                else:
                    clean_project_dict[key] = None
        projects_data.append(clean_project_dict)
    return templates.TemplateResponse("projects.html", {"request": request, "projects": projects_data})


@app.get("/new_project", response_class=HTMLResponse)
async def new_project_page(request: Request, db: AsyncSession = Depends(get_db)):
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

    # Auto-generate current date and time for tender_entry_date
    current_datetime_str = datetime.now().isoformat(timespec='minutes') # Format for datetime-local input

    return templates.TemplateResponse("new_project.html", {"request": request, "next_project_id": next_project_id, "current_datetime_str": current_datetime_str})


@app.post("/projects/", response_class=HTMLResponse)
async def create_project(
    request: Request,
    project_id: str = Form(...),
    project_name: str = Form(...),
    company_name: str = Form(...),
    tender_status: str = Form(...),
    name_of_client: str = Form(...),
    state: str = Form(...),
    project_type: str = Form(...),
    tender_id: Optional[str] = Form(None),
    attch_tender_document: Optional[UploadFile] = File(None),
    tender_estimated_cost_str: str = Form(...), # Required
    completion_period_month_str: Optional[str] = Form(None), # Made Optional at FastAPI level
    tender_submission_date_str: str = Form(...), # Required
    tender_opening_date_str: Optional[str] = Form(None),
    pre_bid_meeting_date_str: Optional[str] = Form(None),
    clarifications_issued_date_str: Optional[str] = Form(None),
    corrigendum_received_date_str: Optional[str] = Form(None),
    emd_required_inr_str: str = Form(...), # Required
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
    project_status: str = Form("New"),
    db: AsyncSession = Depends(get_db)
):
    document_path = None
    if attch_tender_document and attch_tender_document.filename:
        file_location = os.path.join(UPLOAD_DIRECTORY, attch_tender_document.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await attch_tender_document.read())
        document_path = file_location

    # Auto-generate tender_entry_date with current datetime
    tender_entry_date = datetime.now()

    # Parse other date and numeric fields, handling empty strings as None
    tender_estimated_cost = parse_numeric_or_none(tender_estimated_cost_str)
    completion_period_month = parse_int_or_none(completion_period_month_str)
    tender_submission_date = parse_date_or_none(tender_submission_date_str)
    tender_opening_date = parse_date_or_none(tender_opening_date_str)
    pre_bid_meeting_date = parse_date_or_none(pre_bid_meeting_date_str)
    clarifications_issued_date = parse_date_or_none(clarifications_issued_date_str)
    corrigendum_received_date = parse_date_or_none(corrigendum_received_date_str)
    emd_required_inr = parse_numeric_or_none(emd_required_inr_str)
    emd_paid_inr = parse_numeric_or_none(emd_paid_inr_str)
    emd_instrument_type = emd_instrument_type
    emd_bg_details_bg_number = emd_bg_details_bg_number
    emd_bg_details_bank_name = emd_bg_details_bank_name
    emd_bg_details_bg_expiry_date = parse_date_or_none(emd_bg_details_bg_expiry_date_str)
    emd_return_date = parse_date_or_none(emd_return_date_str)
    sd_required_percent = parse_numeric_or_none(sd_required_percent_str)
    sd_required_inr = parse_numeric_or_none(sd_required_inr_str)
    sd_instrument_type = sd_instrument_type
    pbg_required_percent = parse_numeric_or_none(pbg_required_percent_str)
    pbg_required_inr = parse_numeric_or_none(pbg_required_inr_str)
    pbg_instrument_type = pbg_instrument_type
    competition_no_of_bidders = parse_int_or_none(competition_no_of_bidders_str)

    new_project = Project(
        project_id=project_id,
        project_name=project_name,
        company_name=company_name,
        tender_entry_date=tender_entry_date, # Auto-generated
        tender_status=tender_status,
        name_of_client=name_of_client,
        state=state,
        project_type=project_type,
        tender_id=tender_id,
        attch_tender_document=document_path,
        tender_estimated_cost=tender_estimated_cost,
        completion_period_month=completion_period_month,
        tender_submission_date=tender_submission_date,
        tender_opening_date=tender_opening_date,
        pre_bid_meeting_date=pre_bid_meeting_date,
        clarifications_issued_date=clarifications_issued_date,
        corrigendum_received_date=corrigendum_received_date,
        emd_required_inr=emd_required_inr,
        emd_paid_inr=emd_paid_inr,
        emd_instrument_type=emd_instrument_type,
        emd_bg_details_bg_number=emd_bg_details_bg_number,
        emd_bg_details_bank_name=emd_bg_details_bank_name,
        emd_bg_details_bg_expiry_date=emd_bg_details_bg_expiry_date,
        emd_return_date=emd_return_date,
        sd_required_percent=sd_required_percent,
        sd_required_inr=sd_required_inr,
        sd_instrument_type=sd_instrument_type,
        pbg_required_percent=pbg_required_percent,
        pbg_required_inr=pbg_required_inr,
        pbg_instrument_type=pbg_instrument_type,
        competition_no_of_bidders=competition_no_of_bidders,
        final_status_of_tender=final_status_of_tender,
        remark=remark,
        description=description,
        status=project_status
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

        return templates.TemplateResponse(
            "new_project.html",
            {"request": request, "next_project_id": next_project_id, "error_message": f"Failed to create project: {e}"},
            status_code=status.HTTP_400_BAD_REQUEST
        )

@app.get("/project_success", response_class=HTMLResponse)
async def project_success_page(request: Request):
    return templates.TemplateResponse("project_success.html", {"request": request})