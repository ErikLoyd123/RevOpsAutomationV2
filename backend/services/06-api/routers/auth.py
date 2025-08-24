"""
Authentication Router for API Gateway.

Provides login, logout, token refresh, and user management endpoints.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
import structlog
import uuid

from models import (
    LoginRequest, TokenResponse, RefreshTokenRequest,
    UserCreate, UserResponse, UserUpdate,
    ApiKeyRequest, ApiKeyResponse, ErrorResponse
)
from auth import (
    auth_manager, get_current_user, require_admin,
    AuthenticationError
)
from dependencies import (
    get_database_manager, get_request_logger
)
from database import APIGatewayDatabaseManager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Authenticate user and return JWT tokens.
    
    This endpoint follows OAuth2 password flow standards:
    - Accepts form data (not JSON) with username and password
    - Returns access_token and refresh_token
    - Compatible with FastAPI's OAuth2PasswordBearer
    """
    
    logger.info("User login attempt", username=form_data.username)
    
    try:
        # Authenticate user
        user_data = await auth_manager.authenticate_user(
            username=form_data.username,
            password=form_data.password
        )
        
        if not user_data:
            logger.warning("Login failed - invalid credentials", username=form_data.username)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Generate tokens
        tokens = await auth_manager.create_user_tokens(user_data)
        
        logger.info(
            "User login successful",
            username=form_data.username,
            user_id=user_data["user_id"]
        )
        
        return tokens
        
    except AuthenticationError as e:
        logger.error("Authentication error during login", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected error during login", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Refresh access token using refresh token.
    
    Returns new access and refresh tokens.
    """
    
    logger.info("Token refresh requested")
    
    try:
        tokens = await auth_manager.refresh_access_token(refresh_request.refresh_token)
        
        logger.info("Token refresh successful")
        return tokens
        
    except AuthenticationError as e:
        logger.warning("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected error during token refresh", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed due to server error"
        )


@router.post("/logout")
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Logout current user.
    
    Note: With JWT tokens, logout is mainly for logging purposes
    since tokens cannot be invalidated server-side until expiry.
    """
    
    logger.info(
        "User logout",
        username=current_user.get("username"),
        user_id=current_user.get("user_id"),
        auth_method=current_user.get("auth_method")
    )
    
    return {
        "message": "Successfully logged out",
        "timestamp": logger._context.get("timestamp") if hasattr(logger, '_context') else None
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: APIGatewayDatabaseManager = Depends(get_database_manager),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Get current authenticated user's profile.
    """
    
    logger.info("Get current user profile", user_id=current_user.get("user_id"))
    
    try:
        async with db_manager.get_connection("local") as conn:
            cursor = await conn.cursor()
            
            await cursor.execute("""
                SELECT user_id, username, email, full_name, roles, 
                       is_active, created_at, last_login
                FROM ops.users 
                WHERE user_id = %s
            """, (current_user["user_id"],))
            
            user_record = await cursor.fetchone()
            
            if not user_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            return UserResponse(
                user_id=str(user_record[0]),
                username=user_record[1],
                email=user_record[2],
                full_name=user_record[3],
                roles=user_record[4] or [],
                is_active=user_record[5],
                created_at=user_record[6],
                last_login=user_record[7]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching user profile", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user profile"
        )


@router.patch("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: APIGatewayDatabaseManager = Depends(get_database_manager),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Update current authenticated user's profile.
    
    Users can update their own email, full_name, and password.
    They cannot modify roles or is_active status.
    """
    
    logger.info("Update user profile", user_id=current_user.get("user_id"))
    
    try:
        update_fields = []
        update_values = []
        
        # Build dynamic update query
        if user_update.email is not None:
            update_fields.append("email = %s")
            update_values.append(user_update.email)
        
        if user_update.full_name is not None:
            update_fields.append("full_name = %s")
            update_values.append(user_update.full_name)
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields provided for update"
            )
        
        # Add user_id for WHERE clause
        update_values.append(current_user["user_id"])
        
        async with db_manager.get_connection("local") as conn:
            cursor = await conn.cursor()
            
            # Perform update
            query = f"""
                UPDATE ops.users 
                SET {', '.join(update_fields)}
                WHERE user_id = %s
                RETURNING user_id, username, email, full_name, roles, 
                          is_active, created_at, last_login
            """
            
            await cursor.execute(query, update_values)
            updated_record = await cursor.fetchone()
            
            if not updated_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            await conn.commit()
            
            logger.info("User profile updated successfully")
            
            return UserResponse(
                user_id=str(updated_record[0]),
                username=updated_record[1],
                email=updated_record[2],
                full_name=updated_record[3],
                roles=updated_record[4] or [],
                is_active=updated_record[5],
                created_at=updated_record[6],
                last_login=updated_record[7]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating user profile", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


# Admin endpoints (require admin role)

@router.post("/users", response_model=UserResponse, dependencies=[Depends(require_admin)])
async def create_user(
    user_create: UserCreate,
    db_manager: APIGatewayDatabaseManager = Depends(get_database_manager),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Create a new user (admin only).
    """
    
    logger.info("Admin creating new user", username=user_create.username)
    
    try:
        # Hash password
        hashed_password = auth_manager.get_password_hash(user_create.password)
        
        async with db_manager.get_connection("local") as conn:
            cursor = await conn.cursor()
            
            # Check if user already exists
            await cursor.execute("""
                SELECT user_id FROM ops.users WHERE username = %s OR email = %s
            """, (user_create.username, user_create.email))
            
            if await cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this username or email already exists"
                )
            
            # Create new user
            user_id = str(uuid.uuid4())
            await cursor.execute("""
                INSERT INTO ops.users 
                (user_id, username, email, password_hash, full_name, roles, 
                 is_active, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, true, CURRENT_TIMESTAMP)
                RETURNING user_id, username, email, full_name, roles, 
                          is_active, created_at, last_login
            """, (
                user_id,
                user_create.username,
                user_create.email,
                hashed_password,
                user_create.full_name,
                [role.value for role in user_create.roles]
            ))
            
            new_user = await cursor.fetchone()
            await conn.commit()
            
            logger.info("User created successfully", new_user_id=user_id)
            
            return UserResponse(
                user_id=str(new_user[0]),
                username=new_user[1],
                email=new_user[2],
                full_name=new_user[3],
                roles=new_user[4] or [],
                is_active=new_user[5],
                created_at=new_user[6],
                last_login=new_user[7]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating user", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.post("/api-keys", response_model=ApiKeyResponse, dependencies=[Depends(require_admin)])
async def create_api_key(
    api_key_request: ApiKeyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    logger: structlog.BoundLogger = Depends(get_request_logger)
):
    """
    Create a new API key (admin only).
    """
    
    logger.info("Admin creating API key", name=api_key_request.name)
    
    try:
        api_key = await auth_manager.create_api_key(
            name=api_key_request.name,
            roles=api_key_request.roles,
            expires_in_days=api_key_request.expires_in_days,
            created_by=current_user["user_id"]
        )
        
        # Calculate expiration date
        expires_at = None
        if api_key_request.expires_in_days:
            from datetime import datetime, timedelta
            expires_at = datetime.utcnow() + timedelta(days=api_key_request.expires_in_days)
        
        return ApiKeyResponse(
            key_id=str(uuid.uuid4()),  # This would be returned from create_api_key in real implementation
            api_key=api_key,
            name=api_key_request.name,
            description=api_key_request.description,
            roles=api_key_request.roles,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
    except Exception as e:
        logger.error("Error creating API key", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )