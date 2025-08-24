"""
Authentication and Authorization Module for API Gateway.

This module provides JWT-based authentication, API key management,
and user authorization for the RevOps API Gateway.

Based on FastAPI best practices and current authentication patterns.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import structlog

from models import UserRole, TokenType, UserResponse, TokenResponse
from database import APIGatewayDatabaseManager, get_database_manager

logger = structlog.get_logger(__name__)

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = os.getenv("SECRET_KEY", "")
    logger.warning("JWT_SECRET_KEY not found, using SECRET_KEY fallback")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRY_HOURS", "24")) * 60
REFRESH_TOKEN_EXPIRE_DAYS = 30

class AuthenticationError(Exception):
    """Authentication-related errors"""
    pass

class AuthorizationError(Exception):
    """Authorization-related errors"""
    pass


class AuthManager:
    """Manages authentication and authorization operations"""
    
    def __init__(self, db_manager: Optional[APIGatewayDatabaseManager] = None):
        self.db_manager = db_manager or get_database_manager()
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": TokenType.ACCESS.value})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": TokenType.REFRESH.value})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: TokenType = TokenType.ACCESS) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type.value:
                raise AuthenticationError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp is None or datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise AuthenticationError("Token expired")
            
            return payload
            
        except JWTError as e:
            logger.error("JWT verification failed", error=str(e))
            raise AuthenticationError("Invalid token")
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username/password"""
        try:
            async with self.db_manager.get_connection("local") as conn:
                cursor = await conn.cursor()
                
                # Get user from database
                await cursor.execute("""
                    SELECT user_id, username, email, password_hash, full_name, 
                           roles, is_active, created_at, last_login
                    FROM ops.users 
                    WHERE username = %s AND is_active = true
                """, (username,))
                
                user_record = await cursor.fetchone()
                
                if not user_record:
                    logger.warning("Authentication failed - user not found", username=username)
                    return None
                
                # Verify password
                if not self.verify_password(password, user_record[3]):
                    logger.warning("Authentication failed - invalid password", username=username)
                    return None
                
                # Update last login
                await cursor.execute("""
                    UPDATE ops.users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE user_id = %s
                """, (user_record[0],))
                
                await conn.commit()
                
                # Return user data
                return {
                    "user_id": str(user_record[0]),
                    "username": user_record[1],
                    "email": user_record[2],
                    "full_name": user_record[4],
                    "roles": user_record[5] or [UserRole.USER.value],
                    "is_active": user_record[6],
                    "created_at": user_record[7],
                    "last_login": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error("User authentication error", error=str(e), username=username)
            raise AuthenticationError("Authentication failed")
    
    async def create_user_tokens(self, user_data: Dict[str, Any]) -> TokenResponse:
        """Create access and refresh tokens for user"""
        token_data = {
            "sub": user_data["username"],
            "user_id": user_data["user_id"],
            "roles": user_data["roles"]
        }
        
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user_data["user_id"],
            username=user_data["username"],
            roles=user_data["roles"]
        )
    
    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Create new access token from refresh token"""
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, TokenType.REFRESH)
            
            # Get current user data
            username = payload.get("sub")
            if not username:
                raise AuthenticationError("Invalid token payload")
            
            # Get fresh user data from database
            async with self.db_manager.get_connection("local") as conn:
                cursor = await conn.cursor()
                await cursor.execute("""
                    SELECT user_id, username, roles, is_active
                    FROM ops.users 
                    WHERE username = %s AND is_active = true
                """, (username,))
                
                user_record = await cursor.fetchone()
                if not user_record:
                    raise AuthenticationError("User not found or inactive")
                
                user_data = {
                    "user_id": str(user_record[0]),
                    "username": user_record[1],
                    "roles": user_record[2] or [UserRole.USER.value]
                }
                
                return await self.create_user_tokens(user_data)
                
        except Exception as e:
            logger.error("Token refresh error", error=str(e))
            raise AuthenticationError("Token refresh failed")
    
    async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data"""
        try:
            async with self.db_manager.get_connection("local") as conn:
                cursor = await conn.cursor()
                
                await cursor.execute("""
                    SELECT ak.key_id, ak.name, ak.roles, ak.expires_at, u.user_id, u.username
                    FROM ops.api_keys ak
                    LEFT JOIN ops.users u ON ak.created_by = u.user_id
                    WHERE ak.key_hash = %s 
                    AND ak.is_active = true
                    AND (ak.expires_at IS NULL OR ak.expires_at > CURRENT_TIMESTAMP)
                """, (self.get_password_hash(api_key),))
                
                key_record = await cursor.fetchone()
                
                if key_record:
                    # Update last used timestamp
                    await cursor.execute("""
                        UPDATE ops.api_keys 
                        SET last_used = CURRENT_TIMESTAMP 
                        WHERE key_id = %s
                    """, (key_record[0],))
                    
                    await conn.commit()
                    
                    return {
                        "key_id": str(key_record[0]),
                        "name": key_record[1],
                        "roles": key_record[2] or [UserRole.SERVICE.value],
                        "user_id": str(key_record[4]) if key_record[4] else None,
                        "username": key_record[5] if key_record[5] else "api_key_user"
                    }
                
                return None
                
        except Exception as e:
            logger.error("API key verification error", error=str(e))
            return None
    
    async def create_api_key(self, name: str, roles: List[UserRole], 
                           expires_in_days: Optional[int] = None,
                           created_by: Optional[str] = None) -> str:
        """Create new API key"""
        try:
            # Generate API key
            api_key = f"rapi_{uuid.uuid4().hex}"
            key_hash = self.get_password_hash(api_key)
            
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            async with self.db_manager.get_connection("local") as conn:
                cursor = await conn.cursor()
                
                await cursor.execute("""
                    INSERT INTO ops.api_keys 
                    (key_id, name, key_hash, roles, expires_at, created_by, created_at, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, true)
                """, (
                    str(uuid.uuid4()),
                    name,
                    key_hash,
                    [role.value for role in roles],
                    expires_at,
                    created_by
                ))
                
                await conn.commit()
                
            logger.info("API key created", name=name, expires_at=expires_at)
            return api_key
            
        except Exception as e:
            logger.error("API key creation error", error=str(e), name=name)
            raise AuthenticationError("API key creation failed")


# Global auth manager instance
auth_manager = AuthManager()


# Dependency functions for FastAPI

async def get_current_user_from_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        username = payload.get("sub")
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "username": username,
            "user_id": payload.get("user_id"),
            "roles": payload.get("roles", [UserRole.USER.value]),
            "auth_method": "jwt"
        }
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_from_api_key(api_key: Optional[str] = Depends(api_key_header)) -> Optional[Dict[str, Any]]:
    """Get current user from API key (optional)"""
    if not api_key:
        return None
        
    key_data = await auth_manager.verify_api_key(api_key)
    
    if key_data:
        return {
            "username": key_data["username"],
            "user_id": key_data.get("user_id"),
            "roles": key_data["roles"],
            "auth_method": "api_key",
            "key_id": key_data["key_id"]
        }
    
    return None


async def get_current_user(
    token_user: Optional[Dict[str, Any]] = Depends(lambda: None),
    api_key_user: Optional[Dict[str, Any]] = Depends(get_current_user_from_api_key)
) -> Dict[str, Any]:
    """Get current user from JWT token or API key"""
    
    # Try JWT token first
    try:
        credentials = Depends(security)
        if credentials:
            token_user = await get_current_user_from_token(credentials)
    except:
        pass
    
    # Use JWT token if available, otherwise API key
    current_user = token_user or api_key_user
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return current_user


def require_roles(required_roles: List[UserRole]):
    """Dependency factory for role-based access control"""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_roles = [UserRole(role) for role in current_user.get("roles", [])]
        
        # Admin has access to everything
        if UserRole.ADMIN in user_roles:
            return current_user
        
        # Check if user has any of the required roles
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[role.value for role in required_roles]}"
            )
        
        return current_user
    
    return role_checker


# Common role dependencies
require_admin = require_roles([UserRole.ADMIN])
require_user_or_admin = require_roles([UserRole.USER, UserRole.ADMIN])
require_service_or_admin = require_roles([UserRole.SERVICE, UserRole.ADMIN])