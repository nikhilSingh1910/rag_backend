from flask_jwt_extended import create_access_token
from models.database import User, engine
from sqlalchemy.orm import sessionmaker
import bcrypt
from email_validator import validate_email, EmailNotValidError
from datetime import timedelta

Session = sessionmaker(bind=engine)

class AuthService:
    @staticmethod
    async def register_user(email: str, password: str) -> dict:
        """Register a new user"""
        try:
            # Validate email
            validate_email(email)
            
            session = Session()
            
            # Check if user already exists
            if session.query(User).filter_by(email=email).first():
                session.close()
                raise ValueError("Email already registered")
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Create user
            user = User(email=email, password_hash=password_hash.decode('utf-8'))
            session.add(user)
            session.commit()
            
            # Generate access token
            access_token = create_access_token(
                identity=user.id,
                expires_delta=timedelta(days=1)
            )
            
            session.close()
            
            return {
                'message': 'User registered successfully',
                'access_token': access_token,
                'user_id': user.id
            }
            
        except EmailNotValidError:
            raise ValueError("Invalid email format")
        except Exception as e:
            raise ValueError(str(e))

    @staticmethod
    async def login_user(email: str, password: str) -> dict:
        """Login a user"""
        try:
            session = Session()
            
            # Find user
            user = session.query(User).filter_by(email=email).first()
            if not user:
                session.close()
                raise ValueError("User not found")
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                session.close()
                raise ValueError("Invalid password")
            
            # Generate access token
            access_token = create_access_token(
                identity=user.id,
                expires_delta=timedelta(days=1)
            )
            
            session.close()
            
            return {
                'message': 'Login successful',
                'access_token': access_token,
                'user_id': user.id
            }
            
        except Exception as e:
            raise ValueError(str(e))

    @staticmethod
    async def get_user_by_id(user_id: int) -> User:
        """Get user by ID"""
        session = Session()
        user = session.query(User).get(user_id)
        session.close()
        return user 