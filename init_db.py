from flask import Flask
from config import Config
from extensions import db, bcrypt
from db_models.user import User, GlobalSetting

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    db.drop_all() # Reset for schema changes
    db.create_all()
    
    # Create default settings
    if not GlobalSetting.query.filter_by(key='guest_analysis_limit').first():
        limit = GlobalSetting(key='guest_analysis_limit', value='10')
        db.session.add(limit)
    
    # Create default admin if not exists
    admin_email = "admin@arasent.ai"
    if not User.query.filter_by(email=admin_email).first():
        hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin = User(username='admin', email=admin_email, password_hash=hashed_pw, is_admin=True)
        db.session.add(admin)
        
    db.session.commit()
    print("Database tables created and initialized with defaults!")
