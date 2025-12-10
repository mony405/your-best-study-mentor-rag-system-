from dotenv import load_dotenv
from core.database import DataBase

# Load env to get OpenAI Key
load_dotenv()

if __name__ == "__main__":
    print("📂 Starting Manual Data Ingestion...")
    
    # Initialize DB (which uses Config paths)
    db = DataBase()
    
    # Run the full sync
    db.sync_data()
    
    print("🎉 Ingestion Finished.")