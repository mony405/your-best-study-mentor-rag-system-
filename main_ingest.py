from dotenv import load_dotenv
from core.database import DataBase

# Load env to get OpenAI Key
load_dotenv()

if __name__ == "__main__":
    print("Prepareing the db with new docs in it ....")
    
    # Initialize DB 
    db = DataBase()
    
    # Run the full sync
    db.sync_data()
    
    print("DataBase is ready to use")