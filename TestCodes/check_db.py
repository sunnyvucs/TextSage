import psycopg2, os
from dotenv import load_dotenv
load_dotenv('.env')

conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=os.getenv('POSTGRES_PORT', '5432'),
    dbname=os.getenv('POSTGRES_DB', 'al_learning'),
    user=os.getenv('POSTGRES_USER', 'postgres'),
    password=os.getenv('POSTGRES_PASSWORD', ''),
)
cur = conn.cursor()

for table in ['chunks', 'documents', 'flashcard_progress']:
    print(f'\n=== {table} ===')
    cur.execute(f"""
        SELECT column_name, data_type, character_maximum_length, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table}' AND table_schema = 'public'
        ORDER BY ordinal_position;
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:30s} {row[1]:20s} nullable={row[3]}")

# Show the one document row
print('\n=== documents sample ===')
cur.execute("SELECT * FROM documents LIMIT 1;")
cols = [d[0] for d in cur.description]
row = cur.fetchone()
if row:
    for c, v in zip(cols, row):
        print(f"  {c}: {str(v)[:80]}")

conn.close()
