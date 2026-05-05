import psycopg2, os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('.env'))

try:
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        dbname=os.getenv('POSTGRES_DB', 'al_learning'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
        connect_timeout=5,
    )
    cur = conn.cursor()
    cur.execute('SELECT version();')
    print('Connected:', cur.fetchone()[0])
    cur.execute('SELECT datname FROM pg_database WHERE datistemplate = false;')
    print('Databases:', [r[0] for r in cur.fetchall()])
    conn.close()
    print('OK')
except Exception as e:
    print('FAILED:', e)
