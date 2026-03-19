import sqlite3
conn = sqlite3.connect('portal.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cursor.fetchall())
cursor.execute("SELECT username FROM users")
print("Users:", cursor.fetchall())
conn.close()