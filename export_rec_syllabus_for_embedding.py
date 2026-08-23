# export_rec_syllabus_for_embedding.py
import json, psycopg2

DB_CONFIG = dict(dbname="credittransfer", user="postgres", password="devpass", host="localhost", port=5432)

def export(regulation_id=1, output_path="rec_syllabus_export.json"):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT id, code FROM course WHERE regulation_id = %s", (regulation_id,))
    courses = cur.fetchall()

    output = []
    for course_id, code in courses:
        cur.execute("SELECT unit_no, text FROM topic WHERE course_id = %s ORDER BY unit_no", (course_id,))
        topics = [{"unit_no": u, "text": t} for u, t in cur.fetchall()]
        output.append({"course_id": course_id, "course_code": code, "topics": topics})

    cur.close(); conn.close()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(output)} courses to {output_path}")

if __name__ == "__main__":
    export()