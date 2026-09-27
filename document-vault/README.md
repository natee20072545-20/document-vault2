# Document Vault

Micro-app: ถ่าย/อัปโหลดเอกสาร → ดึงข้อมูลด้วย Vision LLM (หรือ OCR สำรอง) → เก็บใน SQLite → เพิ่มแท็ก/โน้ต/สถานะ → ค้นหา

## ติดตั้ง

```bash
pip install -r requirements.txt
```

ถ้าต้องการใช้ PyTesseract (ตัวสำรองเมื่อไม่มี OpenAI API key) ต้องติดตั้ง Tesseract binary เพิ่มด้วย:
- macOS: `brew install tesseract tesseract-lang`
- Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-tha`
- Windows: ดาวน์โหลดจาก https://github.com/UB-Mannheim/tesseract/wiki

## รัน

```bash
streamlit run app.py
```

ใส่ OpenAI API key ในแถบด้านซ้าย (sidebar) เพื่อใช้ Vision extraction (แม่นยำกว่าและให้ JSON โครงสร้างเต็มรูปแบบ)
ถ้าไม่ใส่ key ระบบจะ fallback ไปใช้ PyTesseract OCR อัตโนมัติ (ได้เฉพาะข้อความดิบ ต้องมาแก้ไข field เองในหน้า detail)

## ฟีเจอร์

- **อัปโหลด/ถ่ายภาพ**: รองรับทั้งอัปโหลดไฟล์และถ่ายภาพผ่านกล้องในเบราว์เซอร์
- **Extract & Parse**: ส่งภาพเข้า `gpt-4o-mini` (หรือ `gpt-4o`) เพื่อดึง JSON แบบมีโครงสร้าง (header / line_items / summary / raw_text) หรือใช้ PyTesseract OCR เป็นตัวสำรอง
- **SQLite storage**: บันทึกอัตโนมัติพร้อม `created_at`/`updated_at`, ค้นหาแบบ full-text ด้วย FTS5
- **Metadata enrichment**: หน้าเลือกเอกสาร ดูรายละเอียด แล้วเพิ่ม/แก้ไขแท็ก โน้ต สถานะอนุมัติ หมวดหมู่ย่อย ได้ทันที (real-time)
- **Search & filter**: ค้นหาด้วยคำค้น + กรองตามสถานะ/หมวดหมู่/แท็ก

## โครงสร้างฐานข้อมูล

ตาราง `documents` เก็บทั้งข้อมูลที่ดึงได้อัตโนมัติ (title, document_number, doc_date, issuer, line_items_json ฯลฯ) และ metadata ที่ผู้ใช้เพิ่มเอง (tags, notes, status, subcategory) พร้อมตาราง `documents_fts` (FTS5) สำหรับค้นหาแบบเต็มข้อความที่ sync อัตโนมัติผ่าน trigger

## หมายเหตุด้านความปลอดภัย

- API key ถูกเก็บใน `st.session_state` เท่านั้น (ไม่บันทึกลงไฟล์หรือฐานข้อมูล)
- รูปภาพต้นฉบับถูกเก็บไว้ใน `doc_images/` — หากเป็นข้อมูลอ่อนไหว ควรเข้ารหัสดิสก์หรือจำกัดสิทธิ์การเข้าถึงโฟลเดอร์นี้เพิ่มเติม
- โค้ดนี้ไม่มีระบบ authentication ในตัว — หากนำไป deploy แบบ public ควรเพิ่ม login layer เอง (เช่น `streamlit-authenticator`)
