"""
Document Vault — Capture, OCR/Extract, Store, Enrich, Search
==============================================================
Single-file Streamlit micro-app.

Stack:
  - UI:        Streamlit
  - Storage:   SQLite (stdlib sqlite3)
  - Extract:   OpenAI Vision (gpt-4o-mini, multimodal) with automatic
               fallback to PyTesseract OCR if no API key is configured
               or the API call fails.

Run:
  pip install -r requirements.txt
  streamlit run app.py

Set your OpenAI key either via environment variable OPENAI_API_KEY,
or paste it into the sidebar at runtime (kept only in session_state,
never written to disk).
"""

import os
import io
import json
import sqlite3
import base64
from datetime import datetime
from contextlib import contextmanager

import streamlit as st
from PIL import Image

# ---- Optional dependencies (app must not crash if these are missing) ----
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
DB_PATH = "documents.db"
UPLOAD_DIR = "doc_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_CATEGORIES = ["ใบเสร็จ", "สัญญา", "ฟอร์ม", "ใบแจ้งหนี้", "อื่นๆ"]
STATUS_OPTIONS = ["รอตรวจสอบ", "อนุมัติแล้ว", "ปฏิเสธ", "ต้องแก้ไข"]

EXTRACTION_SCHEMA_PROMPT = """
คุณคือระบบดึงข้อมูลจากภาพเอกสาร (ใบเสร็จ/สัญญา/ฟอร์ม/ใบแจ้งหนี้) ให้ออกมาเป็น JSON เท่านั้น
ห้ามใส่ข้อความอื่นนอกเหนือจาก JSON วัตถุเดียว ห้ามใส่ ```json หรือ markdown fence ใดๆ

โครงสร้าง JSON ที่ต้องการ (ปรับ field ให้เหมาะกับเอกสารจริง แต่คง key หลักไว้):

{
  "document_type": "receipt | contract | form | invoice | other",
  "header": {
    "title": "ชื่อเอกสาร/ร้านค้า/คู่สัญญา",
    "document_number": "เลขที่เอกสารถ้ามี",
    "date": "วันที่บนเอกสาร (YYYY-MM-DD ถ้าแปลงได้ ไม่งั้นตามที่เห็น)",
    "issuer": "ผู้ออกเอกสาร/ผู้ขาย",
    "recipient": "ผู้รับ/ลูกค้า (ถ้ามี)"
  },
  "line_items": [
    {"description": "รายการ", "quantity": "จำนวน", "unit_price": "ราคาต่อหน่วย", "amount": "ยอดรวมรายการ"}
  ],
  "summary": {
    "subtotal": "ยอดก่อนภาษี/ส่วนลด ถ้ามี",
    "tax": "ภาษี ถ้ามี",
    "discount": "ส่วนลด ถ้ามี",
    "total": "ยอดรวมสุทธิ",
    "currency": "สกุลเงิน ถ้าระบุได้ เช่น THB, USD"
  },
  "raw_text": "ข้อความทั้งหมดที่อ่านได้จากภาพ แบบดิบ ไม่ต้องจัดรูปแบบ"
}

หากข้อมูลบาง field ไม่มีในเอกสาร ให้ใส่ null หรือ [] ตามความเหมาะสม
ห้ามสร้างข้อมูลที่ไม่มีในภาพขึ้นมาเอง (ห้าม hallucinate)
"""


# ============================================================
# DATABASE LAYER
# ============================================================
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                image_path      TEXT NOT NULL,
                document_type   TEXT,
                title           TEXT,
                document_number TEXT,
                doc_date        TEXT,
                issuer          TEXT,
                recipient       TEXT,
                subtotal        TEXT,
                tax             TEXT,
                discount        TEXT,
                total           TEXT,
                currency        TEXT,
                raw_text        TEXT,
                line_items_json TEXT,
                full_json       TEXT,
                extraction_method TEXT,
                -- enrichment fields --
                tags            TEXT DEFAULT '',
                notes           TEXT DEFAULT '',
                status          TEXT DEFAULT 'รอตรวจสอบ',
                subcategory     TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(
                title, document_number, issuer, recipient, raw_text,
                tags, notes, subcategory,
                content='documents', content_rowid='id'
            )
        """)
        # Triggers to keep FTS index in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
              INSERT INTO documents_fts(rowid, title, document_number, issuer, recipient, raw_text, tags, notes, subcategory)
              VALUES (new.id, new.title, new.document_number, new.issuer, new.recipient, new.raw_text, new.tags, new.notes, new.subcategory);
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
              INSERT INTO documents_fts(documents_fts, rowid, title, document_number, issuer, recipient, raw_text, tags, notes, subcategory)
              VALUES ('delete', old.id, old.title, old.document_number, old.issuer, old.recipient, old.raw_text, old.tags, old.notes, old.subcategory);
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
              INSERT INTO documents_fts(documents_fts, rowid, title, document_number, issuer, recipient, raw_text, tags, notes, subcategory)
              VALUES ('delete', old.id, old.title, old.document_number, old.issuer, old.recipient, old.raw_text, old.tags, old.notes, old.subcategory);
              INSERT INTO documents_fts(rowid, title, document_number, issuer, recipient, raw_text, tags, notes, subcategory)
              VALUES (new.id, new.title, new.document_number, new.issuer, new.recipient, new.raw_text, new.tags, new.notes, new.subcategory);
            END;
        """)


def insert_document(image_path: str, parsed: dict, extraction_method: str) -> int:
    header = parsed.get("header") or {}
    summary = parsed.get("summary") or {}
    line_items = parsed.get("line_items") or []
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO documents (
                created_at, updated_at, image_path, document_type,
                title, document_number, doc_date, issuer, recipient,
                subtotal, tax, discount, total, currency,
                raw_text, line_items_json, full_json, extraction_method
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            now, now, image_path, parsed.get("document_type"),
            header.get("title"), header.get("document_number"), header.get("date"),
            header.get("issuer"), header.get("recipient"),
            summary.get("subtotal"), summary.get("tax"), summary.get("discount"),
            summary.get("total"), summary.get("currency"),
            parsed.get("raw_text"), json.dumps(line_items, ensure_ascii=False),
            json.dumps(parsed, ensure_ascii=False), extraction_method
        ))
        return cur.lastrowid


def update_enrichment(doc_id: int, tags: str, notes: str, status: str, subcategory: str):
    with get_conn() as conn:
        conn.execute("""
            UPDATE documents
            SET tags = ?, notes = ?, status = ?, subcategory = ?, updated_at = ?
            WHERE id = ?
        """, (tags, notes, status, subcategory, datetime.now().isoformat(timespec="seconds"), doc_id))


def update_core_fields(doc_id: int, fields: dict):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [datetime.now().isoformat(timespec="seconds"), doc_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE documents SET {cols}, updated_at = ? WHERE id = ?", values)


def delete_document(doc_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT image_path FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    if row and row["image_path"] and os.path.exists(row["image_path"]):
        try:
            os.remove(row["image_path"])
        except OSError:
            pass


def get_all_documents(search_query: str = "", status_filter: str = "ทั้งหมด",
                       category_filter: str = "ทั้งหมด", tag_filter: str = ""):
    with get_conn() as conn:
        if search_query.strip():
            safe_q = search_query.strip().replace('"', '""')
            rows = conn.execute("""
                SELECT d.* FROM documents d
                JOIN documents_fts f ON d.id = f.rowid
                WHERE documents_fts MATCH ?
                ORDER BY d.created_at DESC
            """, (f'"{safe_q}"*',)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()

    results = [dict(r) for r in rows]
    if status_filter != "ทั้งหมด":
        results = [r for r in results if r["status"] == status_filter]
    if category_filter != "ทั้งหมด":
        results = [r for r in results if r["document_type"] == category_filter or r["subcategory"] == category_filter]
    if tag_filter.strip():
        needle = tag_filter.strip().lower()
        results = [r for r in results if needle in (r["tags"] or "").lower()]
    return results


def get_document(doc_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


def get_all_tags():
    with get_conn() as conn:
        rows = conn.execute("SELECT tags FROM documents WHERE tags != ''").fetchall()
    tag_set = set()
    for r in rows:
        for t in (r["tags"] or "").split(","):
            t = t.strip()
            if t:
                tag_set.add(t)
    return sorted(tag_set)


# ============================================================
# EXTRACTION LAYER (Vision LLM -> fallback OCR)
# ============================================================
def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_with_openai_vision(image_bytes: bytes, api_key: str, model: str = "gpt-4o-mini") -> dict:
    if not OPENAI_SDK_AVAILABLE:
        raise RuntimeError("ยังไม่ได้ติดตั้ง openai SDK (pip install openai)")
    client = OpenAI(api_key=api_key)
    b64 = image_to_base64(image_bytes)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract structured data from document images and reply with pure JSON only."},
            {"role": "user", "content": [
                {"type": "text", "text": EXTRACTION_SCHEMA_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        temperature=0,
        max_tokens=2000,
    )
    text = response.choices[0].message.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def extract_with_pytesseract(image_bytes: bytes) -> dict:
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("ยังไม่ได้ติดตั้ง pytesseract (pip install pytesseract) และ Tesseract binary")
    img = Image.open(io.BytesIO(image_bytes))
    raw_text = pytesseract.image_to_string(img, lang="tha+eng")
    # PyTesseract gives raw text only — we can't reliably structure it,
    # so we return a minimal skeleton the user can hand-edit afterward.
    return {
        "document_type": "other",
        "header": {"title": None, "document_number": None, "date": None, "issuer": None, "recipient": None},
        "line_items": [],
        "summary": {"subtotal": None, "tax": None, "discount": None, "total": None, "currency": None},
        "raw_text": raw_text.strip(),
    }


def run_extraction(image_bytes: bytes, api_key: str, model: str) -> tuple[dict, str]:
    """Try Vision LLM first if a key is present; fall back to OCR on any failure."""
    if api_key:
        try:
            return extract_with_openai_vision(image_bytes, api_key, model), f"openai:{model}"
        except Exception as e:
            st.warning(f"เรียก Vision API ไม่สำเร็จ ({e}) — เปลี่ยนไปใช้ PyTesseract OCR แทน")
    return extract_with_pytesseract(image_bytes), "pytesseract"


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Document Vault", page_icon="🗂️", layout="wide")
init_db()

if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    st.session_state.api_key = st.text_input(
        "OpenAI API Key (สำหรับ Vision extraction)",
        value=st.session_state.api_key,
        type="password",
        help="ถ้าไม่ใส่ ระบบจะใช้ PyTesseract OCR แทน (ต้องติดตั้ง Tesseract binary ในเครื่อง)"
    )
    model_choice = st.selectbox("Vision model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption(
        ("✅ OpenAI SDK พร้อมใช้งาน" if OPENAI_SDK_AVAILABLE else "❌ ไม่พบ openai SDK") + " | " +
        ("✅ PyTesseract พร้อมใช้งาน" if PYTESSERACT_AVAILABLE else "❌ ไม่พบ pytesseract")
    )
    st.divider()
    with get_conn() as conn:
        total_docs = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    st.metric("เอกสารทั้งหมด", total_docs)

st.title("🗂️ Document Vault")
st.caption("ถ่าย/อัปโหลดเอกสาร → ดึงข้อมูลอัตโนมัติ → บันทึก → เพิ่มแท็ก/โน้ต → ค้นหา")

tab_upload, tab_browse = st.tabs(["📤 อัปโหลด / ถ่ายภาพ", "🔍 ค้นหา & จัดการเอกสาร"])

# ------------------------------------------------------------
# TAB 1: CAPTURE / UPLOAD / EXTRACT / STORE
# ------------------------------------------------------------
with tab_upload:
    col_input, col_preview = st.columns([1, 1])

    with col_input:
        input_mode = st.radio("แหล่งภาพ", ["📁 อัปโหลดไฟล์", "📷 ถ่ายภาพ"], horizontal=True)
        image_file = None
        if input_mode == "📁 อัปโหลดไฟล์":
            image_file = st.file_uploader("เลือกรูปเอกสาร", type=["png", "jpg", "jpeg", "webp"])
        else:
            image_file = st.camera_input("ถ่ายภาพเอกสาร")

    if image_file is not None:
        image_bytes = image_file.getvalue()
        with col_preview:
            st.image(image_bytes, caption="ตัวอย่างภาพ", use_container_width=True)

        if st.button("🚀 ดึงข้อมูล & บันทึก", type="primary", use_container_width=True):
            with st.spinner("กำลังดึงข้อมูลจากภาพ..."):
                try:
                    parsed, method = run_extraction(image_bytes, st.session_state.api_key, model_choice)
                except Exception as e:
                    st.error(f"ดึงข้อมูลไม่สำเร็จ: {e}")
                    parsed, method = None, None

            if parsed is not None:
                # Persist the image file
                ext = (image_file.type.split("/")[-1] if hasattr(image_file, "type") and image_file.type else "jpg")
                fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{ext}"
                fpath = os.path.join(UPLOAD_DIR, fname)
                with open(fpath, "wb") as f:
                    f.write(image_bytes)

                doc_id = insert_document(fpath, parsed, method)
                st.success(f"บันทึกเอกสาร #{doc_id} สำเร็จ (extraction: {method})")
                with st.expander("ดูผล JSON ที่ดึงได้", expanded=True):
                    st.json(parsed)
                st.info("ไปที่แท็บ '🔍 ค้นหา & จัดการเอกสาร' เพื่อดูรายละเอียด เพิ่มแท็ก/โน้ต หรือแก้ไขข้อมูล")

# ------------------------------------------------------------
# TAB 2: SEARCH / BROWSE / DETAIL / ENRICH
# ------------------------------------------------------------
with tab_browse:
    fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1, 1, 1])
    with fcol1:
        search_q = st.text_input("🔎 คำค้นหา (ค้นในชื่อ, เลขที่เอกสาร, ผู้ออก, เนื้อหา, แท็ก, โน้ต)", "")
    with fcol2:
        status_f = st.selectbox("สถานะ", ["ทั้งหมด"] + STATUS_OPTIONS)
    with fcol3:
        category_f = st.selectbox("หมวดหมู่", ["ทั้งหมด"] + DEFAULT_CATEGORIES)
    with fcol4:
        existing_tags = get_all_tags()
        tag_f = st.selectbox("แท็ก", [""] + existing_tags, format_func=lambda x: "ทั้งหมด" if x == "" else x)

    docs = get_all_documents(search_q, status_f, category_f, tag_f)
    st.caption(f"พบ {len(docs)} รายการ")

    if "selected_doc_id" not in st.session_state:
        st.session_state.selected_doc_id = None

    list_col, detail_col = st.columns([1, 1.4])

    with list_col:
        for d in docs:
            label = f"**{d['title'] or 'ไม่มีชื่อ'}**  \n{d['doc_date'] or ''} · {d['document_type'] or '-'} · {d['status']}"
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(label)
                if d["tags"]:
                    c1.caption("🏷️ " + d["tags"])
                if c2.button("เปิด", key=f"open_{d['id']}", use_container_width=True):
                    st.session_state.selected_doc_id = d["id"]
                    st.rerun()

    with detail_col:
        sel_id = st.session_state.selected_doc_id
        if sel_id is None:
            st.info("เลือกเอกสารจากรายการด้านซ้ายเพื่อดูรายละเอียด")
        else:
            doc = get_document(sel_id)
            if doc is None:
                st.warning("เอกสารนี้ถูกลบไปแล้ว")
                st.session_state.selected_doc_id = None
            else:
                st.subheader(f"📄 เอกสาร #{doc['id']}")
                if os.path.exists(doc["image_path"]):
                    st.image(doc["image_path"], use_container_width=True)

                with st.expander("ข้อมูลหลัก (Header / Line Items / Summary)", expanded=True):
                    ec1, ec2 = st.columns(2)
                    title_v = ec1.text_input("ชื่อเอกสาร", doc["title"] or "", key=f"title_{doc['id']}")
                    docnum_v = ec2.text_input("เลขที่เอกสาร", doc["document_number"] or "", key=f"docnum_{doc['id']}")
                    date_v = ec1.text_input("วันที่", doc["doc_date"] or "", key=f"date_{doc['id']}")
                    issuer_v = ec2.text_input("ผู้ออกเอกสาร", doc["issuer"] or "", key=f"issuer_{doc['id']}")
                    recipient_v = ec1.text_input("ผู้รับ", doc["recipient"] or "", key=f"recipient_{doc['id']}")
                    doctype_v = ec2.text_input("ประเภทเอกสาร", doc["document_type"] or "", key=f"doctype_{doc['id']}")

                    try:
                        items = json.loads(doc["line_items_json"] or "[]")
                    except json.JSONDecodeError:
                        items = []
                    if items:
                        st.markdown("**รายการ (Line Items)**")
                        st.table(items)
                    else:
                        st.caption("ไม่มีรายการย่อย")

                    sc1, sc2, sc3, sc4 = st.columns(4)
                    subtotal_v = sc1.text_input("Subtotal", doc["subtotal"] or "", key=f"subtotal_{doc['id']}")
                    tax_v = sc2.text_input("ภาษี", doc["tax"] or "", key=f"tax_{doc['id']}")
                    discount_v = sc3.text_input("ส่วนลด", doc["discount"] or "", key=f"discount_{doc['id']}")
                    total_v = sc4.text_input("รวมสุทธิ", doc["total"] or "", key=f"total_{doc['id']}")

                    if st.button("💾 บันทึกข้อมูลหลักที่แก้ไข", key=f"save_core_{doc['id']}"):
                        update_core_fields(doc["id"], {
                            "title": title_v, "document_number": docnum_v, "doc_date": date_v,
                            "issuer": issuer_v, "recipient": recipient_v, "document_type": doctype_v,
                            "subtotal": subtotal_v, "tax": tax_v, "discount": discount_v, "total": total_v,
                        })
                        st.success("บันทึกแล้ว")
                        st.rerun()

                    with st.expander("ข้อความดิบทั้งหมด (raw_text)"):
                        st.text(doc["raw_text"] or "-")

                st.markdown("### ✏️ Metadata เพิ่มเติม")
                tags_v = st.text_input("แท็ก (คั่นด้วยจุลภาค)", doc["tags"] or "", key=f"tags_{doc['id']}")
                subcat_v = st.text_input("หมวดหมู่ย่อย", doc["subcategory"] or "", key=f"subcat_{doc['id']}")
                status_v = st.selectbox(
                    "สถานะอนุมัติ", STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(doc["status"]) if doc["status"] in STATUS_OPTIONS else 0,
                    key=f"status_{doc['id']}"
                )
                notes_v = st.text_area("โน้ตเพิ่มเติม", doc["notes"] or "", key=f"notes_{doc['id']}")

                bcol1, bcol2 = st.columns(2)
                if bcol1.button("💾 บันทึก Metadata", type="primary", key=f"save_meta_{doc['id']}", use_container_width=True):
                    update_enrichment(doc["id"], tags_v, notes_v, status_v, subcat_v)
                    st.success("บันทึก metadata เรียบร้อย (real-time update)")
                    st.rerun()
                if bcol2.button("🗑️ ลบเอกสารนี้", key=f"del_{doc['id']}", use_container_width=True):
                    delete_document(doc["id"])
                    st.session_state.selected_doc_id = None
                    st.success("ลบเอกสารแล้ว")
                    st.rerun()
