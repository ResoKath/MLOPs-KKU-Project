# Food Delivery Time Prediction — CP413008

อัปเดต 11 ต.ค. 2569: **TabPFN บน GPU ทดสอบ full test ผ่านแล้ว** ดู [GPU_REPORT.md](GPU_REPORT.md), [GPU_SETUP.md](GPU_SETUP.md) และ artifacts-tabpfn-gpu/


โครงงานต่อจากไฟล์ `mlop.ipynb` และ `Zomato Dataset.csv` ของผู้ใช้ อ่านผลและข้อจำกัดใน [REPORT.md](REPORT.md) และหลักฐานใน `artifacts/` ก่อนนำไปสาธิต

## รันจากเครื่องใหม่

ต้องมี Docker Desktop/Engine พร้อมใช้งาน:

```sh
docker compose up --build
```

คำสั่งนี้สร้าง environment, รัน DAG ภายใน, ตรวจข้อมูล, ฝึกสามโมเดล, เลือกจาก validation, ประเมิน full test, บันทึก MLflow, วัด HTTP, อนุมัติสำหรับ local demo แล้วเปิด API ที่ http://localhost:8000/docs และ Prometheus ที่ http://localhost:9090 ผลเก็บใน `artifacts-container/` ระหว่าง pipeline ยังทำงาน `/health/ready` อาจยังไม่พร้อม ตรวจ `docker compose logs api`

**Dockerfile/Compose ยังไม่ได้ build หรือรันจริงบนเครื่องที่จัดทำงานนี้ เพราะ Docker engine ไม่ทำงาน** การรันบนเครื่องใหม่จึงต้องตรวจซ้ำตามขั้นตอนนี้ สถานะ `approved_local_demo` ไม่ใช่การอนุมัติ production

## รันด้วย Python

แนะนำ Python 3.13 ใน environment แยก:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m food_delivery.pipeline --serve
```

สำหรับเครื่องที่จัดทำชุดนี้ ใช้ dependencies เพิ่มเติมใน `runtime_deps/` และ runtime เดิมของเครื่อง:

```powershell
$env:PYTHONPATH=(Resolve-Path runtime_deps).Path
$env:PREFECT_HOME=Join-Path (Get-Location) 'artifacts/prefect'
$env:PREFECT_SERVER_ANALYTICS_ENABLED='false'
$env:TEMP=(Resolve-Path .tmp).Path
$env:TMP=$env:TEMP
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
python -m food_delivery.pipeline --serve
```

การฝึกซ้ำสร้าง registry version ใหม่ แต่ใช้ seed และข้อมูลเดิม Test ถูกใช้เพื่อประเมินสูตรที่ freeze แล้วเท่านั้น ห้ามเลือกพารามิเตอร์จากผล test หลังเปลี่ยนสูตรต้องใช้ช่วงเวลาทดสอบใหม่เพื่ออ้างผล generalization

## ใช้โมเดลที่ฝึกแล้ว

```sh
python -m uvicorn food_delivery.api:app --host 127.0.0.1 --port 8000
python -m food_delivery.benchmark
python -m food_delivery.monitor
python -m food_delivery.demo
python -m food_delivery.ci_checks
python -m pytest -q
```

ส่ง JSON `{"records": [<ข้อมูลหนึ่งรายการ>]}` ไป `POST /predict` ตัวอย่าง records อยู่ใน `artifacts/example_request_records.json` มี `/health/live`, `/health/ready`, `/metrics`, `/feedback` ส่ง `request_id` และ `actual_minutes` เมื่อทราบเวลาจริง จำกัด 100 records ต่อคำขอ ข้อมูลผิดคืน 422 ไม่มีโมเดลคืน 503 API ใน Compose เปิด port 8000 ให้ภายนอก host ติดต่อได้ แต่ยังไม่มีหลักฐานเรียกจากเครื่องอื่น

## TabPFN บน GPU

รอบ GPU ใหม่รันจริงด้วย RTX 2050 / PyTorch 2.13.0+cu126 / TabPFN 9.1.0 รายละเอียดและคำสั่งอยู่ใน [GPU_SETUP.md](GPU_SETUP.md) ผล full test และข้อจำกัดอยู่ใน [GPU_REPORT.md](GPU_REPORT.md)

```powershell
.venv-tabpfn-gpu/Scripts/python.exe scripts/predict_tabpfn_saved.py
```

คำสั่งนี้โหลด native fitted state ที่ฝึกแล้วพร้อม checkpoint ใน cache และใช้ shared preprocessing โดยไม่ refit ทดสอบกับ raw JSON 5 records ผ่านจริง หากย้ายเครื่องต้องสร้าง environment และดาวน์โหลด checkpoint ด้วยสิทธิ์ของตนเองก่อน ชุด ZIP แบบ compact ไม่รวม environment, checkpoint หรือ credential

`notebooks/continue_original_tabpfn.ipynb` ยังเป็นขั้นตอนสำหรับ **fitted model เดิม** ที่อยู่ใน Colab session นั้นโดยเฉพาะ ซึ่งยังไม่ได้กู้กลับมา ผล GPU ใหม่ใช้ preprocessing ที่ปรับปรุงและ train-only sample จึงรายงานเป็นอีกการทดลอง

## Git / CI / rollback

มี Git local และ branch ของงาน ดู `git log --oneline --all` ยังไม่มี remote/บัญชีที่ระบุ จึงยังไม่ได้เปิด Pull Request จริง มี `PR_DESCRIPTION.md` ให้ใช้เมื่อ push branch

ชุด ZIP มี `food_delivery_history.bundle` สำหรับคืนประวัติ Git: `git clone food_delivery_history.bundle restored-project` แล้วคัดลอก artifacts/input notebook จาก ZIP ตามต้องการ ไฟล์ผลรันไม่ได้อยู่ใน commit เพราะแยกจาก source

`.github/workflows/ci.yml` ตรวจโค้ด ทดสอบ schema/consistency รัน pipeline ตรวจ model gate และ build image; workflow_dispatch มี deploy demo ใน runner ยังไม่ได้รันบน GitHub หลักฐาน `ci_*.json` เป็นการรันคำสั่งเดียวกันในเครื่องนี้ ไม่ใช่ผล GitHub Actions

ตัวรัน DAG ภายในเป็นค่าเริ่มต้นและได้รันจริงครบทุก node แล้ว ตัวเลือก `--engine prefect` ใช้ Prefect flow/tasks แต่การเริ่ม Prefect server ล้มเหลวใน sandbox เครื่องนี้ จึงยังไม่ยืนยันว่ารันสำเร็จทั้งกระบวนการด้วย Prefect

`python -m food_delivery.demo` สาธิต data drift, synthetic label shift +12 นาที, retraining และ registry alias rollback ในชื่อแยกจากโมเดลที่ใช้งาน ผลอยู่ใน `drift_retrain_rollback_demo.json`

การย้อนบริการจริงใช้ `food_delivery.registry.rollback(name, approved_version)` แล้ว restart API เพื่อโหลด artifact ใหม่ ทดสอบ readiness และ prediction หลัง restart ก่อนรับ traffic กลับ ห้ามโหลด artifact จากแหล่งที่ไม่ไว้ใจ

## โครงสร้าง

`food_delivery/`: reusable preprocessing/validation, training, serving, benchmark, registry, monitoring, Prefect DAG และ CI checks; `tests/`: กรณีปกติ/ผิดปกติ; `input/`: สำเนาไฟล์แนบ; `notebooks/`: ขั้นตอนต่อ; `monitoring/`: Prometheus/alerts; `artifacts/`: ผลการรันและโมเดล; `REPORT.md`: รายงานและ checklist

ข้อควรทราบ: MLflow SQLite/artifact URIs ในผลที่แนบอ้างพาธเครื่องที่รัน ให้รัน pipeline ใหม่เมื่อย้ายเครื่องเพื่อสร้าง registry ที่พาธใหม่ `model.joblib` ใช้ได้กับเวอร์ชันไลบรารีที่ตรงกัน Registry เดิมของ Colab ไม่ได้แนบมากับโน้ตบุ๊ก
