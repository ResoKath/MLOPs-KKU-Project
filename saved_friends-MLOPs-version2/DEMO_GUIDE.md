# คู่มือเตรียมและนำเสนอ 12 นาที

ใช้ร่วมกับ REPORT.md, COMPLETION_REPORT.md และ PR #1 ห้ามอ้างสมาชิก/สัมภาษณ์/ผลธุรกิจที่ยังไม่มี แต่ละคนแบ่งหัวข้อตาม contribution จริงและต้องอธิบายโค้ดที่ส่งได้

## ก่อนนำเสนอ

1. เปิด Docker Desktop ให้ engine พร้อม แล้ว `docker compose up --build -d` รอ `docker compose ps` แสดง api healthy บนเครื่องใหม่ดาวน์โหลดและฝึกครั้งแรกใช้หลายนาที ควรทำก่อนขึ้นพูด
2. เปิด http://localhost:8000/docs และ http://localhost:9090 เตรียม `artifacts-container/example_request_records.json` / `evidence/` ไม่แสดง token/private cache
3. เปิด PR และ Actions ที่ผ่าน/ที่ตั้งใจ fail เตรียมหลักฐาน full test CPU/GPU และ preprocessing/registry reload
4. หากจะรันครบวงจรจำลองใหม่ ใช้ Python ที่ติดตั้ง requirements.txt และ Docker CLI: `python scripts/container_acceptance.py baseline`, `feedback`, `alerts`, `cycle`, `system-alert` ตามลำดับ `alerts` ต้องรอจริง ≥5 นาที จึงควรทำก่อนนำเสนอ ไม่ต้องลด threshold เพื่อให้พูดทัน
5. `cycle` สร้าง isolated synthetic registry +12-minute target shift รัน DAG และเปลี่ยน/ย้อนกลับ actual API จากนั้นคืน original model ใน finally ไม่อ้างว่า labels เป็น production data ถ้ารันทวนอีกครั้งต้องใช้ volume/registryจำลองใหม่เพื่อให้เวอร์ชัน1→2ตรงกับassertions เก็บข้อมูลเก่าก่อนเปลี่ยนvolume
6. ไฟล์ compose เปิดพอร์ต8000/9090 ตรวจพอร์ตและเครือข่ายล่วงหน้า เครื่องที่สองยังไม่ได้ทดสอบ และไม่มีpublic production endpoint

## ลำดับพูด

| เวลา | เนื้อหา | หลักฐาน |
|---|---|---|
| 0–2 นาที | ปัญหา ผู้ใช้ที่เสนอ ใครเดือดร้อน KPIและข้อที่ยังไม่ได้ยืนยัน | CanvasในREPORT |
| 2–4 นาที | Schema ข้อมูล45,584แถว quarantine GPS temporal split sharedpreprocess/Excel time | config/tests + ป้อนrecords:{}ให้เห็น422 |
| 4–6 นาที | Dummy/Linear/HGB และTabPFN full test ความต่างMAEและเหตุผลserveHGB | selection/GPU_REPORT |
| 6–8 นาที | เรียกAPIปกติ health/metrics, p50/p95/throughputและSLO, boundgate | Swagger + container_benchmark |
| 8–10 นาที | แยกdata drift/label quality, firing, retrain request, version1→2→1และpredictionกลับ | feedback/alerts/live_rollback evidence |
| 10–12 นาที | DAGหนึ่งคำสั่ง, PR/CI3ด้านทั้งpass+reject/deploy, ข้อจำกัดและงานกลุ่ม | PR1/Actions/REPORTchecklist |

Q&A3นาที: ทำไมไม่randomsplit, เติมmedianจากtrainอย่างไร, multiple_deliveriesทำไมไม่บังคับ, checkpoint835MBทำไมเกินgate, PSI/MAEไม่พิสูจน์conceptdriftอย่างไร, labelvalidation/authยังขาดอะไร, restartrollbackมีdowntimeอย่างไร, ทำไมtestห้ามใช้tune

## Test cases ที่ซ้อมได้

Normalrawrecord→200และpositivefiniteprediction; requiredcolumnหาย/unsupportedcategory/Vehicle_condition9/GPSผิด/badtime/infinite/emptyหรือbatch101→422; healthเมื่อไม่มีmodel→503; feedbackผิดlength/unknown/duplicate→422/404/409; candidatehashเปลี่ยน→approvalreject; registryupdatefail→modelเดิมคงอยู่; monitoringmodelเก่าไม่ผสมและlabelsไม่ใช้ซ้ำ; corruptedtraining→DAGหยุดก่อนtrain

Testcaseผู้สอนในวันนำเสนอยังไม่ทราบและยังไม่ได้ทำจริง ผลลัพธ์ที่ส่งเป็นหลักฐานทดสอบของเราเท่านั้น สมาชิกและการประเมินcontributionต้องให้กลุ่มระบุเอง
