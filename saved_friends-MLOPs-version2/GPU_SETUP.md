# TabPFN GPU environment

Environment ใหม่ชื่อ `.venv-tabpfn-gpu` แยกจาก Python/PyTorch CPU ที่มีอยู่เดิม เครื่องที่ตรวจพบคือ NVIDIA RTX 2050, VRAM 4 GB, driver 566.07 ใช้ PyTorch 2.13.0 รุ่น CUDA 12.6 ตามตัวเลือก wheel ทางการ

```powershell
python -m venv --without-pip .venv-tabpfn-gpu
$env:TEMP=(Resolve-Path .tmp).Path
$env:TMP=$env:TEMP
python -m pip --python .venv-tabpfn-gpu install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip --python .venv-tabpfn-gpu install -r requirements-tabpfn-gpu.txt
.venv-tabpfn-gpu/Scripts/python.exe scripts/check_gpu.py
python scripts/check_tabpfn_access.py
python scripts/download_tabpfn_checkpoint.py
.venv-tabpfn-gpu/Scripts/python.exe scripts/evaluate_tabpfn_gpu.py --batch-size 16
```

Token ใช้ผ่าน `.cache-tabpfn/token.local` ในเครื่องเท่านั้น ไฟล์ credential, cache และ environment นี้ถูกกันออกจาก Git/Docker context และต้องไม่รวมในชุดส่งมอบ

ผล GPU check ต้องมีสถานะ PASSED และมีการคูณเมทริกซ์บน CUDA จริง ไม่ใช้แค่การตรวจชื่อการ์ดจอ ผล evaluation จะอยู่ใน `artifacts-tabpfn-gpu/` โดยแยก validation/test metrics, predictions, fitted state, checkpoint hash และ MLflow run

การทดลองนี้ฝึกใหม่ด้วย sample 5,000 แถว seed 42 และ preprocessing ที่ปรับปรุงแล้ว ไม่ใช่การกู้ fitted model เดิมของ Colab ใช้ full chronological validation และ full test โดยไม่ปรับพารามิเตอร์จาก test

โมเดลรวม preprocessing config และ native fitted state พร้อม checkpoint ใน MLflow เพื่อให้โหลดจากที่อยู่ใหม่ได้ Candidate ยังต้องผ่าน serving/size gates ก่อนอนุมัติใช้งานจริง ไม่มีการแทน champion เดิมโดยอัตโนมัติ

สถานะสำเร็จให้ดูไฟล์ผลรันจริง ไม่ถือว่าคำสั่งในเอกสารนี้เป็นหลักฐานว่ารันผ่านแล้ว CUDA Toolkit ไม่ถูกติดตั้งแยก และไม่มีการเปลี่ยน NVIDIA driver ของเครื่อง

อ้างอิง: [PyTorch CUDA 12.6 wheels](https://pytorch.org/get-started/previous-versions/), [TabPFN 9.1.0](https://pypi.org/project/tabpfn/9.1.0/)
