# ผลทดสอบ TabPFN บน GPU — 11 ตุลาคม 2569

**ประเมิน test จริงครบ 5,929 แถวแล้ว** นี่เป็นการฝึกใหม่ในเครื่องด้วย TabPFN 9.1.0 / checkpoint TabPFN-3.5 ไม่ใช่การโหลด fitted model เดิมจาก Colab

## ผลที่วัดจริง

| ชุดข้อมูล | แถว | MAE นาที | RMSE นาที | R² | ผิดไม่เกิน ±5 นาที |
|---|---:|---:|---:|---:|---:|
| Validation | 5,910 | 3.1468 | 3.9120 | 0.8304 | 81.86% |
| Test | 5,929 | 3.1479 | 3.8965 | 0.8235 | 81.68% |

เทียบกับ HistGradientBoosting รอบก่อนบน test เดียวกัน MAE ลดจาก 3.4018 เป็น 3.1479 นาที หรือประมาณ 7.46% ทั้งสองใช้ train-only sample 5,000 แถว seed 42 และ shared preprocessing ที่รองรับ Excel time การเปรียบเทียบนี้ไม่ใช้ผล TabPFN validation เดิมใน Colab มาปะปน

## วิธีทดลองและหลักฐาน

- GPU: NVIDIA GeForce RTX 2050, VRAM 4 GB; Python 3.13.5; PyTorch 2.13.0+cu126; CUDA runtime 12.6
- GPU verification: คูณเมทริกซ์ 256×256 บน cuda:0 แล้วเทียบกับ CPU ผ่านจริง
- Environment `.venv-tabpfn-gpu` แยกจาก Python เดิมซึ่งยังเป็น PyTorch 2.14.0+cpu ไม่ติดตั้ง Toolkit หรือเปลี่ยน NVIDIA driver
- Whole-date split: train 29,674, validation 5,910, test 5,929; fit medians บน train เต็ม แล้วใช้ estimator sample 5,000 แถว; test อยู่หลัง validation ตามเวลา
- Parameters: n_estimators=1, fit_mode=low_memory, memory_saving_mode=True, random_state=42, inference batch 16
- Checkpoint: tabpfn-v3.5-20260909.safetensors; SHA-256 อยู่ใน checkpoint.json
- Validation และ test ประเมินครบเป็น batches; frozen_test_plan.json บันทึกการ freeze ก่อน test ไม่มีการเลือกพารามิเตอร์จาก test
- Native fitted state + shared preprocessing config บันทึกแล้ว; MLflow model มี checkpoint ประกอบและแก้ path ของ checkpoint เมื่อย้ายที่อยู่
- Preprocessing consistency, logged-model reload และ registry reload ผ่านจริง เปรียบเทียบ prediction ด้วย rtol/atol 1e-4
- ทดสอบโหลด native fitted state และทำนาย raw JSON อีกครั้งโดยไม่ refit ผ่าน 5 records ใน saved_model_prediction_check.json
- Peak allocated ของ PyTorch ระหว่าง fit/evaluation 1.24 GiB; ตัวเลขนี้ไม่ใช่ memory รวมของ GPU/driver/โปรแกรมอื่น
- ครั้งแรก import dependency ติด sandbox cache permission ก่อนฝึก; ปรับ cache ให้อยู่ในโครงงานแล้วรันต่อ มี log ของความล้มเหลวและรอบสำเร็จแยกไว้

## MLflow / Registry / การอนุมัติ

Experiment: Food_Delivery_TabPFN_GPU; Run ID: 6fb4ee943a164825866a2397380b0a7a; Registry: food-delivery-tabpfn-gpu, version 1 มี candidate alias และสถานะ quality_passed_not_serving_approved

MAE gate ≤5 นาทีผ่านทั้ง validation/test แต่ **ยังไม่อนุมัติให้บริการจริง**: ขนาดรวม checkpoint และ fitted state 835.57 MiB เกิน gate 100 MB ที่ประกาศไว้ และยังไม่ได้วัด HTTP latency/throughput ของ TabPFN ไม่เปลี่ยน champion หรือ API เดิมโดยอัตโนมัติ

Checkpoint มีขนาด 835 MiB; native fitted-state file มีขนาดเพียงประมาณ 132 KiB ชุด ZIP หลัก/ผล GPU แบบ compact ไม่รวม foundation checkpoint หรือ environment ให้ดาวน์โหลด checkpoint ด้วยบัญชีที่มีสิทธิ์ตาม GPU_SETUP.md หากย้ายเครื่อง Full MLflow model พร้อม weights มีอยู่ใน artifacts-tabpfn-gpu/mlruns/ ของเครื่องที่รันแล้ว

## ไฟล์ผลลัพธ์

result.json, validation_metrics.json, test_metrics.json, validation_predictions.csv, test_predictions.csv, preprocessing_config.json, training_sample_ids.csv, model.tabpfn_fit, gpu_check.json, checkpoint.json, environment.json, requirements-lock.txt, mlflow.db และ evaluation.log อยู่ใน artifacts-tabpfn-gpu/

Token/cache/venv ถูกกันออกจาก Git, Docker context และ ZIP ไม่ส่ง token ไปยัง Hugging Face; การตรวจ token ส่งไปยัง Prior Labs ตามที่ผู้ใช้อนุญาตเท่านั้น

## ข้อจำกัดที่ยังคงอยู่

โมเดลเดิมที่ fit ใน Colab ยังไม่ได้กู้กลับมา; ผลนี้เป็นการทดลองใหม่โดยใช้ preprocessing ที่ปรับปรุงแล้ว ข้อมูล test เป็น historical dataset ไม่ใช่ production labels ไม่ได้วัดผลธุรกิจจริง ไม่ได้ทดสอบ Docker/hosted CI/CD/external API/production GPU serving ในรอบนี้
