# ผลดำเนินการเพิ่มเติม — 11 ตุลาคม 2569

## ผลที่รันและตรวจสอบจริง

| ขั้นตอน | ผล/หลักฐาน |
|---|---|
| Promotion gate | ผูกcandidatechecksum/run/registryversion/datahashกับHTTPevidence; stalecandidateถูกปฏิเสธ และregistryupdatefailคืนไฟล์เดิม |
| Incumbent gate | candidateต้องไม่แย่กว่าโมเดลเดิมบนvalidationชุดเดียวกันก่อนtest/registry; รันDAGซ้ำversion4→5ผ่าน MAEเท่ากัน3.3636; มีtestrejectregression |
| Unit/integration | ล่าสุด21ผ่านบนWindows ดูevidence/tests-windows.xml |
| Container | Dockerbuild/Linuxfreshinstall/Compose3services/APIpublishedportผ่าน; pipcheckไม่มีdependencyเสีย ดูrequirements-linux.txt |
| SLO | 100requests/concurrency4/warmup10: p50 235.64ms, p95 341.55ms, throughput16.50requests/s, error0%; ผ่านSLO1s/10rps/1% |
| Monitoring | 2หน้าต่างไม่ทับซ้อน≥100labels/หน้าต่าง ผ่านfeedbackAPIจริงและตรวจdataPSI/MAE12/retrainrequest; labels+12นาทีและGPSshiftเป็นการจำลอง |
| Drift/qualityalert | Prometheusscrapeและfiringจริงหลังเกณฑ์5นาที ดูcontainer_drift_alerts.json |
| Systemalert | หยุดAPIสาธิตตามเกณฑ์1นาที DeliveryAPIDownfiringจริง แล้วเริ่มAPIคืน ดูcontainer_system_alert.json |
| Retrain/rollback | registryจำลองแยกversion1→2→1 ผ่านactualHTTPหลังrestart; predictionsเพิ่ม12นาทีและกลับตรงเดิม จากนั้นคืนoriginalliveHGB ดูcontainer_live_rollback.json |
| Git/PR | branchcodex/production-completion และPR1จริง รักษาไฟล์/ประวัติเดิม ยังไม่merge |
| HostedCI | code/schema/recomputedmodelgate/Dockerbuildผ่าน; data9และpredictions+20ถูกปฏิเสธจริง |
| CDdemo | GitHubrunnerสร้างcontainer เปิดAPI healthผ่าน แล้วdownสำเร็จ; ไม่มีpublicpersistentdeployment |
| TabPFN | Fulltest5,929แถวบนCUDAเสร็จก่อนรอบนี้ MAE3.1479, RMSE3.8965, R².8235; candidateยังเกิน100MBgateและไม่มีGPUHTTPbenchmark |

## ขอบเขตและข้อที่ยังเหลือ

ผู้ใช้แจ้งว่าไม่มีข้อมูลสมาชิกและstakeholder จึงยังไม่มีชื่อสมาชิก/contribution/สัมภาษณ์/ผู้ใช้pilotหรือผลKPIธุรกิจจริง สิ่งเหล่านี้ต้องดำเนินการโดยกลุ่ม ไม่สร้างหลักฐานแทน มีCanvasและDEMO_GUIDE.mdพร้อมใช้

วงจรretrainingที่สาธิตใช้historicalCSVกับlabelshiftที่จำลอง ไม่ใช่freshverifiedbusinessdata ระบบมีincumbentgateแล้ว แต่การเก็บข้อมูลใหม่ การยืนยันlabels การอนุมัติจากstakeholder/authenticationและretentionยังต้องทำก่อนproduction MAEเสื่อมเป็นconceptdriftsignal ไม่พิสูจน์causality

HTTPวัดจากWindowshostภายนอกLinuxcontainer ยังไม่วัดเครื่องกายภาพที่สอง/internet/longdurationload/monthlyavailability การrollbackใช้restartมีdowntime Outboundnotificationยังไม่มีAlertmanagerreceiver ไม่มีการส่งข้อความถึงบุคคลภายนอก

เอกสาร/branch/PR/Actionsและหลักฐานมีจริง ข้อสอบผู้สอนและการนำเสนอ12นาทียังไม่ได้เกิดขึ้น GitcommitโดยCodexไม่แทนcontributionนักศึกษา foundationcheckpoint/tokenไม่อยู่ในชุดส่ง ต้องใช้GPU_SETUP.mdหากต้องการย้ายTabPFNไปเครื่องอื่น

## ลิงก์หลักฐานภายนอก

- PR: https://github.com/ResoKath/MLOPs-KKU-Project/pull/1
- CIผ่าน: https://github.com/ResoKath/MLOPs-KKU-Project/actions/runs/38129778934
- CI+deploydemoผ่าน: https://github.com/ResoKath/MLOPs-KKU-Project/actions/runs/38129779033
- datafailตามตั้งใจ: https://github.com/ResoKath/MLOPs-KKU-Project/actions/runs/38129810912
- modelfailตามตั้งใจ: https://github.com/ResoKath/MLOPs-KKU-Project/actions/runs/38129813163

JSONแต่ละrunบันทึกheadSHAและstepstates การเพิ่มsourceหลังrunต้องตรวจCIของrevisionล่าสุดเพิ่มเติม ผลbenchmark/rawtraining/GPU/labelsimulationเป็นคนละขอบเขต ไม่รวมคะแนนเข้าด้วยกัน
