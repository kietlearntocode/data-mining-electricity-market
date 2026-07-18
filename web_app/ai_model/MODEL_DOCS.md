# Tài liệu Mô hình Dự báo Giá Điện Châu Âu (XGBoost)

*Tài liệu này được tạo ra nhằm mục đích tổng hợp danh sách các đặc trưng (Features) và kết quả huấn luyện của mô hình XGBoost, phục vụ bổ sung báo cáo thực nghiệm cho bài báo nghiên cứu Data Mining.*

---

## 1. Thông tin Tập Dữ liệu (Dataset Overview)
- **Phạm vi Quốc gia:** 17 Quốc gia Châu Âu (Tọa độ và đặc điểm được phân cụm từ trước).
- **Giai đoạn Dữ liệu Train/Test:** Từ `01-01-2019` đến `31-12-2025` (Look-back window năm 2018 đã được sử dụng để tạo biến `Lag365`).
- **Chiến lược Split:** Walk-Forward Split (Train: 2019-2025, Test: 2026 out-of-sample). Dữ liệu được huấn luyện mỗi ngày nhằm dự báo giá điện Day-ahead Market.

## 2. Danh sách Đặc trưng Đầu vào (35 Features)
Cấu trúc Features được thiết kế với cơ chế **Zero Data Leakage** (Không rò rỉ dữ liệu). Mốc thời gian mô phỏng thực tế là **12h00 Trưa Ngày T-1**, do đó các biến Phụ tải và Vĩ mô chỉ được phép lấy từ **T-2** trở về trước để đảm bảo tính minh bạch.

### A. Nhóm Vĩ mô - Năng lượng (Macro C1) - *15 biến*
Bao gồm 5 loại hàng hóa Vĩ mô, mỗi loại được nhân bản ra 3 mốc thời gian để mô hình nhận biết "Động lượng" (Gia tốc biến thiên xu hướng giá) trong ngắn hạn và dài hạn:
- `TTF_Gas_Lag2`, `TTF_Gas_Lag3`, `TTF_Gas_Lag14` (Giá khí đốt tự nhiên)
- `Coal_Lag2`, `Coal_Lag3`, `Coal_Lag14` (Giá than API2)
- `EU_ETS_Lag2`, `EU_ETS_Lag3`, `EU_ETS_Lag14` (Giá chứng chỉ CO2 Châu Âu)
- `Brent_Oil_Lag2`, `Brent_Oil_Lag3`, `Brent_Oil_Lag14` (Giá dầu Brent)
- `EU_Gas_Storage_Lag2`, `EU_Gas_Storage_Lag3`, `EU_Gas_Storage_Lag14` (Mức độ tồn kho Khí đốt Anomaly)

### B. Nhóm Giá Điện Lịch sử (Price Lags) - *6 biến*
Vì giá điện của ngày T-1 đã được Sàn chốt xong từ 12h00 trưa T-2, nhóm này được phép dùng mốc Lag 1:
- Ngắn hạn: `Price_Lag1`, `Price_Lag2`
- Chu kỳ dài hạn: `Price_Lag7`, `Price_Lag14`, `Price_Lag30`, `Price_Lag365`

### C. Nhóm Phụ tải Dư (Residual Load) - *4 biến*
- `Load_Lag2`, `Load_Lag3` (Cặp động lượng tiêu thụ ngắn hạn 48h).
- `Load_Lag7`, `Load_Lag14` (Bắt nhịp chu kỳ tiêu thụ theo ngày trong tuần).

### D. Nhóm Động lượng Thống kê (Rolling Stats) - *3 biến*
- `Price_Roll7_Mean`: Trung bình Giá điện trong 7 ngày qua (Mỏ neo giữ thăng bằng xu hướng).
- `Price_Roll7_Std`: Độ biến động (Volatility) của Giá điện 7 ngày qua.
- `Load_Roll7_Mean`: Trung bình Phụ tải 7 ngày qua (Biểu diễn xu hướng thời tiết kéo dài như nắng nóng/giá rét).

### E. Nhóm Chu kỳ Thời gian (Cyclical) - *4 biến*
- `DayOfWeek_Sin`, `DayOfWeek_Cos`, `Month_Sin`, `Month_Cos`

### F. Nhóm Hồ sơ Quốc gia tĩnh (Country Profiles) - *3 biến*
- `Country_Avg_Load`, `Country_Avg_Residual_Load`, `Country_Avg_Price` (Trung bình 2018-2024 để phân định không gian cụm tọa độ).

---

## 3. Kết quả Huấn luyện (Evaluation Results)

Mô hình XGBoost (được tinh chỉnh siêu tham số mặc định của hệ thống) mang lại kết quả cực kỳ khả quan trên tập kiểm thử:

- **R² Score (Test):** **0.6921** (~ 70% phương sai được giải thích)
- **MAE (Test):** **17.13 EUR/MWh**

### Top 3 Đặc trưng Quan trọng nhất (Feature Importance):
1. **`Price_Lag1` (60.6%)**: Mô hình xem mức giá của ngày hôm qua là mỏ neo (Anchor) vững chắc nhất trong mọi điều kiện thị trường.
2. **`Price_Roll7_Mean` (15.7%)**: Khẳng định sự quan trọng của xu hướng tuần (Weekly Trend) trong việc lọc nhiễu Vĩ mô.
3. **`Price_Lag2` (3.9%)**: Kết hợp với Lag 1 tạo thành động lượng gia tốc nội tại mạnh mẽ.

*Kết luận: Việc thiết lập mạng lưới Lag sâu (Deep-Lags) cho Vĩ mô và Phụ tải (Lag 2, 3, 14) mặc dù làm tăng độ phức tạp, nhưng đã tạo ra một lớp giáp chống chịu (Robustness) tuyệt vời, ngăn chặn hiện tượng mô hình phản ứng thái quá với nhiễu loạn thị trường ngắn hạn.*
