# Phân tích lỗi Production RAG — Nguyễn Minh Lương

**Khóa:** K4 - Track 3A  
**Ngày:** 04/10/2026

## Tình trạng đánh giá

`python main.py` đã xử lý 20/20 câu hỏi ở lần chạy offline trước đó và lưu câu trả lời, đáp án chuẩn cùng các đoạn trích tại `reports/ragas_report.json`. Code hiện đã hỗ trợ Gemini và `.env` đã có khóa Gemini, nhưng yêu cầu gửi nội dung chính sách ra dịch vụ này bị hệ thống duyệt tự động chặn trong khi chờ ủy quyền cụ thể cho bộ dữ liệu. Vì vậy RAGAS chưa thể chấm bốn chỉ số. Các ô dưới đây là **N/A**, không phải điểm 0. Mô hình BGE-M3 và Cross-Encoder cũng chưa có trọng số trong cache; lần chạy báo cáo dùng MiniLM 384 chiều và bộ xếp hạng từ vựng dự phòng. Năm ca dưới đây là lỗi nghiêm trọng **chọn qua kiểm tra thủ công**, chưa phải Bottom-5 theo điểm RAGAS.

| Chỉ số | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | N/A | N/A | N/A |
| Answer Relevancy | N/A | N/A | N/A |
| Context Precision | N/A | N/A | N/A |
| Context Recall | N/A | N/A | N/A |

## Năm ca cần sửa ưu tiên

### 1. Senior có 9 năm thâm niên: phép năm và lương (câu 12)

- **Đáp án đúng:** 18 ngày phép theo chính sách 2024; lương Senior 20–35 triệu VNĐ/tháng.
- **Câu trả lời có đúng không?** Chưa. Bản offline trả nguyên đoạn chính sách nghỉ phép năm, nêu được ví dụ 18 ngày nhưng không trả lời phần lương.
- **Các đoạn trích có chứa đáp án không?** Chỉ chứa phần phép năm. Cả ba ngữ cảnh không có bảng lương Senior; hai đoạn còn lại là nghỉ phép không lương và quy định 2023 đã cũ.
- **Câu hỏi cần viết lại không?** Nên tách thành hai truy vấn con: “9 năm thâm niên được bao nhiêu ngày phép theo v2024?” và “Lương Senior P3–P4 là bao nhiêu?”. Câu gốc vẫn đủ rõ để hệ thống tốt trả lời.
- **Gốc lỗi và module cần sửa:** M2/M3 chưa bảo đảm đa dạng nguồn cho câu hỏi nhiều bước; M4 cần đánh dấu thiếu `context_recall`. Thêm truy xuất theo từng ý rồi hợp nhất hai nguồn trước khi sinh đáp án ở `pipeline.py`.

### 2. Nghỉ phép không lương 20 ngày (câu 20)

- **Đáp án đúng:** CEO phê duyệt vì thuộc khoảng 16–30 ngày; nghỉ trên 14 ngày thì nhân viên tự đóng phần bảo hiểm.
- **Câu trả lời có đúng không?** Chưa. Đoạn được trả lời đầu tiên nói về phép năm có lương, khác chế độ được hỏi.
- **Các đoạn trích có chứa đáp án không?** Có. Chính sách nghỉ phép không lương đứng thứ hai và ghi rõ cả thẩm quyền phê duyệt lẫn bảo hiểm.
- **Câu hỏi cần viết lại không?** Không bắt buộc; cụm “không lương 20 ngày” đã rõ. Có thể thêm “theo quy trình phê duyệt nghỉ không lương” để giảm nhầm lẫn.
- **Gốc lỗi và module cần sửa:** M3 xếp sai ngữ cảnh đầu bảng khi dùng fallback từ vựng, và nhánh sinh đáp án offline của `pipeline.py` chỉ trả đoạn đầu. Dùng Cross-Encoder thật khi có trọng số; thêm ưu tiên cụm “không lương” và tổng hợp từ cả ba đoạn.

### 3. Phạt tạm ứng quá hạn (câu 17)

- **Đáp án đúng:** Quá hạn 5 ngày; phí tháng là 2% × 15.000.000 = 300.000 VNĐ, ước tính 5 ngày theo tỷ lệ 30 ngày/tháng là 50.000 VNĐ.
- **Câu trả lời có đúng không?** Chưa. Bản offline chỉ trả toàn bộ chính sách tạm ứng, không tính số tiền phạt cụ thể.
- **Các đoạn trích có chứa đáp án không?** Có dữ liệu đầu vào: thời hạn 15 ngày và phí 2%/tháng. Kết quả 50.000 VNĐ cần phép tính bổ sung.
- **Câu hỏi cần viết lại không?** Nên chỉ rõ quy ước tính tỷ lệ 30 ngày/tháng nếu cần một số tiền duy nhất; câu hỏi hiện tại vẫn đủ để nêu phép tính và giả định.
- **Gốc lỗi và module cần sửa:** `pipeline.py` cần bước tính toán có kiểm tra hoặc LLM có prompt buộc trình bày phép tính từ nguồn. M4 cần đối chiếu đáp án số với `ground_truth`.

### 4. Lương thử việc Junior mức cao nhất (câu 18)

- **Đáp án đúng:** 20.000.000 × 85% = 17.000.000 VNĐ/tháng.
- **Câu trả lời có đúng không?** Chưa. Đoạn đầu là bảng lương, có mức trần Junior và tỷ lệ thử việc, nhưng không kết luận 17 triệu.
- **Các đoạn trích có chứa đáp án không?** Có. Bảng lương 2024 đứng đầu; chính sách thử việc đứng thứ hai. Hai nguồn đủ để tính.
- **Câu hỏi cần viết lại không?** Không; “Junior mức cao nhất” và “lương thử việc” đã xác định rõ hai giá trị cần kết hợp.
- **Gốc lỗi và module cần sửa:** M2/M3 đã tìm đúng tài liệu; lỗi nằm ở bước tổng hợp và tính toán trong `pipeline.py`. Cần trả lời ngắn gọn với công thức và nguồn tương ứng.

### 5. Laptop 30 triệu cho nhân viên mới (câu 13)

- **Đáp án đúng:** Giám đốc phòng ban phê duyệt; cần xác nhận cấu hình từ CNTT và ít nhất ba báo giá cho đơn hàng trên 10 triệu.
- **Câu trả lời có đúng không?** Chưa theo dạng hỏi đáp: bản offline trả cả quy trình mua sắm thay vì kết luận ba điều kiện áp dụng cho 30 triệu.
- **Các đoạn trích có chứa đáp án không?** Có. Quy trình mua sắm đứng đầu và chứa cả bảng mức phê duyệt, quy định báo giá và lưu ý về thiết bị CNTT.
- **Câu hỏi cần viết lại không?** Không bắt buộc. Có thể tách thành “ai phê duyệt?”, “CNTT xác nhận gì?” và “cần mấy báo giá?” để kiểm tra từng ý.
- **Gốc lỗi và module cần sửa:** Bước trả lời trong `pipeline.py` cần chọn đúng hàng 5–50 triệu của bảng rồi tổng hợp với hai quy định khác trong cùng parent chunk. M1 đã giữ nguyên bảng trong đoạn cha; M4 cần kiểm tra đủ ba ý.

## Ca minh họa và bước tiếp theo

Câu 20 cho thấy ngữ cảnh đúng đã được tìm thấy nhưng bị xếp sau tài liệu phép năm có lương. Cây chẩn đoán: **đáp án chưa đúng → ngữ cảnh đúng có ở vị trí 2 → câu hỏi đủ rõ → sửa xếp hạng M3 và bước tạo câu trả lời trong pipeline**. Sau khi được phép gửi dữ liệu lab tới Gemini và có trọng số BGE, chạy lại `python main.py`, lấy Bottom-5 thực theo điểm RAGAS, rồi cập nhật bảng số liệu và năm ca trong tài liệu này.
