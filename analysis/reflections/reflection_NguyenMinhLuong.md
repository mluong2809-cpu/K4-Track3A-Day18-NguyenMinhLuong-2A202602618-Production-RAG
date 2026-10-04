# Bài thu hoạch cá nhân — Nguyễn Minh Lương

**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Liên hệ bài giảng với code

| Khái niệm | Module và hàm | Quan sát trong bài lab |
|---|---|---|
| Semantic chunking | M1 `chunk_semantic()` | Tách câu bằng regex, mã hóa với `all-MiniLM-L6-v2`, rồi ngắt khi cosine similarity thấp hơn 0,85. Dùng để giữ câu hoàn chỉnh theo chủ đề. |
| Parent–child retrieval | M1 `chunk_hierarchical()` và `pipeline.run_query()` | Chunk con tối đa 256 ký tự được đánh chỉ mục; `parent_id` dẫn tới đoạn cha tối đa 2048 ký tự. Pipeline xếp hạng lại và gửi đoạn cha làm ngữ cảnh. |
| Structure-aware chunking | M1 `chunk_structure_aware()` | Tách tại tiêu đề Markdown, giữ bảng và danh sách trong nội dung của section. |
| Hybrid search, BM25 và dense | M2 `segment_vietnamese()`, `BM25Search`, `DenseSearch` | Chuẩn hóa `_` về dấu cách sau `underthesea`. BM25 bắt từ khóa, Qdrant truy vấn vector qua `query_points()`. |
| Reciprocal Rank Fusion | M2 `reciprocal_rank_fusion()` | Cộng `1/(60 + rank + 1)` từ mỗi danh sách; không cần ép điểm BM25 và cosine về cùng thang. |
| Cross-Encoder reranking | M3 `CrossEncoderReranker.rerank()` | Tạo cặp `(query, passage)`, gọi `predict()`, sắp xếp giảm dần và lấy top 3. Bản chạy hiện tại dùng fallback từ vựng vì thiếu trọng số BGE; không xem đây là kết quả Cross-Encoder thật. |
| RAGAS và cây chẩn đoán | M4 `evaluate_ragas()`, `failure_analysis()` | Dữ liệu gồm question, answer, contexts, ground truth; bốn metric dùng để xác định lỗi ở bước trả lời hay truy xuất. Đã cấu hình Gemini làm model chấm; khi chưa được phép gửi dữ liệu, báo cáo ghi N/A thay vì điểm giả. |
| Contextual enrichment và HyQA | M5 `_enrich_single_call()`, `contextual_prepend()` | Một yêu cầu JSON tạo summary, questions, context và metadata. Nếu thiếu API key, fallback trích từ văn bản và thêm nguồn tài liệu vào đầu chunk. |

## Phần 2: Khó khăn và cách xử lý

1. **Thiếu thư viện trong môi trường:** `underthesea`, `qdrant-client` và `pypdf` chưa được cài; lệnh pip trong sandbox bị chặn kết nối. Sau khi cài qua quyền truy cập mạng được chấp thuận, M1/M2 đọc được PDF có text layer và chạy BM25/Qdrant. Hai PDF scan vẫn cần OCR.
2. **Tải mô hình BGE không hoàn tất:** Cache chỉ có MiniLM đầy đủ. Lần thử nạp `BAAI/bge-reranker-v2-m3` từ Hugging Face không hoàn tất, nên bổ sung nhánh offline có thông báo rõ; dense dùng MiniLM 384 chiều và reranker dùng điểm từ vựng. Đường chạy BGE-M3 1024 chiều và Cross-Encoder thật vẫn nằm trong code khi có trọng số.
3. **Lỗi phát sinh khi chỉnh code:** Bản đầu của M5 gọi `_ask_openai()` nhưng thiếu khai báo hàm, gây `NameError` ở 10 test M5; đã bổ sung hàm trợ giúp và chạy lại 10/10 passed. Lần chạy pytest không đặt offline flag còn khiến M1 cố kiểm tra Hub và gặp lỗi kết nối; đã nạp MiniLM từ cache bằng `local_files_only=True`.
4. **Đánh giá qua Gemini chưa được phép gửi dữ liệu:** Lần chạy offline đầu tiên chưa có khóa Gemini, nên `main.py` xử lý 20 câu nhưng RAGAS không tạo điểm. Sau đó đã cấu hình Gemini và xác nhận API trả lời bằng một yêu cầu thử vô hại. Hệ thống duyệt tự động chặn yêu cầu gửi một chunk chính sách cụ thể; báo cáo vẫn giữ prediction/context của lần chạy offline, trạng thái chờ ủy quyền và 5 ca phân tích thủ công.

## Phần 3: Kế hoạch áp dụng

**Dự án đề xuất:** trợ lý tra cứu chính sách và quy trình nội bộ bằng tiếng Việt.

**Hiện trạng giả định:** tài liệu Markdown/PDF, gồm nhiều phiên bản quy định và bảng số liệu; truy vấn từ khóa đơn thuần dễ lấy nhầm bản cũ hoặc thiếu một nguồn trong câu hỏi nhiều bước.

1. **Tuần 1 — chuẩn bị dữ liệu:** OCR PDF scan; trích `source`, phiên bản và ngày hiệu lực thành metadata. Dùng structure-aware cho chính sách có tiêu đề/bảng; dùng parent–child 2048/256 cho truy xuất.
2. **Tuần 2 — tìm kiếm và xếp hạng:** tải đủ BGE-M3 và BGE reranker, dùng BM25 + dense + RRF, lọc quy định đã hết hiệu lực, rồi rerank các đoạn cha. Với câu nhiều ý, tách truy vấn và hợp nhất nguồn trước khi trả lời.
3. **Tuần 3 — làm giàu và đo lường:** bật contextual prepend/HyQA theo batch một API call/chunk sau khi có quyền gửi tài liệu; kiểm tra chi phí và cache kết quả. Tạo bộ câu hỏi có xung đột phiên bản, phủ định, bảng và phép tính. Chạy RAGAS bốn metric bằng Gemini, xem Bottom-5 thực và lặp lại việc sửa lỗi. Đo latency rerank bằng `benchmark_reranker()` trên phần cứng đích; không áp chỉ tiêu 150 ms cho bản fallback như thể đó là Cross-Encoder.

**Tiêu chí nghiệm thu:** 100% test kỹ thuật pass; điểm RAGAS thực được lưu cùng cấu hình mô hình và ngày chạy; các câu hỏi dùng phiên bản cũ hoặc cần nhiều nguồn được kiểm tra thủ công trước khi triển khai.
