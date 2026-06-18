FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY segment/ ./segment/
COPY api/proto/ ./api/proto/
COPY artifacts/ ./artifacts/
EXPOSE 50051
ENTRYPOINT ["python", "-m", "segment.main"]