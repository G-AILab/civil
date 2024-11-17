curl -X POST "http://localhost:8081/models?url=ts2vec-har.mar"
curl -v -X PUT "http://localhost:8081/models/ts2vec-har?min_worker=2"


Dockerfile

joblib
scikit-learn
faiss-cpu
einops
pandas
statsmodels