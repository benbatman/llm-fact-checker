## A project NVIDIA's and LlamaIndex's Developer Competition

**How to Run**

1. Run `pip install -r requirements.txt`
2. Create `.env` file with following api keys:
   - `NGC_API_KEY`=`your-ngc-api-key`
   - `SERP_API_KEY`=`your-serp-api-key`
3. Set the `api_key` value under `parameters` in `config.yml` equal to `your-ngc-api-key`
4. Ensure you have the `vector_index` folder in root
5. Run `streamlit run app.py`
6. Go to the url provided, `http://localhost:8501`
7. Intialization will take a couple minutes to load the vector index into memory
