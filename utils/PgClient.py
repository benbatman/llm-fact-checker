import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dateutil import parser

import uuid
from datetime import datetime


def parse_dates(date_str: str):
    updated_on_match = "Updated on" in date_str
    if updated_on_match:
        date_str = date_str.replace("Updated on", "").strip()
        date_str = date_str.split(" ")
        date_str = " ".join(date_str[:2])
        date_object = parser.parse(date_str)
        update_date = date_object.strftime("%Y-%m-%d")

        # Get published date
        date_str = date_str.split("Published on")[-1].replace("\n", "").strip()
        date_str = date_str.split(" ")
        date_str = " ".join(date_str[:2])
        date_object = parser.parse(date_str)
        publish_date = date_object.strftime("%Y-%m-%d")
        return update_date, publish_date

    else:
        parsed_date = parser.parse(date_str)
        return None, parsed_date.strftime("%Y-%m-%d")


class PgClient:
    def __init__(self, host, db_name, user, password):
        self.host = host
        self.db_name = db_name
        self.user = user
        self.password = password
        self.conn = self.make_connection(
            self.host, self.db_name, self.user, self.password
        )
        cur = self.conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def make_connection(
        self, host: str, db_name: str, user: str, password: str
    ) -> "psycopg2.connection":
        conn = psycopg2.connect(
            host=self.host, user=self.user, password=self.password, dbname=self.db_name
        )
        return conn

    def create_tables(self, vector_dim: int = 512):
        pbs_sql = f"""
              CREATE TABLE IF NOT EXISTS pbs (
                  id SERIAL PRIMARY KEY,
                  title TEXT,
                  timestamp DATE,
                  url TEXT,
                  content TEXT,
                  content_embedding vector({vector_dim}),
                  title_embedding vector({vector_dim}),
                  scaled_content_embedding halfvec({vector_dim}),
                  scaled_title_embedding halfvec({vector_dim}),
                  binary_content_embedding bit({vector_dim}),
                  binary_title_embedding bit({vector_dim}));
              """

        snopes_sql = f"""
              CREATE TABLE IF NOT EXISTS snopes (
                  id SERIAL PRIMARY KEY,
                  claim_content TEXT,
                  claim_rating TEXT,
                  timestamp DATE,
                  url TEXT,
                  content TEXT,
                  content_embedding vector({vector_dim}),
                  claim_content_embedding vector({vector_dim}),
                  scaled_content_embedding halfvec({vector_dim}),
                  scaled_claim_content_embedding halfvec({vector_dim}),
                  binary_content_embedding bit({vector_dim}),
                  binary_claim_content_embedding bit({vector_dim}));
              """

        try:
            cur = self.conn.cursor()
            cur.execute(pbs_sql)
            cur.execute(snopes_sql)
            self.conn.commit()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cur.close()

    def insert_data(self, table_name: str, data: list[dict]):
        cur = self.conn.cursor()

        # Two tables: pbs, snopes
        # In PBS table:
        # title, timestamp, url, content, content_embedding

        try:
            if table_name == "pbs":
                for i, asset in enumerate(data):
                    asset["id"] = str(uuid.uuid4())
                    update_date, publish_date = parse_dates(asset["timestamp"])
                    asset["timestamp"] = publish_date
                    asset["updated_on"] = update_date
                    cur.execute(
                        f"""INSERT INTO {table_name} 
                            (id, title, timestamp, url, content, content_embedding, title_embedding, scaled_content_embedding, scaled_title_embedding, binary_content_embedding, binary_title_embedding, updated_on)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (url) DO NOTHING;
                            """,
                        (
                            asset["id"],
                            asset["title"],
                            asset["timestamp"],
                            asset["url"],
                            asset["body_text"],
                            asset["content_embedding"],
                            asset["title_embedding"],
                            asset["scaled_content_embedding"],
                            asset["scaled_title_embedding"],
                            asset["binary_content_embedding"],
                            asset["binary_title_embedding"],
                            asset["updated_on"],
                        ),
                    )

                if i % 1000 == 0 and i != 0:
                    print(f"Inserted {i} rows")
                    self.conn.commit()

                self.conn.commit()

            elif table_name == "snopes":
                for i, asset in enumerate(data):
                    asset["id"] = str(uuid.uuid4())
                    date_object = parser.parse(asset["published_date"])
                    asset["published_date"] = date_object.strftime("%Y-%m-%d")
                    cur.execute(
                        f"""INSERT INTO {table_name}
                            (id, claim_content, claim_rating, timestamp, url, content, content_embedding, claim_content_embedding, scaled_content_embedding, 
                            scaled_claim_content_embedding, binary_content_embedding, binary_claim_content_embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (url) DO NOTHING;
                            """,
                        (
                            asset["id"],
                            asset["claim_cont"],
                            asset["claim_rating"],
                            asset["published_date"],
                            asset["url"],
                            asset["article_text"],
                            asset["article_embedding"],
                            asset["claim_cont_embedding"],
                            asset["scaled_article_embedding"],
                            asset["scaled_claim_cont_embedding"],
                            asset["binary_article_embedding"],
                            asset["binary_claim_cont_embedding"],
                        ),
                    )

                print(f"Inserted {i} rows")
                self.conn.commit()

        except Exception as e:
            print(f"Error: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def query(self, query: str):
        cur = self.conn.cursor()
        cur.execute(query)
        results = cur.fetchall()
        cur.close()
        return results

    async def hybrid_search(self, query: str):
        pass
