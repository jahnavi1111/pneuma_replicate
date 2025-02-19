import os

import fire
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from index_generator.index_generator import IndexGenerator
from query.query import Query
from registration.registration import Registration
from summarizer.summarizer import Summarizer
from torch import bfloat16
from utils.pipeline_initializer import initialize_pipeline
from utils.storage_config import get_storage_path


class Pneuma:
    def __init__(
        self,
        out_path: str = get_storage_path(),
        hf_token: str = "",
        llm_path: str = "Qwen/Qwen2.5-7B-Instruct",
        embed_path: str = "BAAI/bge-base-en-v1.5",
        max_llm_batch_size: int = 50,
    ):
        os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path
        self.db_path = out_path + "/storage.db"
        self.index_location = out_path + "/indexes"

        self.hf_token = hf_token
        self.llm_path = llm_path
        self.embed_path = embed_path
        self.max_llm_batch_size = max_llm_batch_size

        self.__hf_login()

        self.registration = None
        self.summarizer = None
        self.index_generator = None
        self.query = None
        self.llm = None
        self.embed_model = None

    def __hf_login(self):
        if self.hf_token != "":
            try:
                login(self.hf_token)
            except ValueError:
                pass

    def __init_registration(self):
        self.registration = Registration(db_path=self.db_path)

    def __init_summarizer(self):
        self.__init_llm()
        self.__init_embed_model()
        self.summarizer = Summarizer(
            llm=self.llm,
            embed_model=self.embed_model,
            db_path=self.db_path,
            max_llm_batch_size=self.max_llm_batch_size,
        )

    def __init_index_generator(self):
        self.__init_embed_model()
        self.index_generator = IndexGenerator(
            embed_model=self.embed_model,
            db_path=self.db_path,
            index_path=self.index_location,
        )

    def __init_query(self):
        self.__init_llm()
        self.__init_embed_model()
        self.query = Query(
            llm=self.llm,
            embed_model=self.embed_model,
            db_path=self.db_path,
            index_path=self.index_location,
        )

    def __init_llm(self):
        if self.llm is None:
            self.llm = initialize_pipeline(
                self.llm_path, bfloat16, context_length=32768,
            )
            # Specific setting for batching
            self.llm.tokenizer.pad_token_id = self.llm.model.config.eos_token_id
            self.llm.tokenizer.padding_side = "left"
    
    def __init_embed_model(self):
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(self.embed_path)

    def setup(self) -> str:
        if self.registration is None:
            self.__init_registration()
        return self.registration.setup()

    def add_tables(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
        accept_duplicates: bool = False,
    ) -> str:
        if self.registration is None:
            self.__init_registration()
        return self.registration.add_tables(
            path,
            creator,
            source,
            s3_region,
            s3_access_key,
            s3_secret_access_key,
            accept_duplicates,
        )

    def add_metadata(
        self,
        metadata_path: str,
        metadata_type: str = "",
        table_id: str = "",
    ) -> str:
        if self.registration is None:
            self.__init_registration()
        return self.registration.add_metadata(metadata_path, metadata_type, table_id)

    def summarize(self, table_id: str = "") -> str:
        if self.summarizer is None:
            self.__init_summarizer()
        return self.summarizer.summarize(table_id)

    def purge_tables(self) -> str:
        if self.summarizer is None:
            self.__init_summarizer()
        return self.summarizer.purge_tables()

    def generate_index(self, index_name: str, table_ids: list | tuple = None) -> str:
        if self.index_generator is None:
            self.__init_index_generator()
        return self.index_generator.generate_index(index_name, table_ids)

    def query_index(
        self,
        index_name: str,
        query: str,
        k: int = 1,
        n: int = 5,
        alpha: int = 0.5,
    ) -> str:
        if self.query is None:
            self.__init_query()
        return self.query.query(index_name, query, k, n, alpha)

    def sanity_check(self) -> str:
        return "This works!"


def main():
    print("Hello From Pneuma Main")
    fire.Fire(Pneuma)


if __name__ == "__main__":
    main()
