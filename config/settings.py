"""
全局配置入口：从 .env / 环境变量加载配置。

- 数据库连接配置：DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
- 源表列表：FM_SOURCE_TABLES（逗号分隔）
- 目标样本表：FM_TARGET_TABLE
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 运行环境（dev / stage / prod）
    env: str = Field("dev", alias="ENV")

    # --------------------------
    # PostgreSQL 连接信息
    # --------------------------
    db_host: str = Field("127.0.0.1", alias="DB_HOST")
    db_port: int = Field(5432, alias="DB_PORT")
    db_name: str = Field(..., alias="DB_NAME")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")

    # --------------------------
    # 样本表 & 源表配置
    # --------------------------
    # 逗号分隔的源表列表：例如 "market_price,market_price_25_12"
    fm_source_tables_raw: str = Field("market_price", alias="FM_SOURCE_TABLES")

    # 目标样本表：默认 fm_market_price
    fm_target_table: str = Field("fm_market_price", alias="FM_TARGET_TABLE")

    @property
    def fm_source_tables(self) -> List[str]:
        """
        返回拆分后的源表列表，已去掉空格和空字符串。
        示例：
            FM_SOURCE_TABLES=market_price,market_price_25_12

        -> ["market_price", "market_price_25_12"]
        """
        return [
            t.strip()
            for t in self.fm_source_tables_raw.split(",")
            if t.strip()
        ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """
    使用 lru_cache 确保全项目只初始化一次 Settings。
    """
    return Settings()


# 方便直接 from config.settings import settings
settings = get_settings()
