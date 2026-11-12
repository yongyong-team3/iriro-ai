"""PyCharm에서 이 파일을 실행하면 로컬 AI 서버와 접속 로그가 시작됩니다."""

import os
from pathlib import Path

import uvicorn


if __name__ == "__main__":
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() and not key.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=int(os.getenv("AI_SERVER_PORT", "8765")),
        reload=True,
        access_log=True,
        log_level="info",
    )
