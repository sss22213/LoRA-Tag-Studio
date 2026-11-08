#!/bin/bash
# AI Toolkit 容器的啟動腳本：一定要有密碼，並把工作資料庫放進掛載的 state 資料夾，
# 重建容器後訓練紀錄還在（資料庫路徑是寫死的，不能直接掛單一檔案：檔案不存在時 Docker 會建成資料夾）。
set -e
if [ -z "$AI_TOOLKIT_AUTH" ]; then
  echo "AI_TOOLKIT_AUTH is empty: set a password in .env (LoRA Tag Studio uses the same one)." >&2
  exit 1
fi
cd /app/ai-toolkit
if [ ! -L aitk_db.db ]; then
  if [ ! -f state/aitk_db.db ]; then
    cp aitk_db.db state/aitk_db.db
  fi
  rm -f aitk_db.db
  ln -s state/aitk_db.db aitk_db.db
fi
# 換了新版映像時，資料庫結構要跟著更新
(cd ui && npx prisma db push --skip-generate >/dev/null) || echo "prisma db push failed, continuing" >&2
exec /start.sh
