# ai-talk-core

ローカル音声を transcript に変換し、外部 agent に渡しやすい instruction / handoff を作る単体モジュールです。統合リポジトリ内では、音声入力フロントエンドと handoff 境界を担当します。

## 役割

- 入力: ローカル音声ファイル、固定時間マイク録音、ブラウザ録音
- 処理: Whisper 転写、軽量 VAD、無音トリム、mic-loop の partial/final 判定
- 出力: transcript、instruction draft、JSON/text handoff
- 境界: gesture、network adapter、agent backend の実装は持たず、外部側が `input_enabled` や保存済み handoff を介して接続する

## 初期セットアップ

```bash
uv sync
uv run python -m src.main --doctor
uv run python -m src.main data/sample_audio.mp3 --language ja --model small
```

Ubuntu では `ffmpeg` が必要です。マイク録音で `arecord` を使う場合は `alsa-utils` も入れてください。

```bash
sudo apt update
sudo apt install -y ffmpeg alsa-utils
```

Windows では `ffmpeg` を `PATH` に入れたうえで、必要に応じて helper を使います。

```powershell
.\start_web.ps1 -Sync
```

Whisper モデルは初回実行時に `models/whisper/` へ取得されます。モデルファイル、`.venv/`、`.cache/` はソースとして扱いません。

## dotenv / local config

通常のローカル起動では `.env` ファイルは必須ではありません。保護 API を固定 token で使う場合だけ、
起動プロセスの環境変数として `AI_TALK_CORE_WEB_TOKEN` を渡します。

```powershell
$env:AI_TALK_CORE_WEB_TOKEN = "local-dev-token"
```

実トークン、録音データ、Whisper モデル、`.cache/`、`.venv/` はコミットしません。

## 通常起動と確認

Web UI / JSON API:

```bash
uv run python -m src.web.app
```

起動後に `http://127.0.0.1:8000` を開きます。Windows では次でも起動できます。

```powershell
.\start_web.ps1
```

smoke 確認:

```bash
uv run python smoke_test.py
```

## CLI

ファイル転写:

```bash
uv run python -m src.main data/sample_audio.mp3 --language ja
```

マイク録音:

```bash
uv run python -m src.main --mic --duration 5 --language ja
```

マイクループ:

```bash
uv run python -m src.main --mic-loop --duration 3 --language ja
```

handoff JSON と text prompt を保存:

```bash
uv run python -m src.main --mic --duration 5 --language ja --handoff-output .cache/handoff/manual.json
```

診断:

```bash
uv run python -m src.main --doctor --doctor-format json
uv run python -m src.main --show-input-gate --input-gate-format json
uv run python -m src.main --list-mic-profiles
```

## Web UI と JSON API

Web UI は maintenance UI です。ファイルアップロード、ブラウザ録音、input gate、handoff 保存を扱います。

代表 API:

- `POST /api/transcribe-upload`: form field `audio_file`
- `POST /api/transcribe-browser-recording`: form field `audio_blob`
- `GET /api/health` / `GET /api/status`: server, STT, input gate, handoff 状態
- `GET|POST /api/input-gate`: 外部 adapter からの入力許可制御
- `GET /api/agent-handoff-latest?source=web`: 保存済み handoff の取得
- `POST /api/shutdown`: ローカル Web サーバーの停止要求

外部プロセスから保護 API を呼ぶ場合は、起動時に `AI_TALK_CORE_WEB_TOKEN` を設定し、`X-AI-Core-Token` ヘッダーで送ります。未設定時はプロセスごとのランダム token を Web UI が内部利用します。

```bash
AI_TALK_CORE_WEB_TOKEN=local-dev-token uv run python -m src.web.app
```

## 統合契約

- agent 側の主導線は `src.agent_handoff` と `src.agent_runner`
- `src.codex_handoff`, `src.codex_runner`, `/api/codex-handoff-latest` は互換入口
- gesture や keyboard などの外部入力は、このモジュールへ入る前に `input_enabled` へ変換する
- handoff は `.cache/codex/*.json` と同名 `.txt` prompt として保存される
- Web/API はローカルホスト向けであり、公開サーバーとして扱わない

詳細は [docs/integration-contract.md](docs/integration-contract.md) を参照してください。

## 文書マップ

- [MODULE_REQUIREMENTS.md](MODULE_REQUIREMENTS.md): 要求仕様
- [docs/module-responsibilities.md](docs/module-responsibilities.md): モジュールごとの責務境界
- [docs/integration-contract.md](docs/integration-contract.md): CLI/API/handoff の接続契約
- [docs/retired-paths.md](docs/retired-paths.md): 退避済み記録と保留導線の索引
- [AGENTS.md](AGENTS.md): Codex / contributor 向けの短い作業入口

通常の実装判断では `archive/` を読む必要はありません。必要なときだけ履歴確認として見ます。
