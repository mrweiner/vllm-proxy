# vLLM Proxy + Auto-Continue Watcher

Sits between opencode and a remote vLLM instance. Repairs truncated JSON in tool calls, injects system prompt policies, and auto-continues sessions when responses are cut off.

## Requirements

- Python 3.12+
- opencode server running (default: `http://localhost:4096`)
- vLLM backend reachable (default: `127.0.0.1:8000`)

## Usage

```bash
# Quick start (uses config.toml defaults)
python3 start.py -v

# Custom ports
python3 start.py --listen-port 4097 --vllm-port 8000 -v
```

## Config

Edit `config.toml` or override with env vars:

| Setting | Env Var | Default |
|---------|---------|---------|
| `proxy.listen_port` | `VLLM_PROXY_LISTEN_PORT` | 4097 |
| `proxy.vllm_port` | `VLLM_PROXY_VLLM_PORT` | 8000 |
| `proxy.vllm_host` | `VLLM_PROXY_VLLM_HOST` | 127.0.0.1 |
| `watcher.base_url` | `OC_BASE_URL` | http://localhost:4096 |
| `policies.system_prompts` | — | TDD + plan-mode |

Precedence: CLI flags > env vars > config.toml > hardcoded defaults.

## Tests

```bash
python3 run_tests.py
```
