# Slot-Aware KV Offload Demo

This experimental policy admits reusable instruction, schema, context, system,
and unknown prompt blocks to KV offload storage while skipping personalized
slot-value blocks.

## Inspect the Prompt

```bash
python examples/features/slot_offload_demo.py \
  --model <model-name-or-local-path> \
  --block-size 16
```

The script prints each block's token range, decoded text, type, and
`STORE`/`SKIP` decision. It also writes `slot_request.json` for the HTTP demo.

## Start vLLM

```bash
vllm serve <model-name-or-local-path> \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 10737418240,
      "offload_prompt_only": true,
      "slot_offload_policy": "prefer_template",
      "slot_offload_store_types": [
        "instruction", "schema", "context", "system", "unknown"
      ],
      "slot_offload_log_decisions": true
    }
  }'
```

## Submit the Request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  --data @slot_request.json
```

The scheduler emits log lines like:

```text
SlotOffload request=... block=2 tokens=[32,48) type=schema decision=STORE ...
SlotOffload request=... block=3 tokens=[48,64) type=slot decision=SKIP ...
```

`STORE` is an admission decision. The offload backend may skip the physical
write if the block is already present. `SKIP` means the block is not copied to
the offload tier by this request; it is not a guarantee that the block remains
resident in GPU memory indefinitely.

For prefix reuse, place shared instructions and schema before personalized slot
values. vLLM can only load a consecutive matching prefix; a slot difference near
the beginning prevents later blocks from being prefix hits.
