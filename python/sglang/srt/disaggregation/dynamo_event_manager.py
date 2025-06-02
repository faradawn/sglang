from __future__ import annotations

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ctypes import (CDLL, POINTER, c_char_p, c_int64, c_size_t, c_uint32,
                    c_uint64)
import ctypes
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["KVCacheEventManager"]


class DynamoResult:
    OK = 0
    ERR = 1


class KVCacheEventManager:
    """Publish *stored* / *removed* KV-cache events to the Dynamo runtime."""

    def __init__(
        self,
        namespace: Optional[str] = None,
        component: Optional[str] = None,
        worker_id: Optional[int] = None,
        lib_path: Optional[str] = None,
        *,
        kv_block_size: int,
    ) -> None:
        # Allow users to rely on env vars (same as vLLM)
        namespace = namespace or os.getenv("VLLM_KV_NAMESPACE") or "default"
        component = component or os.getenv("VLLM_KV_COMPONENT") or "sglang"
        worker_id = worker_id if worker_id is not None else int(
            os.getenv("VLLM_WORKER_ID", "0"))
        lib_path = lib_path or os.getenv("VLLM_KV_CAPI_PATH")
        if lib_path is None:
            raise RuntimeError("VLLM_KV_CAPI_PATH environment variable not set")

        self._lib = CDLL(lib_path)

        # -------------  declare arg / return types  -----------------
        # init
        self._lib.dynamo_llm_init.argtypes = [
            c_char_p,  # namespace
            c_char_p,  # component
            c_int64,   # worker_id
            c_uint32,  # block_size
        ]
        self._lib.dynamo_llm_init.restype = c_uint32
        # publish stored
        self._lib.dynamo_kv_event_publish_stored.argtypes = [
            c_uint64,                       # event_id
            POINTER(c_uint32),             # token_ids
            POINTER(c_size_t),             # num_block_tokens per block (array)
            POINTER(c_uint64),             # block_hashes
            c_size_t,                      # num_blocks
            POINTER(c_uint64),             # parent_hash (can be NULL)
            c_uint64,                      # lora_id (0 if none)
        ]
        self._lib.dynamo_kv_event_publish_stored.restype = c_uint32
        # publish removed
        self._lib.dynamo_kv_event_publish_removed.argtypes = [
            c_uint64,               # event_id
            POINTER(c_uint64),      # block_hashes
            c_size_t,               # num_blocks
        ]
        self._lib.dynamo_kv_event_publish_removed.restype = c_uint32

        # -------------  call init  ----------------------------------
        rv = self._lib.dynamo_llm_init(namespace.encode(), component.encode(),
                                       worker_id, kv_block_size)
        if rv != DynamoResult.OK:
            raise RuntimeError(
                f"dynamo_llm_init failed (code={rv}) – check C-API library")
        logger.info(
            "Dynamo KV-event subsystem initialised (ns=%s component=%s id=%s)",
            namespace,
            component,
            worker_id,
        )

        self._event_id = 0

    # -----------------------------------------------------------------
    # Public helpers used by the publisher
    # -----------------------------------------------------------------
    def publish_stored(
        self,
        token_ids: List[int],
        block_hashes: List[int],
        parent_hash: Optional[int],
        block_size: int,
        lora_id: Optional[int] = None,
    ) -> None:
        num_blocks = 1  # SGLang RadixCache emits one block at a time

        token_arr = (c_uint32 * len(token_ids))(*token_ids)
        num_block_tokens_arr = (c_size_t * 1)(len(token_ids))
        block_hash_arr = (c_uint64 * 1)(block_hashes[0])
        parent_hash_ptr = (
            (c_uint64 * 1)(parent_hash) if parent_hash is not None else None
        )
        rv = self._lib.dynamo_kv_event_publish_stored(
            self._event_id,
            token_arr,
            num_block_tokens_arr,
            block_hash_arr,
            num_blocks,
            parent_hash_ptr,
            0 if lora_id is None else lora_id,
        )
        if rv != DynamoResult.OK:
            logger.debug("Failed to publish stored KV event – code %s", rv)
        self._event_id += 1

    def publish_removed(self, block_hashes: List[int]) -> None:
        num_blocks = len(block_hashes)
        hash_arr = (c_uint64 * num_blocks)(*block_hashes)
        rv = self._lib.dynamo_kv_event_publish_removed(
            self._event_id, hash_arr, num_blocks)
        if rv != DynamoResult.OK:
            logger.debug("Failed to publish removed KV event – code %s", rv)
        self._event_id += 1