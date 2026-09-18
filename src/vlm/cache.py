"""
Atomic, collision-resistant disk caching for VLM generated responses.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional, Union

from src.vlm.provider import VLMResponse, VLMGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheStatistics:
    """Tracking statistics for cache access during execution."""
    hits: int = 0
    misses: int = 0
    writes: int = 0
    corrupt_entries: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "corrupt_entries": self.corrupt_entries,
        }


class VLMCache:
    """
    Disk cache keyed by image content, provider kind, synthetic/real provenance,
    model revision, prompt hash, and generation config.
    Guarantees atomic file writes, cache separation, and corrupt entry rejection.
    """

    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStatistics()

    @staticmethod
    def compute_cache_key(
        image_hash: str,
        model_name: str,
        model_revision: str,
        prompt: str,
        gen_config: VLMGenerationConfig,
        provider_kind: str = "synthetic_fixture",
        is_synthetic: bool = True,
    ) -> str:
        """
        Compute a SHA256 cache key that changes whenever any input, prompt, model snapshot,
        provider kind, synthetic provenance, or generation parameter changes.
        """
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        key_dict = {
            "cache_version": "v2",
            "image_hash": image_hash,
            "provider_kind": provider_kind,
            "is_synthetic": is_synthetic,
            "model_name": model_name,
            "model_revision": model_revision,
            "prompt_hash": prompt_hash,
            "gen_param_hash": gen_config.get_param_hash(),
        }
        serialized = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_path_for_key(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        image_hash: str,
        model_name: str,
        model_revision: str,
        prompt: str,
        gen_config: VLMGenerationConfig,
        provider_kind: str = "synthetic_fixture",
        is_synthetic: bool = True,
    ) -> Optional[VLMResponse]:
        """
        Retrieve cached response if it exists and passes data integrity validation.
        """
        cache_key = self.compute_cache_key(
            image_hash=image_hash,
            model_name=model_name,
            model_revision=model_revision,
            prompt=prompt,
            gen_config=gen_config,
            provider_kind=provider_kind,
            is_synthetic=is_synthetic,
        )
        cache_path = self._get_path_for_key(cache_key)

        if not cache_path.exists():
            self.stats.misses += 1
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate essential fields
            if not isinstance(data, dict):
                raise ValueError("Cache entry is not a JSON object")

            response = VLMResponse.from_dict(data)

            # Verify integrity and provenance isolation
            if response.image_hash != image_hash:
                raise ValueError("Image hash mismatch in cache file")
            if response.model_name != model_name:
                raise ValueError("Model name mismatch in cache file")
            if response.is_synthetic != is_synthetic:
                raise ValueError("Synthetic/real provenance mismatch in cache file")
            if response.provider_kind != provider_kind:
                raise ValueError("Provider kind mismatch in cache file")

            self.stats.hits += 1
            return response
        except Exception as e:
            logger.warning(f"Detected corrupt or invalid cache entry at {cache_path}: {e}")
            self.stats.corrupt_entries += 1
            self.stats.misses += 1
            return None

    def put(self, response: VLMResponse, gen_config: VLMGenerationConfig, overwrite: bool = False) -> Path:
        """
        Atomically write a valid VLMResponse to disk.
        """
        if not response.caption or not response.response_id:
            raise ValueError("Refusing to cache empty or invalid response record")

        cache_key = self.compute_cache_key(
            image_hash=response.image_hash,
            model_name=response.model_name,
            model_revision=response.model_revision,
            prompt=response.prompt,
            gen_config=gen_config,
            provider_kind=response.provider_kind,
            is_synthetic=response.is_synthetic,
        )
        target_path = self._get_path_for_key(cache_key)

        if target_path.exists() and not overwrite:
            return target_path

        # Atomic write: write to temp file on same filesystem, then rename
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(self.cache_dir),
            delete=False,
            encoding="utf-8",
            suffix=".tmp"
        )
        try:
            json.dump(response.to_dict(), temp_file, indent=2)
            temp_file.flush()
            temp_file.close()
            Path(temp_file.name).replace(target_path)
            self.stats.writes += 1
        except Exception:
            if Path(temp_file.name).exists():
                Path(temp_file.name).unlink()
            raise

        return target_path
