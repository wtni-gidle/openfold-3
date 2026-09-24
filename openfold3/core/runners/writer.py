# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module for containing writing tools and callbacks for model outputs."""

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.data.prepared_bundle import sanitise_job_name
from openfold3.core.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


@contextmanager
def atomic_output_path(path: Path):
    """Yield a sibling temporary path and publish it with an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict) -> None:
    with atomic_output_path(path) as temporary_path:
        temporary_path.write_text(
            json.dumps(value, indent=4, cls=NumpyEncoder), encoding="utf-8"
        )


class NumpyEncoder(json.JSONEncoder):
    r"""Custom JSON encoder for handling numpy data types.

    https://gist.github.com/jonathanlurie/1b8d12f938b400e54c1ed8de21269b65
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float16:
                return obj.astype(np.float64).round(3).tolist()
            elif obj.dtype == np.float32:
                return obj.astype(np.float64).round(6).tolist()
            return obj.tolist()
        elif isinstance(obj, np.generic):
            if obj.dtype == np.float16:
                return round(obj.item(), 3)
            elif obj.dtype == np.float32:
                return round(obj.item(), 6)
            return obj.item()
        return super().default(obj)


def _take_batch_dim(x, b: int):
    if isinstance(x, torch.Tensor):
        if len(x.shape) > 1:
            return x[b].cpu().float().numpy()
        else:
            return x
    if isinstance(x, dict):
        return {k: _take_batch_dim(v, b) for k, v in x.items()}
    return x


def _take_sample_dim(x, s: int):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.shape[0] == 1:
            return x[0]
        return x[s]
    if isinstance(x, dict):
        return {k: _take_sample_dim(v, s) for k, v in x.items()}
    return x


class OF3OutputWriter(BasePredictionWriter):
    """Callback for writing AF3 predicted structure and confidence outputs"""

    def __init__(
        self,
        output_dir: Path,
        structure_format: str = "pdb",
        full_confidence_output_format: str | None = None,
        full_confidence_output_dtype: Literal["float32", "float16"] = "float16",
        write_features: bool = False,
        write_latent_outputs: bool = False,
        write_full_confidence_scores: bool = True,
        summary_dir: Path | None = None,
        compress_full_confidence: bool | None = None,
    ):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.structure_format = structure_format
        selected = "npz" if compress_full_confidence else "json"
        if compress_full_confidence is not None and full_confidence_output_format is not None and full_confidence_output_format != selected:
            raise ValueError("compress_full_confidence conflicts with full_confidence_output_format")
        self.full_confidence_format = full_confidence_output_format or selected
        self.full_confidence_dtype = np.dtype(full_confidence_output_dtype)
        self.write_features = write_features
        self.write_latent_outputs = write_latent_outputs
        self.write_full_confidence_scores = write_full_confidence_scores
        self.summary_dir = (
            Path(summary_dir) if summary_dir is not None else self.output_dir
        )

        # Track successfully predicted samples
        self.success_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.failed_queries = []

    def on_predict_start(self, trainer, pl_module):
        """Reset counters when one runner executes multiple missing-seed groups."""
        self.success_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.failed_queries = []

    @staticmethod
    def write_structure_prediction(
        atom_array: structure.AtomArray,
        predicted_coords: np.ndarray,
        plddt: np.ndarray,
        output_file: Path,
        make_ost_compatible: bool = True,
    ):
        """Writes predicted coordinates to atom_array and writes mmcif file to disk.

        pLDDT scores are written to the B-factor column of the output file.
        """

        # Set coordinates and plddt scores
        atom_array.coord = predicted_coords
        atom_array.set_annotation("b_factor", plddt)

        # Write the output file
        logger.info(f"Writing predicted structure to {output_file}")
        write_structure(
            atom_array,
            output_file,
            include_bonds=True,
            make_ost_compatible=make_ost_compatible,
        )

    def get_pae_confidence_scores(self, confidence_scores, atom_array):
        pae_confidence_scores = {}
        single_value_keys = [
            "iptm",
            "ptm",
            "disorder",
            "has_clash",
            "sample_ranking_score",
        ]

        for key in single_value_keys:
            pae_confidence_scores[key] = confidence_scores[key]

        # Get map from asym id to chain id
        renum_ids = np.unique(atom_array.chain_id, return_inverse=True)[1] + 1
        asym_id_to_chain_id = {
            k: v
            for (k, v) in set(
                [
                    (int(x[0]), str(x[1]))
                    for x in zip(renum_ids, atom_array.chain_id, strict=True)
                ]
            )
        }

        # Asym id -> chain id for chain_ptm
        pae_confidence_scores["chain_ptm"] = {
            asym_id_to_chain_id[int(k)]: v
            for k, v in confidence_scores["chain_ptm"].items()
        }

        # Asym id -> chain id for chain_pair_iptm
        pae_confidence_scores["chain_pair_iptm"] = {}
        for k, v in confidence_scores["chain_pair_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["chain_pair_iptm"][f"({k1}, {k2})"] = v

        # Asym id -> chain id for bespoke_iptm
        pae_confidence_scores["bespoke_iptm"] = {}
        for k, v in confidence_scores["bespoke_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["bespoke_iptm"][f"({k1}, {k2})"] = v
        return pae_confidence_scores

    def write_confidence_scores(
        self,
        confidence_scores: dict[str, np.ndarray],
        atom_array: structure.AtomArray,
        summary_prefix: Path,
        full_prefix: Path,
        seed: int | None = None,
        sample_index: int | None = None,
    ):
        """Writes confidence scores to disk"""
        plddt = confidence_scores["plddt"]
        pde = confidence_scores["pde"]
        gpde = confidence_scores["gpde"]
        pae = confidence_scores["pae"]
        aggregated_confidence_scores = {"avg_plddt": np.mean(plddt), "gpde": gpde}
        if seed is not None:
            aggregated_confidence_scores["seed"] = seed
        if sample_index is not None:
            aggregated_confidence_scores["sample"] = sample_index

        logger.info("Recording PAE confidence outputs")
        aggregated_confidence_scores |= self.get_pae_confidence_scores(
            confidence_scores, atom_array
        )

        out_file_agg = Path(f"{summary_prefix}_summary_confidences.json")
        atomic_write_json(out_file_agg, aggregated_confidence_scores)

        # Full confidence scores
        if self.write_full_confidence_scores is True:
            full_confidence_scores = {"plddt": plddt, "pde": pde, "pae": pae}
            out_fmt = self.full_confidence_format
            out_file_full = Path(f"{full_prefix}_full_data.{out_fmt}")

            if out_fmt == "json":
                atomic_write_json(out_file_full, full_confidence_scores)
            elif out_fmt == "npz":
                for key, val in full_confidence_scores.items():
                    if (
                        isinstance(val, np.ndarray)
                        and val.dtype != self.full_confidence_dtype
                    ):
                        full_confidence_scores[key] = val.astype(
                            self.full_confidence_dtype
                        )
                with atomic_output_path(out_file_full) as temporary_path:
                    np.savez_compressed(
                        temporary_path,
                        **full_confidence_scores,
                    )
            out_file_full.with_suffix(".json" if out_fmt == "npz" else ".npz").unlink(missing_ok=True)

    def write_all_outputs(self, batch: dict, outputs: dict, confidence_scores: dict):
        """Writes all outputs for a given batch."""

        batch_size = len(batch["atom_array"])
        sample_size = outputs["atom_positions_predicted"].shape[1]

        # Iterate over all predictions in the batch
        for b in range(batch_size):
            seed_value = batch["seed"][b]
            seed = int(seed_value.item() if hasattr(seed_value, "item") else seed_value)
            query_id = sanitise_job_name(str(batch["query_id"][b]))
            query_output_dir = Path(self.output_dir) / query_id

            # Extract attributes for the current batch
            atom_array_batch = batch["atom_array"][b]
            predicted_coords_batch = (
                outputs["atom_positions_predicted"][b].cpu().float().numpy()
            )
            confidence_scores_batch = _take_batch_dim(confidence_scores, b)

            # Iterate over all diffusion samples
            for s in range(sample_size):
                sample_prefix = f"seed-{seed}_sample-{s}"

                confidence_scores_sample = _take_sample_dim(confidence_scores_batch, s)
                predicted_coords_sample = predicted_coords_batch[s]

                # Save predicted structure
                structure_file = (
                    query_output_dir
                    / "models"
                    / f"{sample_prefix}_model.{self.structure_format}"
                )
                with atomic_output_path(structure_file) as temporary_path:
                    self.write_structure_prediction(
                        atom_array=atom_array_batch,
                        predicted_coords=predicted_coords_sample,
                        plddt=confidence_scores_sample["plddt"],
                        output_file=temporary_path,
                    )

                # Save confidence metrics
                self.write_confidence_scores(
                    confidence_scores=confidence_scores_sample,
                    summary_prefix=(
                        query_output_dir / "summary_confidences" / sample_prefix
                    ),
                    full_prefix=(query_output_dir / "full_data" / sample_prefix),
                    atom_array=atom_array_batch,
                    seed=seed,
                    sample_index=s,
                )

            def fetch_cur_batch(t):
                # Get tensor for current batch dim
                # Remove expanded sample dim if it exists to get original tensor shapes
                if t.ndim < 2:
                    return t

                cur_feats = t[b : b + 1].squeeze(1)  # noqa: B023
                return cur_feats.detach().clone().cpu()

            # Write out input feature dictionary
            if self.write_features:
                out_file = query_output_dir / "features" / f"seed-{seed}_features.pt"
                cur_batch = tensor_tree_map(fetch_cur_batch, batch, strict_type=False)
                with atomic_output_path(out_file) as temporary_path:
                    torch.save(cur_batch, temporary_path)
                del cur_batch

            # Write out latent reps / raw model outputs
            if self.write_latent_outputs:
                out_file = (
                    query_output_dir / "latents" / f"seed-{seed}_latent_outputs.pt"
                )
                cur_output = tensor_tree_map(
                    fetch_cur_batch, outputs, strict_type=False
                )
                with atomic_output_path(out_file) as temporary_path:
                    torch.save(cur_output, temporary_path)
                del cur_output

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # Skip repeated samples
        if batch.get("repeated_sample"):
            return

        self.total_count += 1

        # Skip and track failed samples
        if outputs is None:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            return

        batch, outputs = outputs
        confidence_scores = outputs["confidence_scores"]

        # Write predictions and confidence scores
        # Optionally write out input features and latent outputs
        try:
            self.write_all_outputs(
                batch=batch,
                outputs=outputs,
                confidence_scores=confidence_scores,
            )
            self.success_count += 1
        except Exception as e:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            logger.exception(
                f"Failed to write predictions for query_id(s) "
                f"{', '.join(batch['query_id'])}: {e}"
            )

        del batch, outputs

    def on_predict_end(self, trainer, pl_module):
        """
        Print summary of inference run. Includes a timeout failsafe
        for distributed runs.
        """

        try:
            # Gather summary data from all processes
            final_summary_data = {
                "total": self.total_count,
                "success": self.success_count,
                "failed": self.failed_count,
                "failed_queries": self.failed_queries,
            }

            gathered_data = [final_summary_data]
            if dist.is_available() and dist.is_initialized():
                gathered_data = [None] * trainer.world_size
                dist.all_gather_object(gathered_data, final_summary_data)

            # Every rank must report failure, including ranks whose own queries
            # all succeeded. all_gather_object provides the same totals to each.
            total_queries = sum(data["total"] for data in gathered_data)
            success_count = sum(data["success"] for data in gathered_data)
            failed_count = sum(data["failed"] for data in gathered_data)
            final_failed_list = [
                item for data in gathered_data for item in data["failed_queries"]
            ]

            if trainer.is_global_zero:
                # Best effort under distributed process teardown: another rank
                # may raise first. Avoid a barrier that could hang if writing
                # this summary fails; each rank's exception carries the totals.
                self._write_summary(
                    total_queries=total_queries,
                    success_count=success_count,
                    failed_count=failed_count,
                    failed_list=final_failed_list,
                    global_rank=trainer.global_rank,
                    is_complete=True,
                )
        except RuntimeError as e:
            # TODO: Due to additional sync PL does outside of this callback,
            #  this won't be reached before the timeout error occurs.
            #  Leaving this here for now in case we refactor the prediction
            #  logic to avoid the extra syncs.
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                logger.warning(
                    f"[Rank {trainer.global_rank}] Distributed sync timed out! "
                    f"Writing local results to a fallback log."
                )

                self._write_summary(
                    total_queries=self.total_count,
                    success_count=self.success_count,
                    failed_count=self.failed_count,
                    failed_list=self.failed_queries,
                    global_rank=trainer.global_rank,
                    is_complete=False,
                )

            # A fallback summary is diagnostic, not a successful prediction run.
            # Preserve the original collective failure and traceback as well as
            # unexpected runtime errors.
            raise

        if failed_count > 0:
            failed_queries = ", ".join(sorted(set(final_failed_list)))
            raise RuntimeError(
                f"OpenFold3 prediction failed: {failed_count} of "
                f"{total_queries} queries failed; {success_count} successful. "
                f"Failed queries: {failed_queries}"
            )
        if total_queries > 0 and success_count == 0:
            raise RuntimeError("OpenFold3 produced no successful prediction outputs")

    def _write_summary(
        self,
        total_queries: int,
        success_count: int,
        failed_count: int,
        failed_list: list,
        global_rank: int = 0,
        is_complete=True,
    ):
        """Helper to format the final summary."""
        if is_complete:
            status = "COMPLETE"
            out_file = self.summary_dir / "summary.txt"
        else:
            status = f"INCOMPLETE (Rank {global_rank})"
            out_file = self.summary_dir / f"fallback_summary_rank_{global_rank}.txt"

        summary = [
            "\n" + "=" * 50,
            f"    PREDICTION SUMMARY ({status})    ",
            "=" * 50,
            f"Total Queries Processed: {total_queries}",
            f"  - Successful Queries:  {success_count}",
            f"  - Failed Queries:      {failed_count}",
        ]

        if failed_list:
            failed_str = ", ".join(sorted(list(set(failed_list))))
            summary.append(f"\nFailed Queries: {failed_str}")

        summary.append("=" * 50 + "\n")
        summary = "\n".join(summary)

        with atomic_output_path(out_file) as temporary_path:
            temporary_path.write_text(summary)

        if is_complete:
            print(summary)
        else:
            logger.warning(
                f"Fallback summary for Rank {global_rank} saved to: {out_file}"
            )
