"""
Script to load and inspect debug dumps from prior > marginal bug occurrences.

Usage:
    python inspect_debug_dump.py                    # Load most recent dump
    python inspect_debug_dump.py dump_0001.pt      # Load specific dump
    python inspect_debug_dump.py --list            # List all dumps
"""
from pathlib import Path

import argparse
import torch
import torch.nn.functional as F
# import numpy as np

from FL_code.cancer_protocol.cancer_quantizer import WZQuantizerCancer
from FL_code.cancer_protocol.prior_calculator import PriorCalculator
from FL_code.cancer_protocol import CancerConfig
# from FL_reworked.brent_wz_models import EncoderDecoderLayeredRNN


class DebugInspector:
    """Class to load and inspect debug dumps."""

    def __init__(self, dump_path: str):
        self.dump_path = Path(dump_path)
        self.data = torch.load(self.dump_path, weights_only=False)
        print(f"Loaded debug dump from: {self.dump_path}")

    def print_summary(self):
        """Print summary of the debug dump."""
        d = self.data
        print("\n" + "="*70)
        print("DEBUG DUMP SUMMARY")
        print("="*70)
        print(f"Round: {d['round_id']}, Client: {d['client_id']}, Type: {d['round_type']}, Phase: {d['phase']}")
        print(f"Bits per plane: {d['bins_per_plane']}, Num planes: {d['num_planes']}")
        print(f"\n>>> Prior rate: {d['prior_rate']:.4f} bpp")
        print(f">>> Marginal rate: {d['marginal_rate']:.4f} bpp")
        print(f">>> Difference: {d['prior_rate'] - d['marginal_rate']:.4f} (should be negative!)")
        print(f"\nDelta vec shape: {d['delta_vec'].shape}")
        print(f"Bins shape: {d['bins'].shape}")
        print(f"Unique bins per plane: {[torch.unique(d['bins'][i]).numel() for i in range(d['bins'].shape[0])]}")
        print(f"\nSide info used: {len(d['side_info_list_used']) if isinstance(d['side_info_list_used'], list) else d['side_info_list_used']}")
        print(f"Hash of delta: {d['hash_of_delta']}")
        print(f"Hash in cache: {d['hash_in_cache']}")
        print(f"\nQuantizer config: {d['quantizer_config']}")
        print(f"Cached prior keys: {d['cached_priors_dict_keys']}")

    def analyze_prior_distribution(self):
        """Analyze the prior probability distribution."""
        d = self.data
        prior = d['prior'].float()
        bins = d['bins']
        num_planes = d['num_planes']

        print("\n" + "="*70)
        print("PRIOR DISTRIBUTION ANALYSIS")
        print("="*70)

        for plane_idx in range(num_planes):
            plane_prior = prior[plane_idx]
            plane_bins = bins[plane_idx]

            # Get probability of actual bins
            actual_bin_probs = plane_prior[torch.arange(plane_prior.shape[0]), plane_bins.long()]

            print(f"\nPlane {plane_idx}:")
            print(f"  Prior shape: {plane_prior.shape}")
            print(f"  Prior mean per bin: {plane_prior.mean(dim=0).tolist()}")
            print(f"  Prior min: {plane_prior.min().item():.8f}")
            print(f"  Prior max: {plane_prior.max().item():.8f}")
            print(f"  ")
            print(f"  Prob of actual bin - mean: {actual_bin_probs.mean().item():.6f}")
            print(f"  Prob of actual bin - min: {actual_bin_probs.min().item():.8f}")
            print(f"  Prob of actual bin - max: {actual_bin_probs.max().item():.6f}")
            print(f"  Prob of actual bin - median: {actual_bin_probs.median().item():.6f}")

            # Count problematic samples
            for threshold in [0.001, 0.01, 0.05, 0.1]:
                count = (actual_bin_probs < threshold).sum().item()
                pct = 100 * count / len(actual_bin_probs)
                print(f"  Samples with prob < {threshold}: {count} ({pct:.2f}%)")

            # Rate contribution
            rate = -torch.log2(actual_bin_probs.clamp(min=1e-8)).mean().item()
            print(f"  Rate contribution: {rate:.4f} bpp")

            # Marginal rate for this plane
            bin_counts = torch.bincount(plane_bins.long(), minlength=d['bins_per_plane']).float()
            marginal_probs = bin_counts / len(plane_bins)
            marginal_rate = -torch.log2(marginal_probs[plane_bins.long()].clamp(min=1e-8)).mean().item()
            print(f"  Marginal rate: {marginal_rate:.4f} bpp")
            print(f"  Difference: {rate - marginal_rate:.4f}")

    def analyze_marginal_distribution(self):
        """Analyze the marginal (bin count) distribution."""
        d = self.data
        bins = d['bins']
        bins_per_plane = d['bins_per_plane']

        print("\n" + "="*70)
        print("MARGINAL (BIN COUNT) DISTRIBUTION")
        print("="*70)

        for plane_idx in range(d['num_planes']):
            plane_bins = bins[plane_idx]
            bin_counts = torch.bincount(plane_bins.long(), minlength=bins_per_plane)
            probs = bin_counts.float() / len(plane_bins)

            print(f"\nPlane {plane_idx}:")
            print(f"  Bin counts: {bin_counts.tolist()}")
            print(f"  Bin probs: {[f'{p:.4f}' for p in probs.tolist()]}")
            print(f"  Entropy: {-(probs * torch.log2(probs.clamp(min=1e-8))).sum().item():.4f} bits")

    def analyze_side_info(self):
        """Analyze the side information."""
        d = self.data

        print("\n" + "="*70)
        print("SIDE INFORMATION ANALYSIS")
        print("="*70)

        si_list = d['side_info_list_used']
        if not isinstance(si_list, list) or len(si_list) == 0:
            print(f"Side info: {si_list}")
            return

        delta = d['delta_vec']

        print(f"Number of SI tensors: {len(si_list)}")
        for i, si in enumerate(si_list):
            print(f"\nSI[{i}]:")
            print(f"  Shape: {si.shape}")
            print(f"  Mean: {si.float().mean().item():.6f}")
            print(f"  Std: {si.float().std().item():.6f}")
            print(f"  Min: {si.float().min().item():.6f}")
            print(f"  Max: {si.float().max().item():.6f}")

            # Correlation with delta
            if si.shape[0] == delta.shape[0]:
                corr = torch.corrcoef(torch.stack([delta.float(), si.float()]))[0, 1].item()
                print(f"  Correlation with delta: {corr:.6f}")

    def analyze_history(self):
        """Analyze reconstruction history."""
        d = self.data

        print("\n" + "="*70)
        print("RECONSTRUCTION HISTORY")
        print("="*70)

        print("\nClient past reconstructions:")
        for cid, reconst_list in enumerate(d['client_past_reconst']):
            print(f"  Client {cid}: {len(reconst_list)} reconstructions")

        print("\nServer past reconstructions:")
        for cid, reconst_list in enumerate(d['srvr_past_reconst']):
            print(f"  Client {cid}: {len(reconst_list)} reconstructions")

    def recreate_quantizer(self) -> WZQuantizerCancer:
        """Recreate the quantizer from saved state."""
        d = self.data
        cfg = d['quantizer_config']

        c_cfg = CancerConfig()

        # Determine si_size from saved side info
        si_list = d['side_info_list_used']
        si_size = len(si_list) if isinstance(si_list, list) else 0

        quantizer = WZQuantizerCancer(
            c_cfg=c_cfg,
            num_planes=cfg['num_planes'],
            bins_per_plane=cfg['bins_per_plane'],
            si_size=si_size,
            norm_slices=cfg['vec_slices'],
            outlier_threshold=cfg['outlier_threshold'],
        )

        # Load model state
        quantizer.coding_model.load_state_dict(d['quantizer_state_dict'])
        quantizer.wmspe_denom = cfg['wmspe_denom']
        quantizer.si_vec_size = cfg['si_vec_size']
        quantizer.side_info_list_used = si_list

        print("\nRecreated quantizer from saved state.")
        return quantizer

    def recompute_priors(self):
        """Recompute priors using saved state to verify."""
        d = self.data

        print("\n" + "="*70)
        print("RECOMPUTING PRIORS")
        print("="*70)

        quantizer = self.recreate_quantizer()
        bins = d['bins']
        delta_vec = d['delta_vec']

        # Get side info
        si_list = d['side_info_list_used']
        if isinstance(si_list, list) and len(si_list) > 0:
            si_trans = torch.stack(si_list).T.float()
            # Apply preprocessing if needed
            if quantizer.vec_slices not in [False, None]:
                si_trans_list = []
                for si in si_list:
                    preprocessed, _, _ = quantizer._apply_pre_process(si.float(), force_no_outlier_handling=True)
                    si_trans_list.append(preprocessed)
                si_trans = torch.stack(si_trans_list).T.float()
        else:
            si_trans = torch.zeros(bins.shape[1], 1)

        print(f"SI trans shape: {si_trans.shape}")

        # Compute priors using coding model directly
        quantizer.coding_model.cuda().eval()
        codes = [F.one_hot(b.long(), num_classes=quantizer.bins_per_plane).float().cuda()
                 for b in bins]

        priors = quantizer._get_posterior(delta_vec, bins)
        # priors = d['prior']

        prior_tensor = torch.stack([p.cpu() for p in priors])

        # Compute rate
        recomputed_rate = PriorCalculator.compute_rate_from_prior_tensor(prior_tensor, bins, quantizer.num_planes)

        print(f"\nOriginal prior rate: {d['prior_rate']:.4f}")
        print(f"Recomputed prior rate: {recomputed_rate:.4f}")
        print(f"Difference: {abs(d['prior_rate'] - recomputed_rate):.6f}")

        # Compare with saved prior
        saved_prior = d['prior'].float()
        diff = (prior_tensor - saved_prior).abs()
        print(f"\nMax diff between saved and recomputed prior: {diff.max().item():.8f}")
        print(f"Mean diff: {diff.mean().item():.8f}")

        return quantizer, prior_tensor

    def interactive_debug(self):
        """Drop into interactive debugging mode with all variables loaded."""
        d = self.data

        # Extract commonly needed variables
        delta_vec = d['delta_vec']
        bins = d['bins']
        prior = d['prior']
        m_prior = d['m_prior']
        si_list = d['side_info_list_used']

        quantizer = self.recreate_quantizer()

        print("\n" + "="*70)
        print("INTERACTIVE DEBUG MODE")
        print("="*70)
        print("Available variables:")
        print("  d          - Full debug data dict")
        print("  delta_vec  - Input delta vector")
        print("  bins       - Encoded bins")
        print("  prior      - Prior distribution tensor")
        print("  m_prior    - Marginal prior tensor")
        print("  si_list    - Side information list")
        print("  quantizer  - Recreated quantizer object")
        print("  inspector  - This DebugInspector instance")
        print("\nUseful methods:")
        print("  inspector.analyze_prior_distribution()")
        print("  inspector.analyze_side_info()")
        print("  inspector.recompute_priors()")
        print("\nDropping into IPython shell...")

        try:
            from IPython import embed
            embed()
        except ImportError:
            import code
            code.interact(local=locals())


def list_dumps(debug_dir: Path):
    """List all available debug dumps."""
    dumps = sorted(debug_dir.glob("dump_*.pt"))
    if not dumps:
        print("No debug dumps found.")
        return

    print(f"\nFound {len(dumps)} debug dump(s):")
    print("-" * 70)
    for dump_path in dumps:
        try:
            data = torch.load(dump_path, weights_only=False)
            print(f"  {dump_path.name}: R{data['round_id']}C{data['client_id']} "
                  f"[{data['round_type']}] prior={data['prior_rate']:.2f} marg={data['marginal_rate']:.2f}")
        except Exception as e:
            print(f"  {dump_path.name}: Error loading - {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect debug dumps from prior > marginal occurrences")
    parser.add_argument("dump_file", nargs="?", default=None, help="Specific dump file to load")
    parser.add_argument("--list", action="store_true", help="List all available dumps")
    parser.add_argument("--interactive", "-i", action="store_true", help="Drop into interactive mode")
    args = parser.parse_args()

    debug_dir = Path(__file__).parent / "debug_dumps"

    if args.list:
        list_dumps(debug_dir)
        return

    # Determine which dump to load
    if args.dump_file:
        dump_path = debug_dir / args.dump_file if not Path(args.dump_file).is_absolute() else Path(args.dump_file)
    else:
        # Load most recent dump
        dumps = sorted(debug_dir.glob("dump_*.pt"))
        if not dumps:
            print("No debug dumps found. Run the workflow to generate dumps when prior > marginal occurs.")
            return
        dump_path = dumps[-1]
        print(f"Loading most recent dump: {dump_path.name}")

    if not dump_path.exists():
        print(f"Dump file not found: {dump_path}")
        return

    # Create inspector and run analysis
    inspector = DebugInspector(dump_path)
    inspector.print_summary()
    inspector.analyze_prior_distribution()
    inspector.analyze_marginal_distribution()
    inspector.analyze_side_info()
    inspector.analyze_history()
    inspector.recompute_priors()

    if args.interactive:
        inspector.interactive_debug()


if __name__ == "__main__":
    main()
