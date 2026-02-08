# add to the qunatizer protocol file for recording debug data

class CancerRecord(CompressionRecord):
    def _compress(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict:
        if record.round_type != 'F':
            self._train_quantizer_or_load(delta_vec, record)

        # Encode using current quantizer
        quantizer = self.frozen_quantizers[record.client_id]
        bins, prep_metadata = quantizer.encoding_process(delta_vec)

        # Build payload
        payload = self._build_payload((bins, prep_metadata), quantizer, record)

        # Add prior info to record for analysis
        prior = quantizer._get_posterior(delta_vec, bins_vec_save_compute=bins)
        record.prior_rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)

        # Compute marginal prior for comparison
        m_prior = PriorCalculator.compute_marginal_prior(
            bins, quantizer.bins_per_plane, quantizer.num_planes)
        record.marginal_rate = PriorCalculator.compute_rate_from_prior_tensor(m_prior, bins, quantizer.num_planes)

        # Debug: Save state when prior > marginal for investigation
        if record.prior_rate > record.marginal_rate:
            self._save_debug_state(
                delta_vec, bins, prep_metadata, prior, m_prior,
                quantizer, record, self.client_past_reconst, self.srvr_past_reconst
            )

        return payload


    def _save_debug_state(self, delta_vec, bins, prep_metadata, prior, m_prior,
                          quantizer, record, client_past_reconst, srvr_past_reconst):
        """Save all relevant state when prior > marginal for debugging."""
        from pathlib import Path

        debug_dir = Path("experiments/debuging/debug_dumps")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Find next available dump number
        existing = list(debug_dir.glob("dump_*.pt"))
        dump_num = len(existing)
        dump_path = debug_dir / f"dump_{dump_num:04d}.pt"

        print(f"\n⚠️ PRIOR > MARGINAL detected! Saving debug state to {dump_path}")
        print(f"   Round {record.round_id}, Client {record.client_id}, Type {record.round_type}")
        print(f"   Prior: {record.prior_rate:.4f}, Marginal: {record.marginal_rate:.4f}")

        # Gather all important data
        debug_data = {
            # Record info
            'round_id': record.round_id,
            'client_id': record.client_id,
            'round_type': record.round_type,
            'phase': record.phase,
            'bins_per_plane': record.bins_per_plane,
            'num_planes': record.num_planes,
            'prior_rate': record.prior_rate,
            'marginal_rate': record.marginal_rate,

            # Input data
            'delta_vec': delta_vec.cpu(),
            'bins': bins.cpu(),
            'prep_metadata': prep_metadata,

            # Prior distributions
            'prior': prior.cpu() if isinstance(prior, torch.Tensor) else prior,
            'm_prior': m_prior.cpu() if isinstance(m_prior, torch.Tensor) else m_prior,

            # Quantizer state
            'quantizer_state_dict': quantizer.coding_model.state_dict(),
            'quantizer_config': {
                'num_planes': quantizer.num_planes,
                'bins_per_plane': quantizer.bins_per_plane,
                'no_si': quantizer.no_si,
                'vec_slices': quantizer.vec_slices,
                'outlier_threshold': quantizer.outlier_threshold,
                'wmspe_denom': quantizer.wmspe_denom,
                'si_vec_size': quantizer.si_vec_size,
            },
            'side_info_list_used': [si.cpu() if isinstance(si, torch.Tensor) else si
                                    for si in quantizer.side_info_list_used] if isinstance(quantizer.side_info_list_used,
                                                                                           list) else quantizer.side_info_list_used,
            'cached_priors_dict_keys': list(quantizer.cached_priors_dict.keys()),

            # History state
            'client_past_reconst': [[r.cpu() for r in client_list] for client_list in client_past_reconst],
            'srvr_past_reconst': [[r.cpu() for r in srvr_list] for srvr_list in srvr_past_reconst],

            # Additional debug info
            'hash_of_delta': PriorCalculator.get_hash(delta_vec),
            'hash_in_cache': PriorCalculator.get_hash(delta_vec) in quantizer.cached_priors_dict,
        }

        torch.save(debug_data, dump_path)
        print(f"   Saved debug data with {len(debug_data)} fields")

