from components.broadcast_components.broadcasting_process.WorkersideTraining import WorkersideTrainingProtocol
from components.broadcast_components.WZ_models.BasicQuantizer import \
    RoundBasicQuantizer, RoundDSCQuantizer, SignBasicQuantizer, SignDSCQuantizer


class _ConventionalProtocols(WorkersideTrainingProtocol):
    def __init__(self, basic_quant_class, worker_count, base_wz_quantizer, *args, **kwargs):
        print('NOTE: this protocol ignores the wz_base_quantizer given and turns off the global quantization')
        assert basic_quant_class in [RoundBasicQuantizer, RoundDSCQuantizer, SignBasicQuantizer, SignDSCQuantizer]
        qz_f = lambda: basic_quant_class(base_wz_quantizer.wz_pl_model, to_clone_wz_qz=base_wz_quantizer)
        kwargs['no_global_quantization'] = True
        super(_ConventionalProtocols, self).__init__(worker_count, qz_f(), *args, **kwargs)

        self.wz_quantizer_list = [qz_f() for _ in range(worker_count)]


class RoundBasicProtocol(_ConventionalProtocols):
    def __init__(self, *args, **kwargs):
        super(RoundBasicProtocol, self).__init__(RoundBasicQuantizer, *args, **kwargs)


class RoundDSCProtocol(_ConventionalProtocols):
    def __init__(self, *args, **kwargs):
        super(RoundDSCProtocol, self).__init__(RoundDSCQuantizer, *args, **kwargs)


class SignBasicProtocol(_ConventionalProtocols):
    def __init__(self, *args, **kwargs):
        super(SignBasicProtocol, self).__init__(SignBasicQuantizer, *args, **kwargs)


class SignDSCProtocol(_ConventionalProtocols):
    def __init__(self, *args, **kwargs):
        super(SignDSCProtocol, self).__init__(SignDSCQuantizer, *args, **kwargs)


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        RoundDSCProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)

    bp_f = lambda worker_count, base_quantizer: (
        SignDSCProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)

    bp_f = lambda worker_count, base_quantizer: (
        RoundBasicProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)

    bp_f = lambda worker_count, base_quantizer: (
        SignBasicProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)
