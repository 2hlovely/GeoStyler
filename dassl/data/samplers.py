from torch.utils.data import RandomSampler, SequentialSampler


def build_sampler(
    sampler_type,
    cfg=None,
    data_source=None,
    batch_size=32,
    n_domain=0,
    n_ins=16
):
    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)
    if sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)
    raise ValueError("Unsupported sampler: {}".format(sampler_type))
