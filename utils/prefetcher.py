import torch


class DataPrefetcher(object):
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.source = None

        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.source = next(self.loader)
        except StopIteration:
            self.source = None
        else:
            with torch.cuda.stream(self.stream):
                self.source = [item.to(self.device, non_blocking=True)
                               for item in self.source if type(item) == torch.Tensor]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        source = self.source
        if source is not None:
            for s in source:
                s.record_stream(torch.cuda.current_stream())
        self.preload()
        return source
