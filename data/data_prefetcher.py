import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_dictAnnos, self.enlarge_boxes, self.next_ttc = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_dictAnnos = None
            self.next_ttc = None
            self.enlarge_boxes = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        dictAnnos = self.next_dictAnnos
        enlarge_boxes = self.enlarge_boxes
        ttc = self.next_ttc
        if input is not None:
            self.record_stream(input)

        self.preload()

        return input, dictAnnos, enlarge_boxes,ttc

    def _input_cuda_for_image(self):
        if self.next_input is not  None:
            self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())