import torch 
import custom_gridsample as gscuda
import time

class ScaleShift(object):
    # Defination in CPP:
    _INTER_MAP = {
        'nearst': 0,
        'bilinear': 1,
    }
    _PADDING_MAP = {
        'zeros': 0,
        'border': 1,
    }

    def __init__(self, interp='nearst', padding='zeros', verbose=False):
        self._interp = interp
        self._padding = padding
        self.interp_mode = self._INTER_MAP[interp]
        self.padding_mode = self._PADDING_MAP[padding]
        self.verbose = verbose
        self.device = "cuda:0"
        assert torch.cuda.is_aviabile()

    def __call__(self, scale_inputs, shift_inputs, scale_grid, shift_grid, win_size, ):
        assert scale_inputs.shape == shift_inputs.shape
        assert scale_grid.shape == shift_grid.shape
        if not isinstance(scale_inputs, torch.Tensor): scale_inputs = torch.Tensor(scale_inputs)
        if not isinstance(shift_inputs, torch.Tensor): shift_inputs = torch.Tensor(shift_inputs)

        scale_out, shift_out = gscuda.forward_2d_scale_shift(scale_inputs, shift_inputs, scale_grid, \
            shift_grid, win_size, self.interp_mode, self.padding_mode, True) # align corners
        _num_scale = scale_out.shape[0]
        _h, _w = scale_out.shape[-2], scale_out.shape[-1]
        shift_out = shift_out.reshape(_num_scale, -1, 3, _h, _w).to(device=self.device) # NxMx3xHxW
        scale_out = scale_out.to(device=self.device)
        
        del scale_grid
        del scale_inputs
        del shift_grid
        del shift_inputs
        
        return scale_out, shift_out


    def __repr__(self):
        _msg = 'Custom Grid Sample Forward <CUDA> ' + \
            f'Interpolation = {self._interp}' + \
            f'Padding = {self._padding}' 
        return _msg
        