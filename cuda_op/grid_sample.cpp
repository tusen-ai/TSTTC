#include <torch/extension.h>
#include <vector>

// INTERPOLATION_MODES = {
//     "Nearest": 0,
//     "Bilinear": 1,
// }

// PADDING_MODES = {
//     "Zeros": 0,
//     "Border": 1,
// } 


// cuda declarations

torch::Tensor grid_sample2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    int interp,
    int padding,
    bool align_corners);

torch::Tensor grid_sample3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    int interp,
    int padding,
    bool align_corners);


torch::Tensor grid_sample_repeat_2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    int interp,
    int padding,
    bool align_corners);


std::vector<torch::Tensor> grid_sample_scale_shift_2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor input_shift,
    torch::Tensor grid,
    torch::Tensor grid_shift,
    int shift_win,
    int interp,
    int padding,
    bool align_corners);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor grid_sample2d_forward(
    torch::Tensor input,
    torch::Tensor grid,
    int interp,
    int padding,
    bool align_corners) {
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  return grid_sample2d_cuda_forward(input, grid, interp , padding, align_corners);
}

torch::Tensor grid_sample_repeat_2d_forward(
    torch::Tensor input,
    torch::Tensor grid,
    int interp,
    int padding,
    bool align_corners) {
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  return grid_sample_repeat_2d_cuda_forward(input, grid, interp , padding, align_corners);
}

torch::Tensor grid_sample3d_forward(
    torch::Tensor input,
    torch::Tensor grid,
    int interp,
    int padding,
    bool align_corners) {
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  return grid_sample3d_cuda_forward(input, grid, interp, padding, align_corners);
}

std::vector<torch::Tensor> grid_sample_scale_shift_2d_forward(
    torch::Tensor input,
    torch::Tensor input_shift,
    torch::Tensor grid,
    torch::Tensor grid_shift,
    int shift_win,
    int interp,
    int padding,
    bool align_corners) {
  CHECK_INPUT(input);
  CHECK_INPUT(input_shift);
  CHECK_INPUT(grid);
  return grid_sample_scale_shift_2d_cuda_forward(input, input_shift, grid, grid_shift, shift_win, interp, padding, align_corners);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_2d", &grid_sample2d_forward, "grid sample 2d forward (CUDA)");
  m.def("forward_2d_repeat", &grid_sample_repeat_2d_forward, "grid sample repeat 2d forward (CUDA)");
  m.def("forward_2d_scale_shift", &grid_sample_scale_shift_2d_forward, "grid sample scale shift 2d forward (CUDA)");
  m.def("forward_3d", &grid_sample3d_forward, "grid sample 3d forward (CUDA)");
}