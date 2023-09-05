// reimplementation for grid sample FORWARD
// scalar_t alias for tensor type
// #include <c10/macros/Macros.h>
#include <assert.h>
#include "grid_sample_helper.hpp"


namespace {
  using grid_sample::GridSamplerInterpolation;
  using grid_sample::GridSamplerPadding;
  using grid_sample::TensorDesc;
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
    if (align_corners) {
      // unnormalize coord from [-1, 1] to [0, size - 1]
      return ((coord + 1.f) / 2) * (size - 1);
    } else {
      // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
      return ((coord + 1.f) * size - 1) / 2;
    }
  }

  // Clips coordinates to between 0 and clip_limit - 1
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t clip_coordinates(scalar_t in, int clip_limit) {
    return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
  }
  
  // Reflects coordinates until they fall between low and high (inclusive).
  // The bounds are passed as twice their value so that half-integer values
  // can be represented as ints.
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
    if (twice_low == twice_high) {
      return static_cast<scalar_t>(0);
    }
    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = ::fabs(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    scalar_t extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));
    if (flips % 2 == 0) {
      return extra + min;
    } else {
      return span - extra + min;
    }
  }
    
  template<typename scalar_t>
  static __forceinline__ __device__
  scalar_t safe_downgrade_to_int_range(scalar_t x){
    // -100.0 does not have special meaning. This is just to make sure
    // it's not within_bounds_2d or within_bounds_3d, and does not cause
    // undefined behavior. See #35506.
    if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
      return static_cast<scalar_t>(-100.0);
    return x;
  }
  
  template<typename scalar_t>
  static __forceinline__ __device__
  scalar_t compute_coordinates(scalar_t coord, int size,
                                GridSamplerPadding padding_mode,
                                bool align_corners) {
    if (padding_mode == GridSamplerPadding::Border) { 
      // clip coordinates to image borders
      coord = clip_coordinates(coord, size);
    }
    // zero pading
    coord = safe_downgrade_to_int_range(coord);
    return coord;
  }
    
  // Computes the pixel source index value for a grid coordinate
  // only support zero-padding and 
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t grid_sampler_compute_source_index(
      scalar_t coord,
      int size,
      GridSamplerPadding padding_mode,
      bool align_corners) {
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    coord = compute_coordinates(coord, size, padding_mode, align_corners);
    return coord;
  }

    
  static __forceinline__ __device__
  bool within_bounds_2d(int h, int w, int H, int W) {
      return h >= 0 && h < H && w >= 0 && w < W;
  }
  static __forceinline__ __device__
  bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
      return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
  }

  template <typename scalar_t>
  // C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_kernel(
      const int nthreads, 
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grid,
      scalar_t* __restrict__ output,
      TensorDesc input_desc,
      TensorDesc grid_desc,
      TensorDesc output_desc,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode, 
      bool align_corners) {

    // load dimension for parametrts
    // input shape: N x C x H x W
    // grid  shape: N x H'x W'x 2 
    // out   shape: N x C x H'x W'
    int C = input_desc.shape[1];
    int inp_H = input_desc.shape[2];
    int inp_W = input_desc.shape[3];
    int out_H = grid_desc.shape[1];
    int out_W = grid_desc.shape[2];

    int inp_sN = input_desc.stride[0]; //  C x H xW
    int inp_sC = input_desc.stride[1]; // H
    int inp_sH = input_desc.stride[2]; // W
    int inp_sW = input_desc.stride[3]; // 1

    int grid_sN = grid_desc.stride[0];
    int grid_sH = grid_desc.stride[1];
    int grid_sW = grid_desc.stride[2];
    int grid_sCoor = grid_desc.stride[3];

    int out_sN = output_desc.stride[0];
    int out_sC = output_desc.stride[1];
    int out_sH = output_desc.stride[2];
    int out_sW = output_desc.stride[3];

 

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int n = index / (out_H * out_W);
      const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y coordinates from grid
      scalar_t ix = grid[grid_offset];
      scalar_t iy = grid[grid_offset + grid_sCoor];

      ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
      iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        // NE: top-right, NW: top-left, SE: bottom-right, SW: bottom-left;
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix) * (iy_se - iy);
        scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
        scalar_t se = (ix - ix_nw) * (iy - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input + n * inp_sN;
        auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
            *out_ptr_NCHW = static_cast<scalar_t>(0);
            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
            }
            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
            }
            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
            }
            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
            }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ix));
        int iy_nearest = static_cast<int>(::round(iy));

        // assign nearest neighbor pixel value to output pixel
        auto inp_ptr_NC = input + n * inp_sN;
        auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
            *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }


  template <typename scalar_t>
  // C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sample_repeat_2d_kernel(
      const int nthreads, 
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grid,
      scalar_t* __restrict__ output,
      TensorDesc input_desc,
      TensorDesc grid_desc,
      TensorDesc output_desc,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode, 
      bool align_corners) {
    // load dimension for parametrts
    // input shape: C x H x W
    // grid  shape: N x H'x W'x 2, N = num_scale
    // out   shape: N x C x H'x W'

    int C = input_desc.shape[0];
    int inp_H = input_desc.shape[1];
    int inp_W = input_desc.shape[2];
    
    int out_H = grid_desc.shape[1];
    int out_W = grid_desc.shape[2];
    int inp_sC = input_desc.stride[0];
    int inp_sH = input_desc.stride[1];
    int inp_sW = input_desc.stride[2];
    
    int grid_sN = grid_desc.stride[0];
    int grid_sH = grid_desc.stride[1];
    int grid_sW = grid_desc.stride[2];
    int grid_sCoor = grid_desc.stride[3];
    int out_sN = output_desc.stride[0];
    int out_sC = output_desc.stride[1];
    int out_sH = output_desc.stride[2];
    int out_sW = output_desc.stride[3];

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int n = index / (out_H * out_W);
      const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y coordinates from grid
      scalar_t ix = grid[grid_offset];
      scalar_t iy = grid[grid_offset + grid_sCoor];

      ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode,
                                          align_corners);
      iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode,
                                          align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix) * (iy_se - iy);
        scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
        scalar_t se = (ix - ix_nw) * (iy - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input; 
        auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;

        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ix));
        int iy_nearest = static_cast<int>(::round(iy));

        // assign nearest neighbor pixel value to output pixel
        auto inp_ptr_NC = input;
        auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
            *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }



  template <typename scalar_t>
  __global__ void grid_sample_scale_shift_kernel(
      const int nthreads, 
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ input_shift,
      const scalar_t* __restrict__ grid,
      const scalar_t* __restrict__ grid_shift, 
      const int shift_win,
      scalar_t* __restrict__ output,
      scalar_t* __restrict__ output_shift,
      TensorDesc input_desc,
      TensorDesc grid_desc,
      TensorDesc output_desc,
      TensorDesc output_shift_desc,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode, 
      bool align_corners) {
    // load dimension for parameters
    // input shape: C x H x W
    // input shift: C x H x W
    // grid  shape: N x H'x W'x 2, N = num_scale
    // grid  shift: N x H'x W'x 2,  meshgrid 
    // out   shape: N x C x H'x W'
    // out   shift: N x M * C x H'x W'
    // shift windows: range x-shift_win, x+shift_win
    int C = input_desc.shape[0];
    int inp_H = input_desc.shape[1];
    int inp_W = input_desc.shape[2];
    
    int out_H = grid_desc.shape[1];
    int out_W = grid_desc.shape[2];
    int inp_sC = input_desc.stride[0];
    int inp_sH = input_desc.stride[1];
    int inp_sW = input_desc.stride[2];
    
    int grid_sN = grid_desc.stride[0];
    int grid_sH = grid_desc.stride[1];
    int grid_sW = grid_desc.stride[2];
    int grid_sCoor = grid_desc.stride[3];
    int out_sN = output_desc.stride[0];
    int out_sC = output_desc.stride[1];
    int out_sH = output_desc.stride[2];
    int out_sW = output_desc.stride[3];

    int out_shift_sN = output_shift_desc.stride[0];
    int out_shift_sMC = output_shift_desc.stride[1];
    int out_shift_sH = output_shift_desc.stride[2];
    int out_shift_sW = output_shift_desc.stride[3];
  
    int out_MC = output_shift_desc.shape[1]; // Nx M*C xHxWW
    int shift_win_size = shift_win * 2 + 1; 
    
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      // index, do NHW LOOP
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int n = index / (out_H * out_W);
      
      const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
      const int grid_shift_offset =  n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y coordinates from grid
      scalar_t ix = grid[grid_offset];
      scalar_t iy = grid[grid_offset + grid_sCoor];

      ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
      iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
      // ix, iy has been scaled
      scalar_t ix_shift = grid_shift[grid_shift_offset];
      scalar_t iy_shift = grid_shift[grid_shift_offset + grid_sCoor];
      
      ix_shift = grid_sampler_compute_source_index(ix_shift, inp_W, padding_mode, align_corners);
      iy_shift = grid_sampler_compute_source_index(iy_shift, inp_H, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix) * (iy_se - iy);
        scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
        scalar_t se = (ix - ix_nw) * (iy - iy_nw);


        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input; 
        auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;  // N (C) H W
        auto out_shift_ptr_NCHW = output_shift + n  * out_shift_sN  + h * out_shift_sH + w * out_shift_sW;

        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
          // check bounds
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
          }          
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
          }
        }
        // get shift results
        for (int m = 0; m < out_MC / C; ++m){
          auto inp_shift_ptr = input_shift;
          scalar_t ix_shift_win = ix_shift + m % shift_win_size - shift_win;
          scalar_t iy_shift_win = iy_shift + (m / shift_win_size) % shift_win_size - shift_win;

          int ix_shift_nw = static_cast<int>(::floor(ix_shift_win));
          int iy_shift_nw = static_cast<int>(::floor(iy_shift_win));
          int ix_shift_ne = ix_shift_nw + 1;
          int iy_shift_ne = iy_shift_nw;
          int ix_shift_sw = ix_shift_nw;
          int iy_shift_sw = iy_shift_nw + 1;
          int ix_shift_se = ix_shift_nw + 1;
          int iy_shift_se = iy_shift_nw + 1;

          scalar_t nw_shift = (ix_shift_se - ix_shift) * (iy_shift_se - iy_shift);
          scalar_t ne_shift = (ix_shift - ix_shift_sw) * (iy_shift_sw - iy_shift);
          scalar_t sw_shift = (ix_shift_ne - ix_shift) * (iy_shift - iy_shift_ne);
          scalar_t se_shift = (ix_shift - ix_shift_nw) * (iy_shift - iy_shift_nw);

          for (int c = 0; c < C; ++c, inp_shift_ptr += inp_sC, out_shift_ptr_NCHW += out_sC){
            *out_shift_ptr_NCHW = static_cast<scalar_t>(0);
            if (within_bounds_2d(iy_shift_nw, ix_shift_nw, inp_H, inp_W)){
              *out_shift_ptr_NCHW += inp_shift_ptr[iy_shift_nw * inp_sH + ix_shift_nw * inp_sW] * nw_shift;
            }
            if (within_bounds_2d(iy_shift_ne, ix_shift_ne, inp_H, inp_W)){
              *out_shift_ptr_NCHW += inp_shift_ptr[iy_shift_ne * inp_sH + ix_shift_ne * inp_sW] * ne_shift;
            }
            if (within_bounds_2d(iy_shift_sw, ix_shift_sw, inp_H, inp_W)) {
              *out_shift_ptr_NCHW += inp_shift_ptr[iy_shift_sw * inp_sH + ix_shift_sw * inp_sW] * sw;
            }
            if (within_bounds_2d(iy_shift_se, ix_shift_se, inp_H, inp_W)) {
              *out_shift_ptr_NCHW += inp_shift_ptr[iy_shift_se * inp_sH + ix_shift_se * inp_sW] * se;
            }
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ix));
        int iy_nearest = static_cast<int>(::round(iy));

        // assign nearest neighbor pixel value to output pixel
        auto inp_ptr_NC = input;
        auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        auto out_shift_ptr_NCHW = output_shift + n  * out_shift_sN  + h * out_shift_sH + w * out_shift_sW;
        auto inp_shift_ptr_NC = input_shift;

        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC) {
          if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
            *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCHW = static_cast<scalar_t>(0);
          }
          out_ptr_NCHW += out_sC;
        }

        for (int m=0; m < out_MC / C; ++m){
          auto inp_shift_ptr_NC = input_shift;
          scalar_t ix_shift_win = ix_shift + m % shift_win_size - shift_win;
          scalar_t iy_shift_win = iy_shift + (m / shift_win_size) % shift_win_size - shift_win;
          ix_shift_win = compute_coordinates(ix_shift_win, inp_W, padding_mode, align_corners);
          iy_shift_win = compute_coordinates(iy_shift_win, inp_H, padding_mode, align_corners);
          int ix_shift_nearest = static_cast<int>(::round(ix_shift_win));
          int iy_shift_nearest = static_cast<int>(::round(iy_shift_win));
          for (int c=0; c < C; ++c, inp_shift_ptr_NC += inp_sC){
            if (within_bounds_2d(iy_shift_nearest, ix_shift_nearest, inp_H, inp_W)) {
              *out_shift_ptr_NCHW = inp_shift_ptr_NC[iy_shift_nearest * inp_sH + ix_shift_nearest * inp_sW];
            } else {
              *out_shift_ptr_NCHW = static_cast<scalar_t>(0);
            }
            out_shift_ptr_NCHW += out_shift_sMC;
          } 
        }
      }
    }
  }

    
  template <typename scalar_t>
  // C10_LAUNCH_BOUNDS_1(512)
  __global__ void grid_sampler_3d_kernel(
      const int nthreads, 
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grid,
      scalar_t* __restrict__ output,
      TensorDesc input_desc,
      TensorDesc grid_desc,
      TensorDesc output_desc,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode, 
      bool align_corners) {
    // input: N x C x [D x H x W] 5-d 
    int C = input_desc.shape[1];
    int inp_D = input_desc.shape[2];
    int inp_H = input_desc.shape[3];
    int inp_W = input_desc.shape[4];
    int out_D = grid_desc.shape[1];
    int out_H = grid_desc.shape[2];
    int out_W = grid_desc.shape[3];
    int inp_sN = input_desc.stride[0];
    int inp_sC = input_desc.stride[1];
    int inp_sD = input_desc.stride[2];
    int inp_sH = input_desc.stride[3];
    int inp_sW = input_desc.stride[4];
    int grid_sN = grid_desc.stride[0];
    int grid_sD = grid_desc.stride[1];
    int grid_sH = grid_desc.stride[2];
    int grid_sW = grid_desc.stride[3];
    int grid_sCoor = grid_desc.stride[4];
    int out_sN = output_desc.stride[0];
    int out_sC = output_desc.stride[1];
    int out_sD = output_desc.stride[2];
    int out_sH = output_desc.stride[3];
    int out_sW = output_desc.stride[4];


    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int d = (index / (out_H * out_W)) % out_D;
      const int n = index / (out_D * out_H * out_W);
      const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z coordinates from grid
      scalar_t ix = grid[grid_offset];
      scalar_t iy = grid[grid_offset + grid_sCoor];
      scalar_t iz = grid[grid_offset + 2 * grid_sCoor];

      ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
      iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
      iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        int ix_tnw = static_cast<int>(::floor(ix));
        int iy_tnw = static_cast<int>(::floor(iy));
        int iz_tnw = static_cast<int>(::floor(iz));

        int ix_tne = ix_tnw + 1;
        int iy_tne = iy_tnw;
        int iz_tne = iz_tnw;

        int ix_tsw = ix_tnw;
        int iy_tsw = iy_tnw + 1;
        int iz_tsw = iz_tnw;

        int ix_tse = ix_tnw + 1;
        int iy_tse = iy_tnw + 1;
        int iz_tse = iz_tnw;

        int ix_bnw = ix_tnw;
        int iy_bnw = iy_tnw;
        int iz_bnw = iz_tnw + 1;

        int ix_bne = ix_tnw + 1;
        int iy_bne = iy_tnw;
        int iz_bne = iz_tnw + 1;

        int ix_bsw = ix_tnw;
        int iy_bsw = iy_tnw + 1;
        int iz_bsw = iz_tnw + 1;

        int ix_bse = ix_tnw + 1;
        int iy_bse = iy_tnw + 1;
        int iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
        scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
        scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
        scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
        scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
        scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
        scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
        scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

        auto inp_ptr_NC = input + n * inp_sN;
        auto out_ptr_NCDHW = output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
          // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
          // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) *  bne
          // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) *  bse
          *out_ptr_NCDHW = static_cast<scalar_t>(0);
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
              *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ix));
        int iy_nearest = static_cast<int>(::round(iy));
        int iz_nearest = static_cast<int>(::round(iz));

        // assign nearest neighbor pixel value to output pixel
        auto inp_ptr_NC = input + n * inp_sN;
        auto out_ptr_NCDHW = output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCDHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }

}



void create_desc(const int *dims, int nb_dims, TensorDesc &desc) {
  memcpy(&desc.shape[0], dims, sizeof(int) * nb_dims);
  desc.stride[nb_dims - 1] = 1;
  for (int i = nb_dims - 2; i >= 0; --i) {
    desc.stride[i] = desc.stride[i + 1] * desc.shape[i + 1];
  };
}


std::vector<torch::Tensor>  grid_sample_scale_shift_2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor input_shift,
    torch::Tensor grid,
    torch::Tensor grid_shift,
    int shift_win,
    int interp, 
    int padding,
    bool align_corners){
  
  auto C = input.size(0);
  auto H_in = input.size(1);
  auto W_in = input.size(2);

  
  auto N = grid.size(0); // batch size
  auto H = grid.size(1);
  auto W = grid.size(2);
  // assert in python wrap
  // auto H_shift_in = input_shift.size(1);
  // auto W_shift_in = input_shift.size(2);
  // auto H_shift = grid_shift.size(1);
  // auto W_shift = grid_shift.size(2);
  // assert ((H_in==H_shift_in) && (W_in==W_shift_in)); // set same stride for two input 
  // assert ((H_shift==H) && (W_shift==W)); // set same stride for two grid 

  int shift_win_size = shift_win*2 + 1; 
  int  M = pow(shift_win_size, 2); // number of shift
      
  torch::Tensor output = torch::zeros({N, C, H, W}, input.options());
  torch::Tensor output_shift = torch::zeros({N, M * C, H, W}, input_shift.options());
  int count = static_cast<int>(N * H * W);
  int input_dims[] = {static_cast<int>(C), static_cast<int>(H_in), static_cast<int>(W_in)};  

  int nb_dims = 4; 
  int input_nb_dims = 3;

  TensorDesc input_desc;
  create_desc(input_dims, input_nb_dims, input_desc);

  int output_dims[] = {static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)}; 
  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  int output_shift_dims[] = {static_cast<int>(N), M * static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)}; 
  TensorDesc output_shift_desc;
  create_desc(output_shift_dims, nb_dims, output_shift_desc);

  TensorDesc grid_desc;
  int grid_dims[] = {static_cast<int>(N), static_cast<int>(H), static_cast<int>(W), static_cast<int>(grid.size(3))}; 
  create_desc(grid_dims, nb_dims, grid_desc);
  // for (int i=0; i<nb_dims; ++i) {
  //   std::cout <<  output_shift_desc.shape[i] << " " << output_shift_desc.stride[i] << std::endl;
  // }
  
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sample_scale_shift_kernel", ([&] {
      grid_sample_scale_shift_kernel<scalar_t>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          input.data<scalar_t>(),
          input_shift.data<scalar_t>(),
          grid.data<scalar_t>(),
          grid_shift.data<scalar_t>(),
          shift_win,
          output.data<scalar_t>(),
          output_shift.data<scalar_t>(),
          input_desc,
          grid_desc,
          output_desc,
          output_shift_desc,
          static_cast<GridSamplerInterpolation>(interp), 
          static_cast<GridSamplerPadding>(padding),
          align_corners);
    }));
    // CUDA_CHECK(cudaPeekAtLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
  };
  return {output, output_shift};
}


torch::Tensor grid_sample_repeat_2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    int interp, int padding,
    bool align_corners){

  auto C = input.size(0);
  auto H_in = input.size(1);
  auto W_in = input.size(2);

  auto N = grid.size(0); // batch size
  auto H = grid.size(1);
  auto W = grid.size(2);
      
  torch::Tensor output = torch::zeros({N, C, H, W}, input.options());
  int count = static_cast<int>(N * H * W);
  int input_dims[] = {static_cast<int>(C), static_cast<int>(H_in), static_cast<int>(W_in)};  
  // TODO need safe convert

  int nb_dims = 4; 
  int input_nb_dims = 3;

  TensorDesc input_desc;
  create_desc(input_dims, input_nb_dims, input_desc);

  int output_dims[] = {static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)}; 
  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  TensorDesc grid_desc;
  int grid_dims[] = {static_cast<int>(N), static_cast<int>(H), static_cast<int>(W), static_cast<int>(grid.size(3))}; 
  create_desc(grid_dims, nb_dims, grid_desc);

 if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sample2d_cuda_forward", ([&] {
      grid_sample_repeat_2d_kernel<scalar_t>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          input.data<scalar_t>(),
          grid.data<scalar_t>(),
          output.data<scalar_t>(),
          input_desc,
          grid_desc,
          output_desc,
          static_cast<GridSamplerInterpolation>(interp), 
          static_cast<GridSamplerPadding>(padding),
          align_corners);
    }));
  };
  return output;
}

torch::Tensor grid_sample2d_cuda_forward(
    torch::Tensor input, 
    torch::Tensor grid,
    int interp, int padding,
    bool align_corners) {

  auto N = input.size(0); // batch size
  auto C = input.size(1);
  auto H_in = input.size(2);
  auto W_in = input.size(3);
  
  auto H = grid.size(1);
  auto W = grid.size(2);
  
  
  torch::Tensor output = torch::zeros({N, C, H, W}, input.options());
  int count = static_cast<int>(N * H * W);

  int input_dims[] = {static_cast<int>(N),static_cast<int>(C), static_cast<int>(H_in), static_cast<int>(W_in)};  
  int nb_dims = 4; // number of dimension
  TensorDesc input_desc;
  create_desc(input_dims, nb_dims, input_desc);

  int output_dims[] = {static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)}; 
  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  TensorDesc grid_desc;
  int grid_dims[] = {static_cast<int>(N), static_cast<int>(H), static_cast<int>(W), static_cast<int>(grid.size(3))}; 
  create_desc(grid_dims, nb_dims, grid_desc);
  

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sample2d_cuda_forward", ([&] {
      grid_sampler_2d_kernel<scalar_t>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          input.data<scalar_t>(),
          grid.data<scalar_t>(),
          output.data<scalar_t>(),
          input_desc,
          grid_desc,
          output_desc,
          static_cast<GridSamplerInterpolation>(interp), 
          static_cast<GridSamplerPadding>(padding),
          align_corners);
    }));
  };
  return output;
}

torch::Tensor grid_sample3d_cuda_forward(
  torch::Tensor input, 
  torch::Tensor grid,
  int interp, 
  int padding,
  bool align_corners) {

  auto N = input.size(0); // batch size
  auto C = input.size(1);
  auto H_in = input.size(3);
  auto W_in = input.size(4);
  
  auto D = grid.size(1); 
  auto H = grid.size(2);
  auto W = grid.size(3);
  

  torch::Tensor output = torch::zeros({N, C, D, H, W}, input.options());
  int count = static_cast<int>(N * D * H * W);

  int input_dims[] = {N, C, D, H_in, W_in}; 
  int nb_dims = 5; // number of dimension
  TensorDesc input_desc;
  create_desc(input_dims, nb_dims, input_desc);

  int output_dims[] = {N, C, D, H, W}; 
  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  TensorDesc grid_desc;
  int grid_dims[] = {N, D, H, W, grid.size(4)};
   
  create_desc(grid_dims, nb_dims, grid_desc);

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sample3d_cuda_forward", ([&] {
      grid_sampler_3d_kernel<scalar_t>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          input.data<scalar_t>(),
          grid.data<scalar_t>(),
          output.data<scalar_t>(),
          input_desc,
          grid_desc,
          output_desc,
          static_cast<GridSamplerInterpolation>(interp), 
          static_cast<GridSamplerPadding>(padding),
          align_corners);
    }));
  };
  return output;
}

    
