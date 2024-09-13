#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ double d_params[4];

struct RGBA
{
    unsigned char r, g, b, a;
};

__device__ unsigned int mandelbrot_iteration(double cx, double cy, unsigned int max_iterations)
{
    double zx = cx, zy = cy;
    double zx2 = zx * zx, zy2 = zy * zy;
    unsigned int i;

    double zx0 = 0, zy0 = 0;
    unsigned int period = 0;

    for (i = 0; i < max_iterations; ++i)
    {
        if (zx2 + zy2 > 4.0)
            break;

        if (zx == zx0 && zy == zy0)
        {
            i = max_iterations;
            break;
        }

        if (period > 20)
        {
            zx0 = zx;
            zy0 = zy;
            period = 0;
        }

        zy = __fma_rn(2.0, zx * zy, cy); // Use fused multiply-add
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        ++period;
    }
    return i;
}

__device__ RGBA get_color(unsigned int iteration, unsigned int max_iterations, unsigned char *s_palette)
{
    if (iteration == max_iterations)
    {
        return {0, 0, 0, 255}; // Black for points in the set
    }
    else
    {
        unsigned int index = (iteration % 256) * 4;
        return {s_palette[index], s_palette[index + 1], s_palette[index + 2], 255};
    }
}

__global__ void mandelbrot_kernel(unsigned int width, unsigned int height, unsigned int max_iterations,
                                  RGBA *__restrict__ output, const unsigned char *palette)
{
    __shared__ unsigned char s_palette[256 * 4]; // Shared memory for color palette

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.y == 0 && threadIdx.x < 256)
    {
        ((uint4 *)s_palette)[threadIdx.x] = ((uint4 *)palette)[threadIdx.x];
    }
    __syncthreads();

    if (x < width && y < height)
    {
        double cx = __fma_rn((double)x / (width - 1), d_params[1] - d_params[0], d_params[0]);
        double cy = __fma_rn((double)y / (height - 1), d_params[3] - d_params[2], d_params[2]);
        unsigned int iteration = mandelbrot_iteration(cx, cy, max_iterations);
        RGBA color = get_color(iteration, max_iterations, s_palette);
        output[y * width + x] = color;
    }
}

extern "C" void calculate_mandelbrot(unsigned int width, unsigned int height, unsigned int max_iterations,
                                     double x_min, double x_max, double y_min, double y_max,
                                     const unsigned char *palette, RGBA *output)
{
    double h_params[4] = {x_min, x_max, y_min, y_max};
    cudaMemcpyToSymbol(d_params, h_params, sizeof(double) * 4);

    RGBA *d_output;
    unsigned char *d_palette;
    cudaMalloc(&d_output, width * height * sizeof(RGBA));
    cudaMalloc(&d_palette, 256 * 4 * sizeof(unsigned char));
    cudaMemcpy(d_palette, palette, 256 * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 num_blocks((width + block_size.x - 1) / block_size.x,
                    (height + block_size.y - 1) / block_size.y);

    mandelbrot_kernel<<<num_blocks, block_size>>>(width, height, max_iterations, d_output, d_palette);

    cudaMemcpy(output, d_output, width * height * sizeof(RGBA), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_palette);
}