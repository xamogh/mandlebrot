use cuda_runtime_sys::*;
use gif::{Encoder, Frame, Repeat};
use parking_lot::Mutex;
use std::error::Error;
use std::fs::File;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr::null_mut;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const WIDTH: usize = 1200;
const HEIGHT: usize = 800;
const MAX_ITERATIONS: u32 = 12000;
const TOTAL_FRAMES: usize = 500;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RGBA {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[link(name = "mandelbrot_cuda", kind = "static")]
extern "C" {
    fn calculate_mandelbrot(
        width: u32,
        height: u32,
        max_iterations: u32,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        palette: *const u8,
        output: *mut RGBA,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut device_count = 0;
        cudaGetDeviceCount(&mut device_count);
        if device_count == 0 {
            return Err("No CUDA devices found".into());
        }
        cudaSetDevice(0);
    }

    let color_palette = Arc::new(create_color_palette(256));
    let output_size = WIDTH * HEIGHT * size_of::<RGBA>();
    let output_buffer = Arc::new(Mutex::new(Vec::with_capacity(TOTAL_FRAMES)));

    let cuda_thread = thread::spawn({
        let color_palette = Arc::clone(&color_palette);
        let output_buffer = Arc::clone(&output_buffer);
        move || {
            let mut d_output: *mut RGBA = null_mut();
            unsafe {
                cudaMalloc(
                    &mut d_output as *mut *mut RGBA as *mut *mut c_void,
                    output_size,
                );
            }

            let cuda_start_time = Instant::now();

            for frame_index in 0..TOTAL_FRAMES {
                let zoom = 8.0 * 10f64.powf(-16.0 * frame_index as f64 / TOTAL_FRAMES as f64);
                let center_x = -0.743643887037151;
                let center_y = 0.131825904205330;
                let x_min = center_x - 1.5 * zoom;
                let x_max = center_x + 1.5 * zoom;
                let y_min = center_y - 1.5 * (HEIGHT as f64 / WIDTH as f64) * zoom;
                let y_max = center_y + 1.5 * (HEIGHT as f64 / WIDTH as f64) * zoom;

                let mut h_output = vec![RGBA::default(); WIDTH * HEIGHT];

                unsafe {
                    calculate_mandelbrot(
                        WIDTH as u32,
                        HEIGHT as u32,
                        MAX_ITERATIONS,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        color_palette.as_ptr() as *const u8,
                        d_output,
                    );

                    cudaMemcpy(
                        h_output.as_mut_ptr() as *mut c_void,
                        d_output as *const c_void,
                        output_size,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    );
                }

                output_buffer.lock().push(h_output);
                println!(
                    "Generated data for frame {}/{}",
                    frame_index + 1,
                    TOTAL_FRAMES
                );
            }

            unsafe {
                cudaFree(d_output as *mut c_void);
            }

            cuda_start_time.elapsed()
        }
    });

    let cuda_time = cuda_thread.join().unwrap();
    println!("Total CUDA computation time: {:?}", cuda_time);

    let encoding_start_time = Instant::now();

    let mut output_file = File::create("mandelbrot_zoom.gif")?;
    let mut encoder = Encoder::new(&mut output_file, WIDTH as u16, HEIGHT as u16, &[])?;
    encoder.set_repeat(Repeat::Infinite)?;

    let buffer = output_buffer.lock();
    for (frame_index, frame_data) in buffer.iter().enumerate() {
        let mut gif_frame = Frame::from_rgba(
            WIDTH as u16,
            HEIGHT as u16,
            &mut frame_data
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect::<Vec<u8>>(),
        );
        gif_frame.delay = 5; // 20 ms delay
        encoder.write_frame(&gif_frame)?;
        println!("Encoded frame {}/{}", frame_index + 1, TOTAL_FRAMES);
    }

    let encoding_time = encoding_start_time.elapsed();
    println!("GIF saved as 'mandelbrot_zoom.gif'");
    println!("Total GIF encoding time: {:?}", encoding_time);
    println!("Total execution time: {:?}", cuda_time + encoding_time);

    Ok(())
}

fn create_color_palette(size: usize) -> Vec<RGBA> {
    (0..size)
        .map(|i| {
            let t = i as f64 / size as f64;
            let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u8;
            let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u8;
            let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u8;
            RGBA { r, g, b, a: 255 }
        })
        .collect()
}
