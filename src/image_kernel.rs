use std::cmp::{max, min, PartialOrd};

use rayon::prelude::*;

pub fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

// https://stackoverflow.com/a/596241/3894179
#[inline]
pub fn get_brightness(r: u8, g: u8, b: u8) -> u32 {
    (r as u32 + r as u32 + g as u32 + g as u32 + g as u32 + b as u32) / 6
}

pub fn get_largest_alpha_avg(
    cc: image::Rgba<u8>,
    lightest_color: image::Rgba<u8>,
    a: image::Rgba<u8>,
    b: image::Rgba<u8>,
    c: image::Rgba<u8>,
    strength: u16,
) -> image::Rgba<u8> {
    let new_color = get_alpha_avg(cc, a, b, c, strength);
    if new_color[3] > lightest_color[3] {
        new_color
    } else {
        lightest_color
    }
}

pub fn get_alpha_avg(
    cc: image::Rgba<u8>,
    a: image::Rgba<u8>,
    b: image::Rgba<u8>,
    c: image::Rgba<u8>,
    strength: u16,
) -> image::Rgba<u8> {
    let new_color_r = ((cc[0] as u32 * (0xFF - strength) as u32
        + ((a[0] as u32 + b[0] as u32 + c[0] as u32) / 3) * strength as u32)
        / 0xFF) as u8;
    let new_color_g = ((cc[1] as u32 * (0xFF - strength) as u32
        + ((a[1] as u32 + b[1] as u32 + c[1] as u32) / 3) * strength as u32)
        / 0xFF) as u8;
    let new_color_b = ((cc[2] as u32 * (0xFF - strength) as u32
        + ((a[2] as u32 + b[2] as u32 + c[2] as u32) / 3) * strength as u32)
        / 0xFF) as u8;
    let new_color_a = ((cc[3] as u32 * (0xFF - strength) as u32
        + ((a[3] as u32 + b[3] as u32 + c[3] as u32) / 3) * strength as u32)
        / 0xFF) as u8;

    image::Rgba::<u8>([new_color_r, new_color_g, new_color_b, new_color_a])
}

pub struct ImageKernel {
    pub image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
}

impl ImageKernel {
    pub fn from_image(image: image::DynamicImage) -> ImageKernel {
        ImageKernel {
            image: image.to_rgba8(),
        }
    }

    pub fn width(&self) -> u32 {
        self.image.width()
    }

    pub fn height(&self) -> u32 {
        self.image.height()
    }

    pub fn scale(&mut self, width: u32, height: u32) {
        let mut raster_image = raster::Image {
            width: self.image.width() as i32,
            height: self.image.height() as i32,
            bytes: std::mem::replace(&mut self.image, image::ImageBuffer::new(0, 0)).into_raw(),
        };
        let mode = raster::interpolate::InterpolationMode::Bicubic;
        raster::interpolate::resample(&mut raster_image, width as i32, height as i32, mode)
            .expect("Scale error");
        self.image = image::ImageBuffer::from_raw(width, height, raster_image.bytes)
            .expect("Load from raw raster image error");
    }

    pub fn compute_luminance(&mut self) {
        let row_bytes = (self.image.width() * 4) as usize;
        self.image.par_chunks_mut(row_bytes.max(4)).for_each(|row| {
            for px in row.chunks_exact_mut(4) {
                // get_brightness is a weighted average of u8 channels, so it
                // is already <= 255 -- no clamp needed here.
                px[3] = get_brightness(px[0], px[1], px[2]) as u8;
            }
        });
    }

    pub fn compute_gradient(&mut self) {
        let sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
        let sobely = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

        let width = self.image.width();
        let height = self.image.height();
        let src = &self.image;
        let row_bytes = (width * 4) as usize;
        let mut buf = vec![0u8; row_bytes * height as usize];

        buf.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as u32;
                // Clamp neighbor offsets at the border instead of skipping
                // it, matching push_color/push_gradient's edge handling --
                // otherwise these pixels stay at buf's zero-init (black,
                // fully transparent) and bleed into push_gradient's blend.
                let y_t = if y == 0 { 0 } else { y - 1 };
                let y_b = if y == height - 1 { y } else { y + 1 };
                for x in 0..width {
                    let x_l = if x == 0 { 0 } else { x - 1 };
                    let x_r = if x == width - 1 { x } else { x + 1 };

                    let dx = src.get_pixel(x_l, y_t)[3] as i32 * sobelx[0][0]
                        + src.get_pixel(x, y_t)[3] as i32 * sobelx[0][1]
                        + src.get_pixel(x_r, y_t)[3] as i32 * sobelx[0][2]
                        + src.get_pixel(x_l, y)[3] as i32 * sobelx[1][0]
                        + src.get_pixel(x, y)[3] as i32 * sobelx[1][1]
                        + src.get_pixel(x_r, y)[3] as i32 * sobelx[1][2]
                        + src.get_pixel(x_l, y_b)[3] as i32 * sobelx[2][0]
                        + src.get_pixel(x, y_b)[3] as i32 * sobelx[2][1]
                        + src.get_pixel(x_r, y_b)[3] as i32 * sobelx[2][2];

                    let dy = src.get_pixel(x_l, y_t)[3] as i32 * sobely[0][0]
                        + src.get_pixel(x, y_t)[3] as i32 * sobely[0][1]
                        + src.get_pixel(x_r, y_t)[3] as i32 * sobely[0][2]
                        + src.get_pixel(x_l, y)[3] as i32 * sobely[1][0]
                        + src.get_pixel(x, y)[3] as i32 * sobely[1][1]
                        + src.get_pixel(x_r, y)[3] as i32 * sobely[1][2]
                        + src.get_pixel(x_l, y_b)[3] as i32 * sobely[2][0]
                        + src.get_pixel(x, y_b)[3] as i32 * sobely[2][1]
                        + src.get_pixel(x_r, y_b)[3] as i32 * sobely[2][2];

                    let squared = (dx * dx + dy * dy) as u32;

                    let pixel = src.get_pixel(x, y);
                    let i = (x * 4) as usize;
                    // Only need sqrt when it could land <= 255; anything
                    // above 255*255 is going to be clamped to 0 anyway.
                    let alpha = if squared > 255 * 255 {
                        0
                    } else {
                        (0xFF - (squared as f64).sqrt() as u32) as u8
                    };
                    row[i..i + 4].copy_from_slice(&[pixel[0], pixel[1], pixel[2], alpha]);
                }
            });

        self.image =
            image::ImageBuffer::from_raw(width, height, buf).expect("Rebuild gradient image");
    }

    pub fn push_color(&mut self, strength: u16) {
        // Blend weight below is (0xFF - strength), so anything past 0xFF
        // underflows the subtraction. Matches the clamp the original Java
        // implementation applies before using this strength value.
        let strength = clamp(strength, 0, 0xFF);
        let width = self.image.width();
        let height = self.image.height();
        let src = &self.image;
        let row_bytes = (width * 4) as usize;
        let mut buf = vec![0u8; row_bytes * height as usize];

        buf.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as u32;
                for x in 0..width {
                /*
                 * Kernel defination:
                 * --------------
                 * [tl] [tc] [tr]
                 * [ml] [mc] [mc]
                 * [bl] [bc] [br]
                 * --------------
                 */
                let mut x_r: i32 = 1;
                let mut x_l: i32 = -1;
                let mut y_b: i32 = 1;
                let mut y_t: i32 = -1;

                // Independent checks (not else-if): on a 1px-wide/tall
                // image x == 0 and x == width - 1 are both true, and both
                // sides need to clamp to the same pixel.
                if x == 0 {
                    x_l = 0;
                }
                if x == width - 1 {
                    x_r = 0;
                }

                if y == 0 {
                    y_t = 0;
                }
                if y == height - 1 {
                    y_b = 0;
                }

                // Top column
                let tl = *src.get_pixel((x as i32 + x_l) as u32, (y as i32 + y_t) as u32);
                let tc = *src.get_pixel(x, (y as i32 + y_t) as u32);
                let tr = *src.get_pixel((x as i32 + x_r) as u32, (y as i32 + y_t) as u32);

                // Middle column
                let ml = *src.get_pixel((x as i32 + x_l) as u32, y);
                let mc = *src.get_pixel(x, y);
                let mr = *src.get_pixel((x as i32 + x_r) as u32, y);

                // Bottom column
                let bl = *src.get_pixel((x as i32 + x_l) as u32, (y as i32 + y_b) as u32);
                let bc = *src.get_pixel(x, (y as i32 + y_b) as u32);
                let br = *src.get_pixel((x as i32 + x_r) as u32, (y as i32 + y_b) as u32);

                let mut lightest_color = mc;

                // Kernel 0 and 4
                let mut max_dark = max(bl[3], max(bc[3], br[3]));
                let mut min_light = min(tl[3], min(tc[3], tr[3]));

                if min_light > mc[3] && min_light > max_dark {
                    lightest_color =
                        get_largest_alpha_avg(mc, lightest_color, tl, tc, tr, strength);
                } else {
                    max_dark = max(tl[3], max(tc[3], tr[3]));
                    min_light = min(br[3], min(bc[3], bl[3]));
                    if min_light > mc[3] && min_light > max_dark {
                        lightest_color =
                            get_largest_alpha_avg(mc, lightest_color, br, bc, bl, strength);
                    }
                }

                // Kernel 1 and 5
                max_dark = max(mc[3], max(ml[3], bc[3]));
                min_light = min(mr[3], min(tc[3], tr[3]));

                if min_light > max_dark {
                    lightest_color =
                        get_largest_alpha_avg(mc, lightest_color, mr, tc, tr, strength);
                } else {
                    max_dark = max(mc[3], max(mr[3], tc[3]));
                    min_light = min(bl[3], min(ml[3], bc[3]));
                    if min_light > max_dark {
                        lightest_color =
                            get_largest_alpha_avg(mc, lightest_color, bl, ml, bc, strength);
                    }
                }

                // Kernel 2 and 6
                max_dark = max(ml[3], max(tl[3], bl[3]));
                min_light = min(mr[3], min(tr[3], br[3]));

                if min_light > mc[3] && min_light > max_dark {
                    lightest_color =
                        get_largest_alpha_avg(mc, lightest_color, mr, br, tr, strength);
                } else {
                    max_dark = max(mr[3], max(tr[3], br[3]));
                    min_light = min(ml[3], min(tl[3], bl[3]));
                    if min_light > mc[3] && min_light > max_dark {
                        lightest_color =
                            get_largest_alpha_avg(mc, lightest_color, ml, tl, bl, strength);
                    }
                }

                // Kernel 3 and 7
                max_dark = max(mc[3], max(ml[3], tc[3]));
                min_light = min(mr[3], min(br[3], bc[3]));

                if min_light > max_dark {
                    lightest_color =
                        get_largest_alpha_avg(mc, lightest_color, mr, br, bc, strength);
                } else {
                    max_dark = max(mc[3], max(mr[3], bc[3]));
                    min_light = min(tc[3], min(ml[3], tl[3]));
                    if min_light > max_dark {
                        lightest_color =
                            get_largest_alpha_avg(mc, lightest_color, tc, ml, tl, strength);
                    }
                }

                    let i = (x * 4) as usize;
                    row[i..i + 4].copy_from_slice(&lightest_color.0);
                }
            });

        self.image =
            image::ImageBuffer::from_raw(width, height, buf).expect("Rebuild push_color image");
    }

    pub fn push_gradient(&mut self, strength: u16) {
        // Same underflow hazard as push_color: clamp before it reaches
        // the (0xFF - strength) blend weight below.
        let strength = clamp(strength, 0, 0xFF);
        let width = self.image.width();
        let height = self.image.height();
        let src = &self.image;
        let row_bytes = (width * 4) as usize;
        let mut buf = vec![0u8; row_bytes * height as usize];

        buf.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as u32;
                for x in 0..width {
                /*
                 * Kernel defination:
                 * --------------
                 * [tl] [tc] [tr]
                 * [ml] [mc] [mc]
                 * [bl] [bc] [br]
                 * --------------
                 */
                let mut x_r: i32 = 1;
                let mut x_l: i32 = -1;
                let mut y_b: i32 = 1;
                let mut y_t: i32 = -1;

                // Independent checks (not else-if): on a 1px-wide/tall
                // image x == 0 and x == width - 1 are both true, and both
                // sides need to clamp to the same pixel.
                if x == 0 {
                    x_l = 0;
                }
                if x == width - 1 {
                    x_r = 0;
                }

                if y == 0 {
                    y_t = 0;
                }
                if y == height - 1 {
                    y_b = 0;
                }

                // Top column
                let tl = *src.get_pixel((x as i32 + x_l) as u32, (y as i32 + y_t) as u32);
                let tc = *src.get_pixel(x, (y as i32 + y_t) as u32);
                let tr = *src.get_pixel((x as i32 + x_r) as u32, (y as i32 + y_t) as u32);

                // Middle column
                let ml = *src.get_pixel((x as i32 + x_l) as u32, y);
                let mc = *src.get_pixel(x, y);
                let mr = *src.get_pixel((x as i32 + x_r) as u32, y);

                // Bottom column
                let bl = *src.get_pixel((x as i32 + x_l) as u32, (y as i32 + y_b) as u32);
                let bc = *src.get_pixel(x, (y as i32 + y_b) as u32);
                let br = *src.get_pixel((x as i32 + x_r) as u32, (y as i32 + y_b) as u32);

                let mut lightest_color = mc;

                // Kernel 0 and 4
                let mut max_dark = max(bl[3], max(bc[3], br[3]));
                let mut min_light = min(tl[3], min(tc[3], tr[3]));

                if min_light > mc[3] && min_light > max_dark {
                    lightest_color = get_alpha_avg(mc, tl, tc, tr, strength);
                } else {
                    max_dark = max(tl[3], max(tc[3], tr[3]));
                    min_light = min(br[3], min(bc[3], bl[3]));
                    if min_light > mc[3] && min_light > max_dark {
                        lightest_color = get_alpha_avg(mc, br, bc, bl, strength);
                    }
                }

                // Kernel 1 and 5
                max_dark = max(mc[3], max(ml[3], bc[3]));
                min_light = min(mr[3], min(tc[3], tr[3]));

                if min_light > max_dark {
                    lightest_color = get_alpha_avg(mc, mr, tc, tr, strength);
                } else {
                    max_dark = max(mc[3], max(mr[3], tc[3]));
                    min_light = min(bl[3], min(ml[3], bc[3]));
                    if min_light > max_dark {
                        lightest_color = get_alpha_avg(mc, bl, ml, bc, strength);
                    }
                }

                // Kernel 2 and 6
                max_dark = max(ml[3], max(tl[3], bl[3]));
                min_light = min(mr[3], min(tr[3], br[3]));

                if min_light > mc[3] && min_light > max_dark {
                    lightest_color = get_alpha_avg(mc, mr, br, tr, strength);
                } else {
                    max_dark = max(mr[3], max(tr[3], br[3]));
                    min_light = min(ml[3], min(tl[3], bl[3]));
                    if min_light > mc[3] && min_light > max_dark {
                        lightest_color = get_alpha_avg(mc, ml, tl, bl, strength);
                    }
                }

                // Kernel 3 and 7
                max_dark = max(mc[3], max(ml[3], tc[3]));
                min_light = min(mr[3], min(br[3], bc[3]));

                if min_light > max_dark {
                    lightest_color = get_alpha_avg(mc, mr, br, bc, strength);
                } else {
                    max_dark = max(mc[3], max(mr[3], bc[3]));
                    min_light = min(tc[3], min(ml[3], tl[3]));
                    if min_light > max_dark {
                        lightest_color = get_alpha_avg(mc, tc, ml, tl, strength);
                    }
                }

                    lightest_color[3] = 255;
                    let i = (x * 4) as usize;
                    row[i..i + 4].copy_from_slice(&lightest_color.0);
                }
            });

        self.image = image::ImageBuffer::from_raw(width, height, buf)
            .expect("Rebuild push_gradient image");
    }

    pub fn save(&self, filename: &str) -> image::ImageResult<()> {
        self.image.save(filename)
    }
}
