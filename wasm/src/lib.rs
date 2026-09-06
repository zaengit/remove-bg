use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn threshold_mask(mask: &[u8], threshold: u8) -> Vec<u8> {
    mask.iter().map(|&v| if v >= threshold { 255 } else { 0 }).collect()
}

#[wasm_bindgen]
pub fn union_masks(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x,&y)| x.max(y)).collect()
}

#[wasm_bindgen]
pub fn subtract_masks(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x,&y)| if y > 127 { 0 } else { x }).collect()
}

#[wasm_bindgen]
pub fn invert_mask(mask: &[u8]) -> Vec<u8> { mask.iter().map(|&v| 255-v).collect() }

fn morph(mask: &[u8], width: u32, height: u32, radius: u32, dilate: bool) -> Vec<u8> {
    let (w,h)=(width as i32,height as i32); let r=radius as i32; let mut out=vec![0;mask.len()];
    for y in 0..h { for x in 0..w { let mut value=if dilate {0u8}else{255u8}; 'scan: for yy in (y-r).max(0)..=(y+r).min(h-1) { for xx in (x-r).max(0)..=(x+r).min(w-1) { let v=mask[(yy*w+xx) as usize]; if dilate { value=value.max(v); if value==255 {break 'scan;} } else { value=value.min(v); if value==0 {break 'scan;} } } } out[(y*w+x) as usize]=value; } }
    out
}

#[wasm_bindgen]
pub fn dilate(mask:&[u8],width:u32,height:u32,radius:u32)->Vec<u8>{morph(mask,width,height,radius,true)}
#[wasm_bindgen]
pub fn erode(mask:&[u8],width:u32,height:u32,radius:u32)->Vec<u8>{morph(mask,width,height,radius,false)}

#[wasm_bindgen]
pub fn feather_mask(mask:&[u8],width:u32,height:u32,radius:u32)->Vec<u8>{
    if radius==0{return mask.to_vec()} let (w,h)=(width as i32,height as i32);let r=radius as i32;let mut out=vec![0;mask.len()];
    for y in 0..h{for x in 0..w{let mut sum=0u32;let mut n=0u32;for yy in (y-r).max(0)..=(y+r).min(h-1){for xx in (x-r).max(0)..=(x+r).min(w-1){sum+=mask[(yy*w+xx)as usize]as u32;n+=1}}out[(y*w+x)as usize]=(sum/n)as u8}}out
}

#[wasm_bindgen]
pub fn resize_mask(mask:&[u8],src_w:u32,src_h:u32,dst_w:u32,dst_h:u32)->Vec<u8>{
    let mut out=vec![0;(dst_w*dst_h)as usize];for y in 0..dst_h{for x in 0..dst_w{let sx=(x as u64*src_w as u64/dst_w as u64)as u32;let sy=(y as u64*src_h as u64/dst_h as u64)as u32;out[(y*dst_w+x)as usize]=mask[(sy*src_w+sx)as usize]}}out
}

#[wasm_bindgen]
pub fn apply_remove_mask(pixels:&mut [u8],mask:&[u8],threshold:u8){assert_eq!(pixels.len()/4,mask.len());for(i,&m)in mask.iter().enumerate(){if m>=threshold{pixels[i*4+3]=0}}}

#[wasm_bindgen]
pub fn restore_pixels(pixels:&mut[u8],original:&[u8],mask:&[u8],threshold:u8){assert_eq!(pixels.len(),original.len());assert_eq!(pixels.len()/4,mask.len());for(i,&m)in mask.iter().enumerate(){if m>=threshold{pixels[i*4..i*4+4].copy_from_slice(&original[i*4..i*4+4])}}}

#[wasm_bindgen]
pub fn mask_bounding_box(mask:&[u8],width:u32,height:u32,threshold:u8)->Vec<i32>{let(mut minx,mut miny,mut maxx,mut maxy)=(width as i32,height as i32,-1,-1);for y in 0..height{for x in 0..width{if mask[(y*width+x)as usize]>=threshold{minx=minx.min(x as i32);miny=miny.min(y as i32);maxx=maxx.max(x as i32);maxy=maxy.max(y as i32)}}}if maxx<0{vec![-1,-1,-1,-1]}else{vec![minx,miny,maxx,maxy]}}
