use std::collections::VecDeque;

use glam::Vec4;
use image::{imageops::FilterType, DynamicImage, GenericImage};

#[derive(Clone, Copy)]
pub struct PackingRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PackingRect {
    pub fn to_uvst(&self, atlas_width: u32, atlas_height: u32) -> Vec4 {
        Vec4::new(
            self.x as f32 / atlas_width as f32,
            self.y as f32 / atlas_width as f32,
            self.width as f32 / atlas_width as f32,
            self.height as f32 / atlas_height as f32,
        )
    }
}

pub fn pack_textures(
    textures: &[DynamicImage],
    atlas_width: u32,
    atlas_height: u32,
) -> (DynamicImage, Vec<Vec4>) {
    let root = PackingRect {
        x: 0,
        y: 0,
        width: atlas_width,
        height: atlas_height,
    };
    let mut queue = VecDeque::from([root]);

    while queue.len() <= textures.len() {
        let node = queue.pop_front().expect("Texture packing queue was empty.");
        let half_width = node.width / 2;
        let half_height = node.height / 2;
        queue.extend([
            PackingRect {
                x: node.x,
                y: node.y,
                width: half_width,
                height: half_height,
            },
            PackingRect {
                x: node.x + half_width,
                y: node.y,
                width: half_width,
                height: half_height,
            },
            PackingRect {
                x: node.x,
                y: node.y + half_height,
                width: half_width,
                height: half_height,
            },
            PackingRect {
                x: node.x + half_width,
                y: node.y + half_height,
                width: half_width,
                height: half_height,
            },
        ]);
    }

    let mut leafs = queue.into_iter().collect::<Vec<_>>();
    leafs.sort_by(|a, b| b.width.cmp(&a.width));
    leafs.truncate(textures.len());

    let mut atlas = DynamicImage::new_rgba8(atlas_width, atlas_height);
    for (i, leaf) in leafs.iter().enumerate() {
        let tex = &textures[i];

        let resized_tex = tex.resize_exact(leaf.width, leaf.height, FilterType::Lanczos3);

        atlas
            .copy_from(&resized_tex.flipv(), leaf.x, leaf.y)
            .unwrap();
    }

    let sts = leafs
        .iter()
        .map(|x| x.to_uvst(atlas_width, atlas_height))
        .collect::<Vec<_>>();
    (atlas, sts)
}
