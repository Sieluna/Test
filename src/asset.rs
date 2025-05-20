use glam::{Mat4, UVec4, Vec2, Vec3, Vec4};
use gltf::{Material, Node};
use image::DynamicImage;
use shared_structs::{LightPickEntry, MaterialData, PerVertexData};

use crate::{
    bvh::{BVHBuilder, GpuBVH, BVH},
    light_pick,
    trace::FW,
};

use super::framework::{
    primitives::{
        pixels::{Rgba32Float, Rgba8UintNorm},
        PixelInfo,
    },
    BufOps, GpuBuffer, GpuConstImage, ImgOps,
};

pub struct World {
    pub bvh: BVH,
    pub per_vertex_buffer: Vec<PerVertexData>,
    pub index_buffer: Vec<UVec4>,
    pub atlas: DynamicImage,
    pub material_data_buffer: Vec<MaterialData>,
    pub light_pick_buffer: Vec<LightPickEntry>,
}

pub struct GpuWorld<'fw> {
    pub bvh: GpuBVH<'fw>,
    pub per_vertex_buffer: GpuBuffer<'fw, PerVertexData>,
    pub index_buffer: GpuBuffer<'fw, UVec4>,
    pub atlas: GpuConstImage<'fw, Rgba8UintNorm>,
    pub material_data_buffer: GpuBuffer<'fw, MaterialData>,
    pub light_pick_buffer: GpuBuffer<'fw, LightPickEntry>,
}

fn load_image_from_gltf(
    texture: &gltf::Texture,
    buffers: &[gltf::buffer::Data],
) -> Option<DynamicImage> {
    let source = texture.source();
    if let gltf::image::Source::View { view, mime_type: _ } = source.source() {
        let buffer = &buffers[view.buffer().index()];
        let data = &buffer.0[view.offset()..view.offset() + view.length()];
        image::load_from_memory(data).ok()
    } else if let gltf::image::Source::Uri { uri, mime_type: _ } = source.source() {
        if uri.starts_with("data:") {
            // Handle base64 encoded image data
            if let Some(data) = uri.find("base64,") {
                let data = &uri[data + 7..]; // Skip "base64,"
                let decoded = base64::decode(data).ok()?;
                image::load_from_memory(&decoded).ok()
            } else {
                None
            }
        } else {
            // Try to load from relative path
            let path = std::path::Path::new(uri);
            image::open(path).ok()
        }
    } else {
        None
    }
}

fn load_texture_from_material(
    material: &Material,
    textures: &[gltf::Texture],
    buffers: &[gltf::buffer::Data],
    texture_type: gltf::material::AlphaMode,
) -> Option<DynamicImage> {
    match texture_type {
        gltf::material::AlphaMode::Opaque => {
            if let Some(info) = material.pbr_metallic_roughness().base_color_texture() {
                let texture = &textures[info.texture().index()];
                load_image_from_gltf(texture, buffers)
            } else {
                None
            }
        }
        gltf::material::AlphaMode::Mask => {
            if let Some(info) = material.normal_texture() {
                let texture = &textures[info.texture().index()];
                load_image_from_gltf(texture, buffers)
            } else {
                None
            }
        }
        gltf::material::AlphaMode::Blend => {
            if let Some(info) = material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
            {
                let texture = &textures[info.texture().index()];
                load_image_from_gltf(texture, buffers)
            } else {
                None
            }
        }
    }
}

impl World {
    pub fn from_path(path: &str) -> Option<Self> {
        let buffers = std::fs::read(path).ok()?;
        Self::from_slice(&buffers)
    }

    pub fn from_slice(slice: impl AsRef<[u8]>) -> Option<Self> {
        let (gltf, buffers, _) = gltf::import_slice(slice.as_ref()).ok()?;

        // Get scene and nodes
        let scene = gltf.default_scene().or_else(|| gltf.scenes().next())?;

        // Collect mesh data
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();
        let mut tangents = Vec::new();
        let mut uvs = Vec::new();

        fn process_node(
            node: &Node,
            parent_transform: Mat4,
            vertices: &mut Vec<Vec4>,
            indices: &mut Vec<UVec4>,
            normals: &mut Vec<Vec4>,
            tangents: &mut Vec<Vec4>,
            uvs: &mut Vec<Vec2>,
            buffers: &[gltf::buffer::Data],
        ) {
            // Calculate node's global transformation matrix
            let local_transform = {
                let (translation, rotation, scale) = node.transform().decomposed();
                let translation = Mat4::from_translation(Vec3::new(
                    translation[0],
                    translation[1],
                    translation[2],
                ));
                let rotation = Mat4::from_quat(glam::Quat::from_array([
                    rotation[0],
                    rotation[1],
                    rotation[2],
                    rotation[3],
                ]));
                let scale = Mat4::from_scale(Vec3::new(scale[0], scale[1], scale[2]));
                translation * rotation * scale
            };

            let global_transform = parent_transform * local_transform;
            let (node_scale, node_quat, _) = global_transform.to_scale_rotation_translation();

            // Process mesh
            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let material_index = primitive.material().index().unwrap_or(0) as u32;
                    let triangle_offset = vertices.len() as u32;

                    let reader = primitive
                        .reader(|buffer| Some(&buffers[buffer.index()].0[..buffer.length()]));

                    // Process vertices
                    if let Some(iter) = reader.read_positions() {
                        for pos in iter {
                            let vert =
                                global_transform.mul_vec4(Vec4::new(pos[0], pos[1], pos[2], 1.0));
                            vertices.push(Vec4::new(vert.x, vert.z, vert.y, 1.0));
                        }
                    }

                    // Process normals
                    if let Some(iter) = reader.read_normals() {
                        for norm in iter {
                            let n = (node_quat
                                .mul_vec3(Vec3::new(norm[0], norm[1], norm[2]) / node_scale))
                            .normalize();
                            normals.push(Vec4::new(n.x, n.z, n.y, 0.0));
                        }
                    }

                    // Process tangents
                    if let Some(iter) = reader.read_tangents() {
                        for tan in iter {
                            let t = (node_quat
                                .mul_vec3(Vec3::new(tan[0], tan[1], tan[2]) / node_scale))
                            .normalize();
                            tangents.push(Vec4::new(t.x, t.z, t.y, 0.0));
                        }
                    }

                    // Process UV coordinates
                    if let Some(iter) = reader.read_tex_coords(0) {
                        match iter {
                            gltf::mesh::util::ReadTexCoords::F32(iter) => {
                                for uv in iter {
                                    uvs.push(Vec2::new(uv[0], uv[1]));
                                }
                            }
                            gltf::mesh::util::ReadTexCoords::U8(iter) => {
                                for uv in iter {
                                    uvs.push(Vec2::new(uv[0] as f32 / 255.0, uv[1] as f32 / 255.0));
                                }
                            }
                            gltf::mesh::util::ReadTexCoords::U16(iter) => {
                                for uv in iter {
                                    uvs.push(Vec2::new(
                                        uv[0] as f32 / 65535.0,
                                        uv[1] as f32 / 65535.0,
                                    ));
                                }
                            }
                        }
                    } else {
                        // If no UV coordinates, use zero values
                        for _ in 0..vertices.len() - uvs.len() {
                            uvs.push(Vec2::ZERO);
                        }
                    }

                    // Process indices
                    if let Some(indices_iter) = reader.read_indices() {
                        let indices_vec: Vec<u32> = indices_iter.into_u32().collect();
                        for i in (0..indices_vec.len()).step_by(3) {
                            if i + 2 < indices_vec.len() {
                                indices.push(UVec4::new(
                                    triangle_offset + indices_vec[i],
                                    triangle_offset + indices_vec[i + 2],
                                    triangle_offset + indices_vec[i + 1],
                                    material_index,
                                ));
                            }
                        }
                    }
                }
            }

            // Process child nodes recursively
            for child in node.children() {
                process_node(
                    &child,
                    global_transform,
                    vertices,
                    indices,
                    normals,
                    tangents,
                    uvs,
                    buffers,
                );
            }
        }

        // Process all root nodes in the scene
        for node in scene.nodes() {
            process_node(
                &node,
                Mat4::IDENTITY,
                &mut vertices,
                &mut indices,
                &mut normals,
                &mut tangents,
                &mut uvs,
                &buffers,
            );
        }

        // Process materials
        let textures = gltf.textures().collect::<Vec<_>>();
        let materials = gltf.materials().collect::<Vec<_>>();
        let mut material_datas = vec![MaterialData::default(); materials.len().max(1)];

        let mut texture_images = Vec::new();

        for (material_index, material) in materials.iter().enumerate() {
            let current_material_data: &mut MaterialData = &mut material_datas[material_index];

            // Process base color texture
            if let Some(info) = material.pbr_metallic_roughness().base_color_texture() {
                let texture = &textures[info.texture().index()];
                if let Some(image) = load_image_from_gltf(texture, &buffers) {
                    let mut rgb_img = image.into_rgb8();
                    // Convert gamma space to linear space
                    for pixel in rgb_img.iter_mut() {
                        *pixel = ((*pixel as f32 / 255.0).powf(2.2) * 255.0) as u8;
                    }
                    texture_images.push(image::DynamicImage::ImageRgb8(rgb_img));
                    current_material_data.set_has_albedo_texture(true);
                }
            }

            // Process metallic/roughness texture
            if let Some(info) = material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
            {
                let texture = &textures[info.texture().index()];
                if let Some(image) = load_image_from_gltf(texture, &buffers) {
                    texture_images.push(image.clone());
                    current_material_data.set_has_metallic_texture(true);

                    // Roughness is usually stored in a different channel of the same texture
                    texture_images.push(image);
                    current_material_data.set_has_roughness_texture(true);
                }
            }

            // Process normal texture
            if let Some(info) = material.normal_texture() {
                let texture = &textures[info.texture().index()];
                if let Some(image) = load_image_from_gltf(texture, &buffers) {
                    texture_images.push(image);
                    current_material_data.set_has_normal_texture(true);
                }
            }

            // Process base color factor
            let base_color = material.pbr_metallic_roughness().base_color_factor();
            current_material_data.albedo =
                Vec4::new(base_color[0], base_color[1], base_color[2], base_color[3]);

            // Process metallic factor
            let metallic_factor = material.pbr_metallic_roughness().metallic_factor();
            current_material_data.metallic = Vec4::splat(metallic_factor);

            // Process roughness factor
            let roughness_factor = material.pbr_metallic_roughness().roughness_factor();
            current_material_data.roughness = Vec4::splat(roughness_factor);

            // Process emissive factor
            let factor = material.emissive_factor();
            current_material_data.emissive = Vec4::new(factor[0], factor[1], factor[2], 1.0) * 15.0;
        }

        // Create texture atlas
        let (atlas_raw, mut sts) = crate::atlas::pack_textures(&texture_images, 4096, 4096);

        // Update texture coordinates in material data
        for material_data in material_datas.iter_mut() {
            if material_data.has_albedo_texture() {
                material_data.albedo = sts.remove(0);
            }
            if material_data.has_metallic_texture() {
                material_data.metallic = sts.remove(0);
            }
            if material_data.has_roughness_texture() {
                material_data.roughness = sts.remove(0);
            }
            if material_data.has_normal_texture() {
                material_data.normals = sts.remove(0);
            }
        }

        // Build BVH
        let now = std::time::Instant::now();
        let bvh = BVHBuilder::new(&vertices, &mut indices)
            .sah_samples(128)
            .build();
        #[cfg(debug_assertions)]
        println!("BVH build time: {:?}", now.elapsed());

        // Build light sampling table
        let now = std::time::Instant::now();
        let emissive_mask = light_pick::compute_emissive_mask(&indices, &material_datas);
        let light_pick_table = light_pick::build_light_pick_table(
            &vertices,
            &indices,
            &emissive_mask,
            &material_datas,
        );
        #[cfg(debug_assertions)]
        println!("Light pick table build time: {:?}", now.elapsed());

        // Pack vertex data
        let mut per_vertex_data = Vec::new();
        for i in 0..vertices.len() {
            per_vertex_data.push(PerVertexData {
                vertex: *vertices.get(i).unwrap_or(&Vec4::ZERO),
                normal: *normals.get(i).unwrap_or(&Vec4::ZERO),
                tangent: *tangents.get(i).unwrap_or(&Vec4::ZERO),
                uv0: *uvs.get(i).unwrap_or(&Vec2::ZERO),
                ..Default::default()
            });
        }

        Some(Self {
            bvh,
            per_vertex_buffer: per_vertex_data,
            index_buffer: indices,
            atlas: atlas_raw,
            material_data_buffer: material_datas,
            light_pick_buffer: light_pick_table,
        })
    }

    pub fn into_gpu<'fw>(self) -> GpuWorld<'fw> {
        GpuWorld {
            per_vertex_buffer: GpuBuffer::from_slice(&FW, &self.per_vertex_buffer),
            index_buffer: GpuBuffer::from_slice(&FW, &self.index_buffer),
            bvh: self.bvh.into_gpu(),
            atlas: GpuConstImage::from_bytes(&FW, &self.atlas.to_rgba8(), 4096, 4096),
            material_data_buffer: GpuBuffer::from_slice(&FW, &self.material_data_buffer),
            light_pick_buffer: GpuBuffer::from_slice(&FW, &self.light_pick_buffer),
        }
    }
}

pub fn load_dynamic_image(path: &str) -> Option<DynamicImage> {
    image::ImageReader::open(path).ok()?.decode().ok()
}

pub fn dynamic_image_to_gpu_image<'fw, P: PixelInfo>(img: DynamicImage) -> GpuConstImage<'fw, P> {
    let width = img.width();
    let height = img.height();
    match P::byte_size() {
        16 => GpuConstImage::from_bytes(
            &FW,
            bytemuck::cast_slice(&img.into_rgba32f()),
            width,
            height,
        ),
        _ => GpuConstImage::from_bytes(&FW, &img.into_rgba8(), width, height),
    }
}

pub fn dynamic_image_to_cpu_buffer<'img>(img: DynamicImage) -> Vec<Vec4> {
    let width = img.width();
    let height = img.height();
    let data = img.into_rgb8();
    let cpu_data: Vec<Vec4> = data
        .chunks(3)
        .map(|f| Vec4::new(f[0] as f32, f[1] as f32, f[2] as f32, 255.0) / 255.0)
        .collect();
    assert_eq!(cpu_data.len(), width as usize * height as usize);
    cpu_data
}

pub fn fallback_gpu_image<'fw>() -> GpuConstImage<'fw, Rgba32Float> {
    GpuConstImage::from_bytes(
        &FW,
        bytemuck::cast_slice(&[
            1.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
        ]),
        2,
        2,
    )
}

pub fn fallback_cpu_buffer() -> Vec<Vec4> {
    vec![
        Vec4::new(1.0, 0.0, 1.0, 1.0),
        Vec4::new(1.0, 0.0, 1.0, 1.0),
        Vec4::new(1.0, 0.0, 1.0, 1.0),
        Vec4::new(1.0, 0.0, 1.0, 1.0),
    ]
}
