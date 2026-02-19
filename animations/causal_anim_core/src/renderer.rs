//! Headless wgpu renderer.
//!
//! Renders nodes (instanced SDF circles) and edges (instanced line quads)
//! to an off-screen texture, reads back pixel data, and returns PNG bytes.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ─── GPU data types ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct NodeInstance {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct EdgeInstance {
    pub start: [f32; 3],
    pub width: f32,
    pub end: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

// ─── Shaders ─────────────────────────────────────────────────────────────────

const NODE_SHADER: &str = r#"
struct Camera { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0)       uv:   vec2<f32>,
    @location(1)       col:  vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32,
           @location(0) pos:    vec3<f32>,
           @location(1) radius: f32,
           @location(2) color:  vec4<f32>) -> VsOut {
    let x = f32(vi & 1u) * 2.0 - 1.0;
    let y = f32((vi >> 1u) & 1u) * 2.0 - 1.0;
    let world = pos + vec3<f32>(x * radius, y * radius, 0.0);
    var o: VsOut;
    o.clip = camera.view_proj * vec4<f32>(world, 1.0);
    o.uv   = vec2<f32>(x, y);
    o.col  = color;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if d > 1.0 { discard; }
    let alpha = 1.0 - smoothstep(0.8, 1.0, d);
    return vec4<f32>(in.col.rgb, in.col.a * alpha);
}
"#;

const EDGE_SHADER: &str = r#"
struct Camera { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0)       col:  vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32,
           @location(0) start: vec3<f32>,
           @location(1) width: f32,
           @location(2) end:   vec3<f32>,
           @location(4) color: vec4<f32>) -> VsOut {
    let dir  = end - start;
    let len  = length(dir);
    let fwd  = select(vec3<f32>(0.0, 1.0, 0.0), dir / len, len > 1e-6);
    let perp = normalize(vec3<f32>(-fwd.y, fwd.x, 0.0));
    let hw   = width * 0.5;
    let along = select(start, end, (vi & 2u) != 0u);
    let side  = select(-hw, hw, (vi & 1u) != 0u);
    let world = along + perp * side;
    var o: VsOut;
    o.clip = camera.view_proj * vec4<f32>(world, 1.0);
    o.col  = color;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.col;
}
"#;

// ─── Renderer ────────────────────────────────────────────────────────────────

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    node_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl Renderer {
    /// Create a headless renderer (no window required).
    pub fn new(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("causal_anim"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to create GPU device");

        // Render target.
        let tex_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: tex_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());

        // Camera uniform.
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });

        // --- Node pipeline ---
        let node_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("node_shader"),
            source: wgpu::ShaderSource::Wgsl(NODE_SHADER.into()),
        });
        let node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("node_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &node_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<NodeInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                        wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32 },
                        wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &node_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: tex_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Edge pipeline ---
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edge_shader"),
            source: wgpu::ShaderSource::Wgsl(EDGE_SHADER.into()),
        });
        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edge_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<EdgeInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 }, // start
                        wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32 },   // width
                        wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x3 }, // end
                        // offset 28: pad (skipped — location 3 unused)
                        wgpu::VertexAttribute { offset: 32, shader_location: 4, format: wgpu::VertexFormat::Float32x4 }, // color
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: tex_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Renderer {
            device,
            queue,
            width,
            height,
            texture,
            texture_view,
            node_pipeline,
            edge_pipeline,
            camera_buffer,
            camera_bind_group,
        }
    }

    /// Render one frame and return raw RGBA pixel bytes (row-major, top-left
    /// origin, width × height × 4).
    pub fn render_frame(
        &self,
        nodes: &[NodeInstance],
        edges: &[EdgeInstance],
        camera_center: [f32; 2],
        camera_zoom: f32,
        bg_color: [f64; 3],
    ) -> Vec<u8> {
        let aspect = self.width as f32 / self.height as f32;
        let cam = ortho(camera_center, camera_zoom, aspect);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam));

        let node_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nodes"),
            contents: bytemuck::cast_slice(nodes),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let edge_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edges"),
            contents: bytemuck::cast_slice(edges),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Readback buffer — row-aligned to 256 bytes (wgpu requirement).
        let bytes_per_row = align_to(self.width * 4, 256);
        let readback_size = (bytes_per_row * self.height) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg_color[0],
                            g: bg_color[1],
                            b: bg_color[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            // Edges first (drawn behind nodes).
            if !edges.is_empty() {
                pass.set_pipeline(&self.edge_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, edge_buf.slice(..));
                pass.draw(0..4, 0..edges.len() as u32);
            }

            // Nodes on top.
            if !nodes.is_empty() {
                pass.set_pipeline(&self.node_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, node_buf.slice(..));
                pass.draw(0..4, 0..nodes.len() as u32);
            }
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("GPU readback failed");

        let mapped = slice.get_mapped_range();

        // Strip row padding.
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
        let row_bytes = (self.width * 4) as usize;
        let padded_row = bytes_per_row as usize;
        for y in 0..self.height as usize {
            let start = y * padded_row;
            pixels.extend_from_slice(&mapped[start..start + row_bytes]);
        }
        pixels
    }

    /// Convenience: render and encode to PNG bytes.
    pub fn render_png(
        &self,
        nodes: &[NodeInstance],
        edges: &[EdgeInstance],
        camera_center: [f32; 2],
        camera_zoom: f32,
        bg_color: [f64; 3],
    ) -> Vec<u8> {
        let pixels = self.render_frame(nodes, edges, camera_center, camera_zoom, bg_color);
        let img = image::RgbaImage::from_raw(self.width, self.height, pixels)
            .expect("pixel buffer size mismatch");
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).expect("PNG encoding failed");
        buf.into_inner()
    }

    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn ortho(center: [f32; 2], zoom: f32, aspect: f32) -> CameraUniform {
    let hw = zoom * aspect;
    let hh = zoom;
    let l = center[0] - hw;
    let r = center[0] + hw;
    let b = center[1] - hh;
    let t = center[1] + hh;
    CameraUniform {
        view_proj: [
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-(r + l) / (r - l), -(t + b) / (t - b), 0.0, 1.0],
        ],
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) / alignment * alignment
}
