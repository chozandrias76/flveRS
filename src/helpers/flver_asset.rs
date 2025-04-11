use bevy::{
    asset::{io::Reader, AssetLoader, LoadContext, RenderAssetUsages},
    prelude::*,
    reflect::TypePath,
    render::mesh::Indices,
};
use byteorder::ReadBytesExt;
use byteorder::LE;
use fstools_formats::flver::reader::{
    FLVER, FLVERFaceSet, FLVERFaceSetIndices, FLVERMesh, VertexBuffer, VertexBufferLayout,
    VertexAttributeSemantic, FLVERPartContext
};
use std::io::Seek;
use thiserror::Error;
use log::{debug, info, trace, warn};

#[derive(Asset, TypePath)]
pub struct FlverAsset {
    flver: FLVER,
}

#[derive(Default)]
pub struct FlverAssetLoaderLocal;

/// Possible errors that can be produced by [`FlverAssetLoader`]
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum FlverAssetLoaderError {
    /// An [IO](std::io) Error
    #[error("Could not load asset: {0}")]
    Io(#[from] std::io::Error),
}

impl AssetLoader for FlverAssetLoaderLocal {
    type Asset = FlverAsset;
    type Settings = ();
    type Error = FlverAssetLoaderError;
    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        
        // Create a cursor that implements Read+Seek over our bytes
        let mut cursor = std::io::Cursor::new(bytes);
        
        // Use FLVER::from_reader with our cursor
        let flver = FLVER::from_reader(&mut cursor)?;
        
        // Create the FlverAsset
        Ok(FlverAsset { flver })
    }

    fn extensions(&self) -> &[&str] {
        &["flver"]
    }
}

impl FlverAsset {
    pub fn flver(&self) -> &FLVER {
        &self.flver
    }
    
    // Convert all meshes to Bevy meshes
    pub fn to_bevy_meshes(&self) -> Vec<bevy::render::mesh::Mesh> {
        let mut meshes = Vec::new();
        
        info!("Converting {} FLVER meshes to Bevy format", self.flver.meshes.len());
        
        for (mesh_idx, mesh) in self.flver.meshes.iter().enumerate() {
            info!("Processing mesh {}/{}", 
                mesh_idx + 1, self.flver.meshes.len());
            
            if let Some(bevy_mesh) = self.mesh_to_bevy_mesh(mesh) {
                info!("Successfully converted mesh {} to Bevy format", mesh_idx);
                meshes.push(bevy_mesh);
            } else {
                warn!("Failed to convert mesh {} to Bevy format - skipping", mesh_idx);
            }
        }
        
        info!("Converted {} out of {} meshes successfully", meshes.len(), self.flver.meshes.len());
        meshes
    }
    
    // Helper method to get vertex buffer objects referenced by a mesh
    fn get_mesh_buffers(&self, mesh: &FLVERMesh) -> Vec<&VertexBuffer> {
        mesh.vertex_buffer_indices.iter()
            .filter_map(|&idx| self.flver.vertex_buffers.get(idx as usize))
            .collect()
    }
    
    // Helper method to get face sets referenced by a mesh
    fn get_mesh_face_sets(&self, mesh: &FLVERMesh) -> Vec<&FLVERFaceSet> {
        mesh.face_set_indices.iter()
            .filter_map(|&idx| self.flver.face_sets.get(idx as usize))
            .collect()
    }
    
    // Helper method to get vertex attributes from a layout
    fn get_vertex_attributes<'a>(&self, layout: &'a VertexBufferLayout) -> Vec<(&'a VertexAttributeSemantic, u32)> {
        layout.members.iter()
            .map(|member| (&member.semantic, member.struct_offset))
            .collect()
    }
    
    // Helper method to extract vertex data from a buffer based on the attribute semantic
    fn extract_vertex_data<T, F>(&self, 
                                buffer: &VertexBuffer, 
                                layout: &VertexBufferLayout,
                                semantic: VertexAttributeSemantic,
                                transform: F,
                                output: &mut Vec<T>) 
                                where F: Fn(&[u8], u32) -> Option<T> {
        let _context = FLVERPartContext { data_offset: self.flver.data_offset };
        
        // Find member with matching semantic
        if let Some(member) = layout.members.iter().find(|m| m.semantic == semantic) {
            debug!("Found attribute with semantic {:?} at offset {}", semantic, member.struct_offset);
            
            // Process each vertex
            for i in 0..buffer.vertex_count {
                let offset = buffer.buffer_offset + (i * buffer.vertex_size) + member.struct_offset;
                let vertex_data = transform(&[], offset);
                if let Some(data) = vertex_data {
                    output.push(data);
                }
            }
        }
    }
    
    // Convert a specific mesh to a Bevy mesh
    pub fn mesh_to_bevy_mesh(&self, flver_mesh: &FLVERMesh) -> Option<bevy::render::mesh::Mesh> {
        debug!("Converting mesh to Bevy format: starting conversion process");
        
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut colors = Vec::new();
        let mut bone_weights = Vec::new();
        let mut bone_indices = Vec::new();
        let mut indices = Vec::new();
        
        // Process vertex buffers for this mesh
        debug!("Collecting vertex buffers for mesh");
        let buffers = self.get_mesh_buffers(flver_mesh);
        debug!("Found {} vertex buffers for mesh", buffers.len());
        
        // Find vertex counts to preallocate vectors
        let max_vertices = buffers.iter()
            .map(|b| b.vertex_count as usize)
            .sum();
        
        debug!("Preallocating space for up to {} vertices", max_vertices);
        positions.reserve(max_vertices);
        normals.reserve(max_vertices);
        uvs.reserve(max_vertices);
        
        // Create a cursor for accessing raw buffer data
        let mut cursor = std::io::Cursor::new(Vec::new());
        // Reuse cursor for all data accesses
        if let Ok(bytes) = std::fs::read(std::env::var("TEST_FILE").unwrap_or_default()) {
            cursor = std::io::Cursor::new(bytes);
            
            for (buffer_idx, buffer) in buffers.iter().enumerate() {
                debug!("Processing vertex buffer {}/{} with {} vertices", 
                    buffer_idx + 1, buffers.len(), buffer.vertex_count);
                
                // Get the layout index for this buffer
                let layout_idx = buffer.layout_index as usize;
                debug!("Buffer uses layout index: {}", layout_idx);
                
                // Find the layout for this buffer (safely)
                if layout_idx < self.flver.buffer_layouts.len() {
                    let layout = &self.flver.buffer_layouts[layout_idx];
                    debug!("Found vertex buffer layout with {} members", layout.members.len());
                    
                    // For each vertex in the buffer
                    for vertex_idx in 0..buffer.vertex_count {
                        let mut vertex_position = [0.0f32, 0.0f32, 0.0f32];
                        let mut vertex_normal = [0.0f32, 0.0f32, 1.0f32]; // Default to +Z
                        let mut vertex_uv = [0.0f32, 0.0f32];
                        let mut vertex_tangent = [1.0f32, 0.0f32, 0.0f32, 1.0f32]; // Default tangent +X with +1 handedness
                        let mut vertex_color = [1.0f32, 1.0f32, 1.0f32, 1.0f32]; // Default white
                        let mut vertex_bone_weights = [1.0f32, 0.0f32, 0.0f32, 0.0f32]; // Default first bone full weight
                        let mut vertex_bone_indices = [0u16, 0, 0, 0]; // Default to bone 0
                        
                        // For each attribute in the layout
                        for member in &layout.members {
                            // Calculate the offset for this vertex attribute
                            let attr_offset = self.flver.data_offset + buffer.buffer_offset + 
                                             (vertex_idx * buffer.vertex_size) + member.struct_offset;
                            
                            debug!("Processing vertex {} attribute {:?} at offset {}", 
                                vertex_idx, member.semantic, attr_offset);
                            
                            // Use the cursor to read data at the correct offset
                            if let Ok(_) = cursor.seek(std::io::SeekFrom::Start(attr_offset as u64)) {
                                // Extract the data based on the semantic
                                match member.semantic {
                                    VertexAttributeSemantic::Position => {
                                        // Handle position based on format (probably Float3)
                                        if let (Ok(x), Ok(y), Ok(z)) = (
                                            cursor.read_f32::<LE>(),
                                            cursor.read_f32::<LE>(),
                                            cursor.read_f32::<LE>()
                                        ) {
                                            vertex_position = [x, y, z];
                                        }
                                    },
                                    VertexAttributeSemantic::Normal => {
                                        // Handle normal based on format
                                        if let (Ok(x), Ok(y), Ok(z)) = (
                                            cursor.read_f32::<LE>(),
                                            cursor.read_f32::<LE>(),
                                            cursor.read_f32::<LE>()
                                        ) {
                                            vertex_normal = [x, y, z];
                                        }
                                    },
                                    VertexAttributeSemantic::UV => {
                                        // Handle UV based on format
                                        if let (Ok(u), Ok(v)) = (
                                            cursor.read_f32::<LE>(),
                                            cursor.read_f32::<LE>()
                                        ) {
                                            vertex_uv = [u, v];
                                        }
                                    },
                                    VertexAttributeSemantic::Tangent => {
                                        // Handle tangent vectors
                                        debug!("Processing tangent attribute");
                                        match member.format {
                                            // For Float4 format
                                            0x3 => {
                                                if let (Ok(x), Ok(y), Ok(z), Ok(w)) = (
                                                    cursor.read_f32::<LE>(),
                                                    cursor.read_f32::<LE>(),
                                                    cursor.read_f32::<LE>(),
                                                    cursor.read_f32::<LE>()
                                                ) {
                                                    vertex_tangent = [x, y, z, w];
                                                }
                                            },
                                            // For normalized byte formats
                                            0x10 | 0x11 | 0x13 => {
                                                if let (Ok(x), Ok(y), Ok(z), Ok(w)) = (
                                                    cursor.read_u8().map(|v| (v as f32 / 127.5) - 1.0),
                                                    cursor.read_u8().map(|v| (v as f32 / 127.5) - 1.0),
                                                    cursor.read_u8().map(|v| (v as f32 / 127.5) - 1.0),
                                                    cursor.read_u8().map(|v| (v as f32 / 127.5) - 1.0)
                                                ) {
                                                    vertex_tangent = [x, y, z, w];
                                                }
                                            },
                                            // For short-to-float formats
                                            0x1A | 0x2E => {
                                                if let (Ok(x), Ok(y), Ok(z), Ok(w)) = (
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0),
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0),
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0),
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0)
                                                ) {
                                                    vertex_tangent = [x, y, z, w];
                                                }
                                            },
                                            _ => {
                                                warn!("Unsupported tangent format: 0x{:x}", member.format);
                                            }
                                        }
                                    },
                                    VertexAttributeSemantic::Bitangent => {
                                        // Skip bitangent - can be computed from normal and tangent
                                    },
                                    VertexAttributeSemantic::VertexColor => {
                                        // Handle vertex colors
                                        debug!("Processing vertex color attribute");
                                        match member.format {
                                            // For RGBA byte formats
                                            0x10 | 0x11 => {
                                                if let (Ok(r), Ok(g), Ok(b), Ok(a)) = (
                                                    cursor.read_u8().map(|v| v as f32 / 255.0),
                                                    cursor.read_u8().map(|v| v as f32 / 255.0),
                                                    cursor.read_u8().map(|v| v as f32 / 255.0),
                                                    cursor.read_u8().map(|v| v as f32 / 255.0)
                                                ) {
                                                    vertex_color = [r, g, b, a];
                                                }
                                            },
                                            _ => {
                                                warn!("Unsupported color format: 0x{:x}", member.format);
                                            }
                                        }
                                    },
                                    VertexAttributeSemantic::BoneWeights => {
                                        // Handle bone weights
                                        debug!("Processing bone weights attribute");
                                        match member.format {
                                            // Float4 format (full precision weights)
                                            0x3 => {
                                                if let (Ok(w1), Ok(w2), Ok(w3), Ok(w4)) = (
                                                    cursor.read_f32::<LE>(),
                                                    cursor.read_f32::<LE>(),
                                                    cursor.read_f32::<LE>(),
                                                    cursor.read_f32::<LE>()
                                                ) {
                                                    vertex_bone_weights = [w1, w2, w3, w4];
                                                }
                                            },
                                            // Byte4 formats (normalized)
                                            0x10 | 0x11 => {
                                                if let (Ok(w1), Ok(w2), Ok(w3), Ok(w4)) = (
                                                    cursor.read_u8().map(|v| v as f32 / 255.0),
                                                    cursor.read_u8().map(|v| v as f32 / 255.0),
                                                    cursor.read_u8().map(|v| v as f32 / 255.0),
                                                    cursor.read_u8().map(|v| v as f32 / 255.0)
                                                ) {
                                                    vertex_bone_weights = [w1, w2, w3, w4];
                                                }
                                            },
                                            // Format 0x13 - normalized byte to float
                                            0x13 => {
                                                if let (Ok(w1), Ok(w2), Ok(w3), Ok(w4)) = (
                                                    cursor.read_u8().map(|v| v as f32 / 127.0),
                                                    cursor.read_u8().map(|v| v as f32 / 127.0),
                                                    cursor.read_u8().map(|v| v as f32 / 127.0),
                                                    cursor.read_u8().map(|v| v as f32 / 127.0)
                                                ) {
                                                    vertex_bone_weights = [w1, w2, w3, w4];
                                                }
                                            },
                                            // Short4 formats (normalized)
                                            0x1A | 0x2E => {
                                                if let (Ok(w1), Ok(w2), Ok(w3), Ok(w4)) = (
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0),
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0),
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0),
                                                    cursor.read_i16::<LE>().map(|v| v as f32 / 32767.0)
                                                ) {
                                                    vertex_bone_weights = [w1, w2, w3, w4];
                                                }
                                            },
                                            _ => {
                                                warn!("Unsupported bone weight format: 0x{:x}", member.format);
                                            }
                                        }
                                    },
                                    VertexAttributeSemantic::BoneIndices => {
                                        // Handle bone indices
                                        debug!("Processing bone indices attribute");
                                        match member.format {
                                            // UByte4 format for bone indices
                                            0x10 | 0x11 => {
                                                if let (Ok(i1), Ok(i2), Ok(i3), Ok(i4)) = (
                                                    cursor.read_u8().map(|v| v as u16),
                                                    cursor.read_u8().map(|v| v as u16),
                                                    cursor.read_u8().map(|v| v as u16),
                                                    cursor.read_u8().map(|v| v as u16)
                                                ) {
                                                    vertex_bone_indices = [i1, i2, i3, i4];
                                                }
                                            },
                                            // Short4 format for bone indices
                                            0x18 => {
                                                if let (Ok(i1), Ok(i2), Ok(i3), Ok(i4)) = (
                                                    cursor.read_u16::<LE>(),
                                                    cursor.read_u16::<LE>(),
                                                    cursor.read_u16::<LE>(),
                                                    cursor.read_u16::<LE>()
                                                ) {
                                                    vertex_bone_indices = [i1, i2, i3, i4];
                                                }
                                            },
                                            _ => {
                                                warn!("Unsupported bone indices format: 0x{:x}", member.format);
                                            }
                                        }
                                    },
                                }
                            } else {
                                warn!("Failed to seek to offset {} for vertex attribute", attr_offset);
                            }
                        }
                        
                        // After processing all attributes for this vertex, add them to our vectors
                        positions.push(vertex_position);
                        normals.push(vertex_normal);
                        uvs.push(vertex_uv);
                        tangents.push(vertex_tangent);
                        colors.push(vertex_color);
                        bone_weights.push(vertex_bone_weights);
                        bone_indices.push(vertex_bone_indices);
                    }
                } else {
                    warn!("Invalid layout index {} (max: {}) - skipping buffer", 
                          layout_idx, self.flver.buffer_layouts.len() - 1);
                }
            }
        } else {
            warn!("Could not open FLVER file for reading vertex data");
        }
        
        // Process face sets (indices) for this mesh
        let face_sets = self.get_mesh_face_sets(flver_mesh);
        debug!("Processing {} face sets for mesh", face_sets.len());
        
        for (face_set_idx, face_set) in face_sets.iter().enumerate() {
            debug!("Processing face set {}/{}", 
                face_set_idx + 1, face_sets.len());
            
            match &face_set.indices {
                FLVERFaceSetIndices::Byte0 => {
                    warn!("Face set has no indices (FaceSetIndices::Byte0) - skipping");
                },
                FLVERFaceSetIndices::Byte1(indices_slice) => {
                    debug!("Processing U8 indices: {} total values ({} triangles)", 
                        indices_slice.len(), indices_slice.len() / 3);
                    
                    let start_len = indices.len();
                    for i in (0..indices_slice.len()).step_by(3) {
                        if i + 2 < indices_slice.len() {
                            indices.push(indices_slice[i] as u32);
                            indices.push(indices_slice[i+1] as u32);
                            indices.push(indices_slice[i+2] as u32);
                            trace!("Triangle: indices[{}, {}, {}]", 
                                indices_slice[i], indices_slice[i+1], indices_slice[i+2]);
                        } else {
                            warn!("Incomplete triangle at end of indices array (idx {}), skipping", i);
                        }
                    }
                    debug!("Added {} triangles from U8 indices", (indices.len() - start_len) / 3);
                },
                FLVERFaceSetIndices::Byte2(indices_slice) => {
                    debug!("Processing U16 indices: {} total values ({} triangles)", 
                        indices_slice.len(), indices_slice.len() / 3);
                    
                    let start_len = indices.len();
                    for i in (0..indices_slice.len()).step_by(3) {
                        if i + 2 < indices_slice.len() {
                            indices.push(indices_slice[i] as u32);
                            indices.push(indices_slice[i+1] as u32);
                            indices.push(indices_slice[i+2] as u32);
                            trace!("Triangle: indices[{}, {}, {}]", 
                                indices_slice[i], indices_slice[i+1], indices_slice[i+2]);
                        } else {
                            warn!("Incomplete triangle at end of indices array (idx {}), skipping", i);
                        }
                    }
                    debug!("Added {} triangles from U16 indices", (indices.len() - start_len) / 3);
                },
                FLVERFaceSetIndices::Byte4(indices_slice) => {
                    debug!("Processing U32 indices: {} total values ({} triangles)", 
                        indices_slice.len(), indices_slice.len() / 3);
                    
                    let start_len = indices.len();
                    for i in (0..indices_slice.len()).step_by(3) {
                        if i + 2 < indices_slice.len() {
                            indices.push(indices_slice[i]);
                            indices.push(indices_slice[i+1]);
                            indices.push(indices_slice[i+2]);
                            trace!("Triangle: indices[{}, {}, {}]", 
                                indices_slice[i], indices_slice[i+1], indices_slice[i+2]);
                        } else {
                            warn!("Incomplete triangle at end of indices array (idx {}), skipping", i);
                        }
                    }
                    debug!("Added {} triangles from U32 indices", (indices.len() - start_len) / 3);
                },
            }
        }
        
        // Create and populate a Bevy mesh
        debug!("Creating Bevy mesh with {} vertices and {} indices ({} triangles)", 
            positions.len(), indices.len(), indices.len() / 3);
        
        let mut mesh = bevy::render::mesh::Mesh::new(bevy::render::mesh::PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
        
        if !positions.is_empty() {
            debug!("Adding {} position vertices to mesh", positions.len());
            mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION, positions);
        } else {
            warn!("No position data available - cannot create mesh");
            return None; // Can't create a mesh without positions
        }
        
        if !normals.is_empty() {
            debug!("Adding {} normals to mesh", normals.len());
            mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_NORMAL, normals);
        } else {
            debug!("No normal data available - mesh will have default normals");
        }
        
        if !uvs.is_empty() {
            debug!("Adding {} UV coordinates to mesh", uvs.len());
            mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_UV_0, uvs);
        } else {
            debug!("No UV data available - mesh will have default UVs");
        }
        
        if !tangents.is_empty() {
            debug!("Adding {} tangents to mesh", tangents.len());
            mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_TANGENT, tangents);
        } else {
            debug!("No tangent data available - mesh will have default tangents");
        }
        
        if !colors.is_empty() {
            debug!("Adding {} vertex colors to mesh", colors.len());
            mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_COLOR, colors);
        } else {
            debug!("No vertex color data available");
        }
        
        // Add skinning data if available
        if !bone_weights.is_empty() && !bone_indices.is_empty() {
            debug!("Adding skinning data to mesh: {} weights, {} indices", 
                bone_weights.len(), bone_indices.len());
            // mesh.insert_attribute("Weights", bone_weights);
            // mesh.insert_attribute("Joints", bone_indices);
        }
        
        if !indices.is_empty() {
            debug!("Adding {} indices ({} triangles) to mesh", indices.len(), indices.len() / 3);
            mesh.insert_indices(Indices::U32(indices));
        } else {
            warn!("No indices data - mesh may not render properly");
        }
        
        info!("Successfully created Bevy mesh");
        Some(mesh)
    }
}