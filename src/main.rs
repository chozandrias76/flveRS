//! A simple 3D scene with light shining over a cube sitting on a plane.

use bevy::{input::{keyboard::{Key, KeyboardInput}, mouse::{MouseButton, MouseMotion}}, prelude::*, render::mesh::Capsule3dMeshBuilder};
use crate::helpers::flver_asset::{FlverAsset, FlverAssetLoaderLocal};
pub mod helpers;
extern crate dotenv;

use dotenv::dotenv;

#[derive(Resource, Default)]
struct State {
    model: Handle<FlverAsset>,
    printed: bool,
}

// Camera controller component
#[derive(Component)]
struct CameraController {
    // Movement speed
    pan_speed: f32,
    // Mouse sensitivity
    rotation_speed: f32,
    // Current pitch and yaw
    pitch: f32,
    yaw: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            pan_speed: 5.0,
            rotation_speed: 0.5,
            pitch: -0.1, // slight downward look by default
            yaw: 0.0,
        }
    }
}

// Mouse state resource to track right click state
#[derive(Resource, Default)]
struct MouseState {
    right_button_pressed: bool,
}

fn main() {
    dotenv().ok();

    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<State>()
        .init_resource::<MouseState>()
        .init_asset::<FlverAsset>()
        .init_asset_loader::<FlverAssetLoaderLocal>()
        .add_systems(Startup, setup)
        .add_systems(Update, print_on_load)
        // Add camera control systems
        .add_systems(Update, camera_keyboard_movement)
        .add_systems(Update, camera_mouse_rotation)
        .add_systems(Update, mouse_button_system)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut state: ResMut<State>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        CameraController::default(),
    ));
    let flver_path = std::env::var("TEST_FILE").unwrap();
    state.model = asset_server.load(flver_path.clone());
    
}

pub fn print_on_load(
    mut state: ResMut<State>,
    mut commands: Commands,
    flver_assets: Res<Assets<FlverAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let flver_asset = flver_assets.get(&state.model);

    // Can't process if the assets aren't ready
    if state.printed {
        return;
    }

    if let Some(flver) = flver_asset {
        info!("Flver asset loaded: {:#x}", flver as *const _ as usize);
        
        // Convert FLVER meshes to Bevy meshes
        let bevy_meshes = flver.to_bevy_meshes();
        
        // Create an entity for each mesh with a better material
        for (i, mesh) in bevy_meshes.into_iter().enumerate() {
            let mesh_handle = meshes.add(mesh);
            
            // Create double-sided material with good default properties for visibility
            let material = materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.8, 0.8), // Light gray as default
                reflectance: 0.1, // Slightly reflective
                perceptual_roughness: 0.4, // Smoother surface
                metallic: 0.1, // Slightly metallic
                cull_mode: None, // Important: Disable culling to see both sides
                ..default()
            });
            
            // Spawn mesh with the new API components
            commands.spawn((
                Mesh3d(mesh_handle),
                MeshMaterial3d(material),
                Transform::from_xyz(0., 0., 0.)
                    .with_scale(Vec3::splat(0.5)), // Scale down to ensure it's in view
                GlobalTransform::default(),
                Visibility::default(),
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
            
            info!("Spawned mesh {}", i);
        }

        // Add a point light to better illuminate the model
        commands.spawn((
            PointLight {
                intensity: 1500.0,
                shadows_enabled: true,
                range: 100.0,
                radius: 0.0,
                color: Color::WHITE,
                ..default()
            },
            Transform::from_xyz(4.0, 6.0, 4.0),
        ));

        // Add a second light from another angle to reduce shadows
        commands.spawn((
            PointLight {
                intensity: 1000.0,
                shadows_enabled: true,
                range: 100.0,
                radius: 0.0,
                color: Color::srgb(0.9, 0.9, 1.0), // Slight blue tint
                ..default()
            },
            Transform::from_xyz(-4.0, 5.0, -2.0),
        ));
        // Add ambient light to fill shadows
        commands.insert_resource(AmbientLight {
            color: Color::srgb(0.2, 0.2, 0.25), // Subtle ambient light
            brightness: 0.3,
        });

        // Add coordinate cross to help with orientation
        spawn_coordinate_cross(&mut commands, &mut meshes, &mut materials);

        // Once processed, we won't process again
        state.printed = true;
    } else {
        info!("Flver Asset Not Ready");
    }
}

// Helper function to spawn coordinate cross to understand model orientation
fn spawn_coordinate_cross(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let axis_mesh = Capsule3dMeshBuilder::new(0.005, 0.20, 16, 16).build();
    // X-axis (red)
    commands.spawn((
        Mesh3d(meshes.add(axis_mesh.clone())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::LinearRgba(LinearRgba { red: 255., green: 0., blue: 0., alpha: 0.625 }),
            emissive: Color::srgba(1.0, 0.0, 0.0, 0.5).into(),
            ..default()
        })),
        Transform::from_xyz(1.0, 0.0, 0.0)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
    ));

    // Y-axis (green)
    commands.spawn((
        Mesh3d(meshes.add(axis_mesh.clone())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::LinearRgba(LinearRgba { red: 0., green: 255., blue: 0., alpha: 0.625 }),

            emissive: Color::srgba(0.0, 1.0, 0.0, 0.5).into(),
            ..default()
        })),
        Transform::from_xyz(0.0, 1.0, 0.0),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
    ));

    // Z-axis (blue)
    commands.spawn((
        Mesh3d(meshes.add(axis_mesh.clone())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::LinearRgba(LinearRgba { red: 0., green: 0., blue: 255., alpha: 0.625 }),
            emissive: Color::srgba(0.0, 0.0, 1.0, 0.5).into(),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 1.0)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
    ));
}

// System to handle mouse button state
fn mouse_button_system(
    mut mouse_state: ResMut<MouseState>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
) {
    mouse_state.right_button_pressed = mouse_button_input.pressed(MouseButton::Right);
}

// System to handle keyboard movement with WASD keys
fn camera_keyboard_movement(
    time: Res<Time>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&CameraController, &mut Transform)>,
) {
    for (controller, mut transform) in query.iter_mut() {
        let mut direction = Vec3::ZERO;

        // Forward/backward movement (W/S)
        if keyboard_input.pressed(KeyCode::KeyW) {
            direction += Vec3::from(transform.forward());
        }
        if keyboard_input.pressed(KeyCode::KeyS) {
            direction += Vec3::from(transform.back());
        }

        // Left/right movement (A/D)
        if keyboard_input.pressed(KeyCode::KeyA) {
            direction += Vec3::from(transform.left());
        }
        if keyboard_input.pressed(KeyCode::KeyD) {
            direction += Vec3::from(transform.right());
        }

        // Apply movement
        if direction != Vec3::ZERO {
            // Normalize direction to ensure consistent speed in all directions
            direction = direction.normalize();
            transform.translation += direction * controller.pan_speed * time.delta_secs();
        }
    }
}

// System to handle mouse rotation when right click is held
fn camera_mouse_rotation(
    mouse_state: Res<MouseState>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut query: Query<(&mut CameraController, &mut Transform)>,
) {
    // Only process mouse motion if right click is held
    if !mouse_state.right_button_pressed {
        // Clear events, but don't process
        mouse_motion_events.clear();
        return;
    }
    
    // Get accumulated mouse delta
    let mut delta = Vec2::ZERO;
    for event in mouse_motion_events.read() {
        delta += event.delta;
    }
    
    // Apply rotation to each camera with the controller component
    for (mut controller, mut transform) in query.iter_mut() {
        if delta != Vec2::ZERO {
            // Update pitch and yaw based on mouse movement
            controller.yaw -= delta.x * controller.rotation_speed * 0.002;  // horizontal
            controller.pitch -= delta.y * controller.rotation_speed * 0.002; // vertical
            
            // Clamp pitch to prevent camera from flipping
            controller.pitch = controller.pitch.clamp(-1.54, 1.54); // ~88 degrees
            
            // Create rotation quaternion from yaw and pitch
            let yaw_quat = Quat::from_rotation_y(controller.yaw);
            let pitch_quat = Quat::from_rotation_x(controller.pitch);
            
            // Apply combined rotation
            transform.rotation = yaw_quat * pitch_quat;
        }
    }
}
