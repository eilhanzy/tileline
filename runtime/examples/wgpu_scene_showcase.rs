fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[deprecated] 'wgpu_scene_showcase' is kept for compatibility. Use '--bin tlapp'.");
    runtime::run_tlapp_from_env()
}
