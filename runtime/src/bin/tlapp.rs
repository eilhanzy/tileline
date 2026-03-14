#[path = "../../examples/common/tlapp_app.rs"]
mod tlapp_app;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tlapp_app::run_from_env()
}
