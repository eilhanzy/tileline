use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    mgs::render_benchmark::run_from_env()
}
