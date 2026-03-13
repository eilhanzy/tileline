use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    gms::render_benchmark::run_from_env()
}
