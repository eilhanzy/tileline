#[cfg(not(target_os = "android"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runtime::run_tlapp_from_env()
}

#[cfg(target_os = "android")]
fn main() {}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    if let Err(err) = runtime::run_tlapp_with_android_app(app) {
        eprintln!("failed to run tlapp on android: {err}");
    }
}
