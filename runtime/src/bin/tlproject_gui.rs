#[cfg(not(target_os = "android"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runtime::run_tlproject_gui_from_env()
}

#[cfg(target_os = "android")]
fn main() {}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    if let Err(err) = runtime::run_tlproject_gui_with_android_app(app) {
        eprintln!("failed to run tlproject_gui on android: {err}");
    }
}
