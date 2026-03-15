//! Platform-aware winit app runner helpers.
//!
//! This module centralizes event-loop creation so runtime entrypoints can share one code path on
//! desktop and Android.

use std::error::Error;

use winit::application::ApplicationHandler;
use winit::event_loop::EventLoop;

type BoxError = Box<dyn Error>;

/// Run an application on desktop targets with a standard event loop.
pub fn run_app_desktop<A>(mut app: A) -> Result<(), BoxError>
where
    A: ApplicationHandler,
{
    let event_loop = EventLoop::builder().build()?;
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Run an application on Android with an `AndroidApp`-bound event loop.
#[cfg(target_os = "android")]
pub fn run_app_android<A>(
    android_app: winit::platform::android::activity::AndroidApp,
    mut app: A,
) -> Result<(), BoxError>
where
    A: ApplicationHandler,
{
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let mut event_loop_builder = EventLoop::builder();
    event_loop_builder.with_android_app(android_app);
    let event_loop = event_loop_builder.build()?;
    event_loop.run_app(&mut app)?;
    Ok(())
}
