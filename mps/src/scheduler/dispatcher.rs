//! Dispatcher for native and WASM jobs.
//! WASM modules are compiled and executed in-memory through Wasmer.

use super::{TaskPayload, WasmTask};
use wasmer::{imports, Instance, Module, Store, Value};

/// Dispatcher-level result type.
pub type DispatchResult<T = ()> = Result<T, DispatchError>;

/// Dispatch errors for native/WASM execution.
#[derive(Debug, Clone)]
pub enum DispatchError {
    Compile(String),
    Instantiate(String),
    MissingExport(String),
    Runtime(String),
}

/// Stateless dispatcher.
///
/// The implementation currently compiles WASM modules on demand.
/// A lock-free module cache can be layered on top in a later phase.
#[derive(Debug, Default)]
pub struct Dispatcher;

impl Dispatcher {
    /// Build a new dispatcher.
    pub fn new() -> Self {
        Self
    }

    /// Execute a queued task payload.
    pub fn execute(&self, payload: TaskPayload) -> DispatchResult {
        match payload {
            TaskPayload::Native(task) => {
                task();
                Ok(())
            }
            TaskPayload::Wasm(wasm_task) => self.execute_wasm(wasm_task),
        }
    }

    fn execute_wasm(&self, wasm_task: WasmTask) -> DispatchResult {
        // Wasmer's default engine provides JIT/AOT capabilities depending on platform.
        let mut store = Store::default();

        let module = Module::new(&store, wasm_task.module_bytes.as_ref())
            .map_err(|err| DispatchError::Compile(err.to_string()))?;

        let import_object = imports! {};
        let instance = Instance::new(&mut store, &module, &import_object)
            .map_err(|err| DispatchError::Instantiate(err.to_string()))?;

        let entrypoint = wasm_task.entrypoint;
        let function = instance
            .exports
            .get_function(&entrypoint)
            .map_err(|_| DispatchError::MissingExport(entrypoint.clone()))?;

        let params: Vec<Value> = wasm_task.args.into_iter().map(Value::I64).collect();
        function
            .call(&mut store, &params)
            .map_err(|err| DispatchError::Runtime(err.to_string()))?;

        Ok(())
    }
}
