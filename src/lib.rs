pub mod array;
pub mod graph;
pub mod io;
pub mod types;
pub mod vm;

mod bindings;

#[allow(non_snake_case)]
#[export_name = "PyInit__lib"]
pub unsafe extern "C" fn __python_init() -> *mut pyo3::ffi::PyObject {
    pyo3::impl_::trampoline::module_init(|py| {
        bindings::root::lib::_PYO3_DEF.make_module(py, bindings::root::lib::__PYO3_GIL_USED)
    })
}
