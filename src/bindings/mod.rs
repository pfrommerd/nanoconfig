use pyo3::prelude::*;

mod utils;

#[pymodule]
#[pyo3(name = "picoml")]
pub mod root {
    use pyo3::prelude::*;
    #[pymodule]
    #[pyo3(name = "_lib")]
    pub mod lib {
        use super::super::utils::*;
        use pyo3::prelude::*;

        #[pyclass]
        struct RonFormat;

        #[pymethods]
        impl RonFormat {
            #[new]
            fn new() -> Self {
                RonFormat {}
            }
            fn serialize<'py>(&self, py: Python<'py>, obj: PyObject) -> PyDataBuffer {
                todo!()
            }
            fn serialize_to<'py>(&self, py: Python<'py>, obj: PyObject, file: PyIO) {
                todo!()
            }
            fn deserialize<'py>(&self, py: Python<'py>) {
                todo!()
            }
        }
    }
}
